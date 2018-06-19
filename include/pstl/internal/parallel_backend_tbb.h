/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __PSTL_parallel_backend_tbb_H
#define __PSTL_parallel_backend_tbb_H

#include <cassert>

#include "parallel_backend_utils.h"

// Bring in minimal required subset of Intel TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>
#include <tbb/tbb_allocator.h>

#if TBB_INTERFACE_VERSION < 10000
#error Intel(R) Threading Building Blocks 2018 is required; older versions are not supported.
#endif

namespace pstl {
namespace par_backend {

//! Raw memory buffer with automatic freeing and no exceptions.
/** Some of our algorithms need to start with raw memory buffer,
not an initialize array, because initialization/destruction
would make the span be at least O(N). */
// tbb::allocator can improve performance in some cases.
template<typename T>
class buffer {
    tbb::tbb_allocator<T> allocator;
    T* ptr;
    const std::size_t buf_size;
    buffer(const buffer&) = delete;
    void operator=(const buffer&) = delete;
public:
    //! Try to obtain buffer of given size to store objects of T type
    buffer(std::size_t n) : allocator(), ptr(allocator.allocate(n)), buf_size(n) {}
    //! True if buffer was successfully obtained, zero otherwise.
    operator bool() const { return ptr != NULL; }
    //! Return pointer to buffer, or  NULL if buffer could not be obtained.
    T* get() const { return ptr; }
    //! Destroy buffer
    ~buffer() { allocator.deallocate(ptr, buf_size); }
};

// Wrapper for tbb::task
inline void cancel_execution() {
    tbb::task::self().group()->cancel_group_execution();
}

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

template <class Index, class RealBody>
class parallel_for_body {
public:
    parallel_for_body( const RealBody& body) : my_body( body ) { }
    parallel_for_body(const parallel_for_body& body): my_body(body.my_body) { }
    void operator()(const tbb::blocked_range<Index>& range) const {
        my_body(range.begin(), range.end());
    }
private:
    RealBody my_body;
};

//! Evaluation of brick f[i,j) for each subrange [i,j) of [first,last)
// wrapper over tbb::parallel_for
template<class Index, class F>
void parallel_for(Index first, Index last, F f) {
    tbb::this_task_arena::isolate([=]() {
        tbb::parallel_for(tbb::blocked_range<Index>(first, last), parallel_for_body<Index, F>(f));
    });
}

//! Evaluation of brick f[i,j) for each subrange [i,j) of [first,last)
// wrapper over tbb::parallel_reduce
template<class Value, class Index, typename RealBody, typename Reduction>
Value parallel_reduce(Index first, Index last, const Value& identity, const RealBody& real_body, const Reduction& reduction) {
    return tbb::this_task_arena::isolate([first, last, &identity, &real_body, &reduction]()->Value {
        return tbb::parallel_reduce(tbb::blocked_range<Index>(first, last), identity,
            [real_body](const tbb::blocked_range<Index>& r, const Value& value)-> Value {
            return real_body(r.begin(), r.end(), value);
        },
        reduction);
    });
}

//------------------------------------------------------------------------
// parallel_transform_reduce
//
// Notation:
//      r(i,j,init) returns reduction of init with reduction over [i,j)
//      u(i) returns f(i,i+1,identity) for a hypothetical left identity element of r
//      c(x,y) combines values x and y that were the result of r or u
//------------------------------------------------------------------------

template<class Index, class U, class T, class C, class R>
struct par_trans_red_body {
    alignas(T) char sum_storage[sizeof(T)]; // Holds generalized non-commutative sum when has_sum==true
    R brick_reduce;                 // Most likely to have non-empty layout
    U u;
    C combine;
    bool has_sum;             // Put last to minimize size of class
    T& sum() {
        __TBB_ASSERT(has_sum, "sum expected");
        return *(T*)sum_storage;
    }
    par_trans_red_body( U u_, T init, C c_, R r_) :
        brick_reduce(r_),
        u(u_),
        combine(c_),
        has_sum(true) {
        new(sum_storage) T(init);
    }
    par_trans_red_body( par_trans_red_body& left, tbb::split ) :
        brick_reduce(left.brick_reduce),
        u(left.u),
        combine(left.combine),
        has_sum(false)
    {}
    ~par_trans_red_body() {
        // 17.6.5.12 tells us to not worry about catching exceptions from destructors.
        if( has_sum )
            sum().~T();
    }
    void join(par_trans_red_body& rhs) {
        sum() = combine(sum(), rhs.sum());
    }
    void operator()(const tbb::blocked_range<Index>& range) {
        Index i = range.begin();
        Index j = range.end();
        if(!has_sum) {
        __TBB_ASSERT(range.size() > 1,"there should be at least 2 elements");
            new(&sum_storage) T(combine(u(i), u(i+1))); // The condition i+1 < j is provided by the grain size of 3
            has_sum = true;
            std::advance(i,2);
            if(i==j)
                return;
        }
        sum() = brick_reduce(i, j, sum());
    }
};

template<class Index, class U, class T, class C, class R>
T parallel_transform_reduce( Index first, Index last, U u, T init, C combine, R brick_reduce) {
    par_trans_red_body<Index, U, T, C, R> body(u, init, combine, brick_reduce);
    // The grain size of 3 is used in order to provide mininum 2 elements for each body
    tbb::this_task_arena::isolate([first, last, &body]() {
        tbb::parallel_reduce(tbb::blocked_range<Index>(first, last, 3), body);
    });
    return body.sum();
}

//------------------------------------------------------------------------
// parallel_scan
//------------------------------------------------------------------------

template <class Index, class U, class T, class C, class R, class S>
class trans_scan_body {
    alignas(T) char sum_storage[sizeof(T)]; // Holds generalized non-commutative sum when has_sum==true
    R brick_reduce;                 // Most likely to have non-empty layout
    U u;
    C combine;
    S brick_scan;
    bool has_sum;             // Put last to minimize size of class
public:
    trans_scan_body(U u_, T init, C combine_, R reduce_, S brick_scan_) :
        brick_reduce(reduce_),
        u(u_),
        combine(combine_),
        brick_scan(brick_scan_),
        has_sum(true) {
        new(sum_storage) T(init);
    }

    trans_scan_body( trans_scan_body& b, tbb::split ) :
        brick_reduce(b.brick_reduce),
        u(b.u),
        combine(b.combine),
        brick_scan(b.brick_scan),
        has_sum(false) {}

    ~trans_scan_body() {
        // 17.6.5.12 tells us to not worry about catching exceptions from destructors.
        if( has_sum )
            sum().~T();
    }

    T& sum() const {
        __TBB_ASSERT(has_sum,"sum expected");
        return *(T*)sum_storage;
    }

    void operator()(const tbb::blocked_range<Index>& range, tbb::pre_scan_tag) {
        Index i = range.begin();
        Index j = range.end();
        if(!has_sum) {
            new(&sum_storage) T(u(i));
            has_sum = true;
            ++i;
            if(i==j)
                return;
        }
        sum() = brick_reduce(i, j, sum());
    }

    void operator()(const tbb::blocked_range<Index>& range, tbb::final_scan_tag) {
        sum() = brick_scan(range.begin(), range.end(), sum());
    }

    void reverse_join(trans_scan_body& a) {
        if(has_sum) {
            sum() = combine(a.sum(), sum());
        }
        else {
            new(&sum_storage) T(a.sum());
            has_sum = true;
        }
    }

    void assign(trans_scan_body& b) {
        sum() = b.sum();
    }
};

template<typename Index>
Index split(Index m) {
    Index k = 1;
    while( 2*k<m ) k*=2;
    return k;
}

//------------------------------------------------------------------------
// parallel_strict_scan
//------------------------------------------------------------------------


template<typename Index, typename T, typename R, typename C>
void upsweep(Index i, Index m, Index tilesize, T* r, Index lastsize, R reduce, C combine) {
    if (m == 1)
        r[0] = reduce(i*tilesize, lastsize);
    else {
        Index k = split(m);
        tbb::parallel_invoke(
            [=] {upsweep(i, k, tilesize, r, tilesize, reduce, combine); },
            [=] {upsweep(i + k, m - k, tilesize, r + k, lastsize, reduce, combine); }
        );
        if (m == 2 * k)
            r[m - 1] = combine(r[k - 1], r[m - 1]);
    }
}

template<typename Index, typename T, typename C, typename S>
void downsweep(Index i, Index m, Index tilesize, T* r, Index lastsize, T initial, C combine, S scan) {
    if (m == 1)
        scan(i*tilesize, lastsize, initial);
    else {
        const Index k = split(m);
        tbb::parallel_invoke(
            [=] {downsweep(i, k, tilesize, r, tilesize, initial, combine, scan); },
            // Assumes that combine never throws.
            [=, &combine] {downsweep(i + k, m - k, tilesize, r + k, lastsize, combine(initial, r[k - 1]), combine, scan); }
        );
    }
}

// Adapted from Intel(R) Cilk(TM) version from cilkpub.
// Let i:len denote a counted interval of length n starting at i.  s denotes a generalized-sum value.
// Expected actions of the functors are:
//     reduce(i,len) -> s  -- return reduction value of i:len.
//     combine(s1,s2) -> s -- return merged sum
//     apex(s) -- do any processing necessary between reduce and scan.
//     scan(i,len,initial) -- perform scan over i:len starting with initial.
// The initial range 0:n is partitioned into consecutive subranges.
// reduce and scan are each called exactly once per subrange.
// Thus callers can rely upon side effects in reduce.
// combine must not throw an exception.
// apex is called exactly once, after all calls to reduce and before all calls to scan.
// For example, it's useful for allocating a buffer used by scan but whose size is the sum of all reduction values.
// T must have a trivial constructor and destructor.
template<typename Index, typename T, typename R, typename C, typename S, typename A>
void parallel_strict_scan(Index n, T initial, R reduce, C combine, S scan, A apex) {
    tbb::this_task_arena::isolate([=, &combine]() {
        if (n > 1) {
            Index p = tbb::this_task_arena::max_concurrency();
            const Index slack = 4;
            Index tilesize = (n - 1) / (slack*p) + 1;
            Index m = (n - 1) / tilesize;
            buffer<T> buf(m + 1);
            if (buf) {
                T* r = buf.get();
                upsweep(Index(0), Index(m + 1), tilesize, r, n - m*tilesize, reduce, combine);
                // When apex is a no-op and combine has no side effects, a good optimizer
                // should be able to eliminate all code between here and apex.
                // Alternatively, provide a default value for apex that can be
                // recognized by metaprogramming that conditionlly executes the following.
                size_t k = m + 1;
                T t = r[k - 1];
                while ((k &= k - 1))
                    t = combine(r[k - 1], t);
                apex(combine(initial, t));
                downsweep(Index(0), Index(m + 1), tilesize, r, n - m*tilesize, initial, combine, scan);
                return;
            }
        }
        // Fewer than 2 elements in sequence, or out of memory.  Handle has single block.
        T sum = initial;
        if (n)
            sum = combine(sum, reduce(Index(0), n));
        apex(sum);
        if (n)
            scan(Index(0), n, initial);
    });
}

template<typename Index, typename U, typename T, typename C, typename R, typename S, typename OutputIterator>
T parallel_transform_scan(Index n, U u, T init, C combine, R brick_reduce, S brick_scan, OutputIterator) {
    if (n <= 0) {
        return init;
    }

    trans_scan_body<Index, U, T, C, R, S> body(u, init, combine, brick_reduce, brick_scan);
    auto range = tbb::blocked_range<Index>(0, n);
    tbb::this_task_arena::isolate([range, &body]() {
        tbb::parallel_scan(range, body);
    });
    return body.sum();
}

//------------------------------------------------------------------------
// parallel_stable_sort
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// stable_sort utilities
//
// These are used by parallel implementations but do not depend on them.
//------------------------------------------------------------------------

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare, typename Cleanup, typename LeafMerge>
class merge_task: public tbb::task {
    /*override*/tbb::task* execute();
    RandomAccessIterator1 xs, xe;
    RandomAccessIterator2 ys, ye;
    RandomAccessIterator3 zs;
    Compare comp;
    Cleanup cleanup;
    LeafMerge leaf_merge;
public:
    merge_task( RandomAccessIterator1 xs_, RandomAccessIterator1 xe_,
                RandomAccessIterator2 ys_, RandomAccessIterator2 ye_,
                RandomAccessIterator3 zs_,
                Compare comp_, Cleanup cleanup_, LeafMerge leaf_merge_) :
        xs(xs_), xe(xe_), ys(ys_), ye(ye_), zs(zs_), comp(comp_), cleanup(cleanup_), leaf_merge(leaf_merge_)
    {}
};

const size_t MERGE_CUT_OFF = 2000;
template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare, typename Cleanup, typename LeafMerge>
tbb::task* merge_task<RandomAccessIterator1, RandomAccessIterator2, RandomAccessIterator3, Compare, Cleanup, LeafMerge>::execute() {
    const auto n = (xe-xs) + (ye-ys);
    if(n <= MERGE_CUT_OFF) {
        leaf_merge(xs, xe, ys, ye, zs, comp);

        //we clean the buffer one time on last step of the sort
        cleanup(xs, xe);
        cleanup(ys, ye);
        return NULL;
    }
    else {
        RandomAccessIterator1 xm;
        RandomAccessIterator2 ym;
        if(xe-xs < ye-ys) {
            ym = ys+(ye-ys)/2;
            xm = std::upper_bound(xs, xe, *ym, comp);
        }
        else {
            xm = xs+(xe-xs)/2;
            ym = std::lower_bound(ys, ye, *xm, comp);
        }
        const RandomAccessIterator3 zm = zs + ((xm-xs) + (ym-ys));
        tbb::task* right = new(allocate_additional_child_of(*parent()))
            merge_task(xm, xe, ym, ye, zm, comp, cleanup, leaf_merge);
        spawn(*right);
        recycle_as_continuation();
        xe = xm;
        ye = ym;
    }
    return this;
}

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare, typename LeafSort>
class stable_sort_task : public tbb::task {
    /*override*/tbb::task* execute();
    RandomAccessIterator1 xs, xe;
    RandomAccessIterator2 zs;
    Compare comp;
    LeafSort leaf_sort;
    int32_t inplace;
    std::size_t nsort;
public:
    stable_sort_task(RandomAccessIterator1 xs_, RandomAccessIterator1 xe_, RandomAccessIterator2 zs_, int32_t inplace_, Compare comp_, LeafSort leaf_sort_, std::size_t n) :
        xs(xs_), xe(xe_), zs(zs_), inplace(inplace_), comp(comp_), leaf_sort(leaf_sort_), nsort(n)
    {}
};

//! Binary operator that does nothing
struct binary_no_op {
    template<typename T>
    void operator()(T, T) {}
};

const size_t STABLE_SORT_CUT_OFF = 500;

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare, typename LeafSort>
tbb::task* stable_sort_task<RandomAccessIterator1, RandomAccessIterator2, Compare, LeafSort>::execute() {
    const auto n = xe - xs;
    const auto nmerge = nsort > 0 ? nsort : n;
    if (n <= STABLE_SORT_CUT_OFF) {
        leaf_sort(xs, xe, comp);
        if (inplace != 2)
            init_buf(xs, xe, zs, inplace == 0);
        return NULL;
    }
    else {
        const RandomAccessIterator1 xm = xs + n / 2;
        const RandomAccessIterator2 zm = zs + (xm - xs);
        const RandomAccessIterator2 ze = zs + n;
        task* m;
        auto move_values = [](RandomAccessIterator2 x, RandomAccessIterator1 z) {*z = std::move(*x); };
        auto move_sequences = [](RandomAccessIterator2 first1, RandomAccessIterator2 last1, RandomAccessIterator1 first2) {return std::move(first1, last1, first2); };
        if (inplace == 2)
            m = new (allocate_continuation()) merge_task<RandomAccessIterator2, RandomAccessIterator2, RandomAccessIterator1, Compare, serial_destroy, serial_move_merge<decltype(move_values), decltype(move_sequences)>>(zs, zm, zm, ze, xs, comp, serial_destroy(), serial_move_merge<decltype(move_values), decltype(move_sequences)>(nmerge, move_values, move_sequences));
        else if (inplace)
            m = new (allocate_continuation()) merge_task<RandomAccessIterator2, RandomAccessIterator2, RandomAccessIterator1, Compare, binary_no_op, serial_move_merge<decltype(move_values), decltype(move_sequences)>>(zs, zm, zm, ze, xs, comp, binary_no_op(), serial_move_merge<decltype(move_values), decltype(move_sequences)>(nmerge, move_values, move_sequences));
        else {
            auto move_values = [](RandomAccessIterator1 x, RandomAccessIterator2 z) {*z = std::move(*x); };
            auto move_sequences = [](RandomAccessIterator1 first1, RandomAccessIterator1 last1, RandomAccessIterator2 first2) {return std::move(first1, last1, first2); };
            m = new (allocate_continuation()) merge_task<RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2, Compare, binary_no_op, serial_move_merge<decltype(move_values), decltype(move_sequences)>>(xs, xm, xm, xe, zs, comp, binary_no_op(), serial_move_merge<decltype(move_values), decltype(move_sequences)>(nmerge, move_values, move_sequences));
        }
        m->set_ref_count(2);
        task* right = new(m->allocate_child()) stable_sort_task(xm, xe, zm, !inplace, comp, leaf_sort, nmerge);
        spawn(*right);
        recycle_as_child_of(*m);
        xe = xm;
        inplace = !inplace;
    }
    return this;
}

template<typename RandomAccessIterator, typename Compare, typename LeafSort>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp, LeafSort leaf_sort, std::size_t nsort = 0) {
    tbb::this_task_arena::isolate([=, &nsort]() {
        //sorting based on task tree and parallel merge
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        const auto n = xe - xs;
        if (nsort == 0)
            nsort = n;

        if (n > STABLE_SORT_CUT_OFF) {
            assert(nsort > 0 && nsort <= n);
            buffer<T> buf(n);
            if (buf) {
                using tbb::task;
                typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
                task::spawn_root_and_wait(*new(task::allocate_root()) stable_sort_task<RandomAccessIterator, T*, Compare, LeafSort>(xs, xe, (T*)buf.get(), 2, comp, leaf_sort, nsort));
                return;
            }
        }
        // Not enough memory available or sort too small - fall back on serial sort
        leaf_sort(xs, xe, comp);
    });
}

//------------------------------------------------------------------------
// parallel_merge
//------------------------------------------------------------------------

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare, typename LeafMerge>
void parallel_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 ys, RandomAccessIterator2 ye, RandomAccessIterator3 zs, Compare comp, LeafMerge leaf_merge) {
    if ((xe - xs) + (ye - ys) <= MERGE_CUT_OFF) {
        // Fall back on serial merge
        leaf_merge(xs, xe, ys, ye, zs, comp);
    }
    else {
        tbb::this_task_arena::isolate([=]() {
            using tbb::task;
            task::spawn_root_and_wait(*new(task::allocate_root()) merge_task<RandomAccessIterator1, RandomAccessIterator2, RandomAccessIterator3, Compare, binary_no_op, LeafMerge>(xs, xe, ys, ye, zs, comp, binary_no_op(), leaf_merge));
        });
    }
}

} // namespace par_backend
} // namespace pstl

#endif /* __PSTL_parallel_backend_tbb_H */
