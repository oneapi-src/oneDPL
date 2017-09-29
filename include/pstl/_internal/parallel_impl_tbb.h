/*
    Copyright (c) 2017 Intel Corporation

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

#ifndef __PSTL_parallel_impl_tbb_H
#define __PSTL_parallel_impl_tbb_H

// This header defines the minimum set of parallel routines required to support Parallel STL,
// implemented on top of Intel(R) Threading Building Blocks (Intel(R) TBB) library

#include <atomic>

#include "common.h"
#include "algorithm_impl.h"
#include "numeric_impl.h"  /* count and count_if use pattern_transform_reduce */

// Bring in minimal required subset of Intel TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

#if TBB_INTERFACE_VERSION < 10000
#error Intel(R) Threading Building Blocks 2018 is required; older versions are not supported.
#endif

namespace __icp_algorithm {

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

//! Evaluate brick f[i,j) to each subrange [i,j) of [first,last)
// wrapper over tbb::parallel_for with exceptions handler
template<class Index, class F>
void parallel_for(Index first, Index last, F f) {
    except_handler([=]() {
        tbb::this_task_arena::isolate([=]() {
            tbb::parallel_for(tbb::blocked_range<Index>(first, last), parallel_for_body<Index, F>(f));
        });
    });
}

// wrapper over tbb::parallel_reduce with exceptions handler
template<class Value,class Index, class Body>
Value parallel_reduce(tbb::blocked_range<Index> range, Body &body) {
    return except_handler([range, &body]()->Value {
        tbb::this_task_arena::isolate([range, &body]() {
            tbb::parallel_reduce(range, body);
        });
        return body.sum();
    });
}

// wrapper over tbb::parallel_reduce with exceptions handler
template<class Value, class Index, typename RealBody, typename Reduction>
Value parallel_reduce(tbb::blocked_range<Index> range, const Value& identity, const RealBody& real_body, const Reduction& reduction) {
    return except_handler([range, &identity, &real_body, &reduction]()->Value {
        return tbb::this_task_arena::isolate([range, &identity, &real_body, &reduction]()->Value {
            return tbb::parallel_reduce(range, identity, real_body, reduction);
        });
    });
}

// wrapper over tbb::parallel_scan with exceptions handler
template<class Value, class Index, class Body>
Value parallel_scan(tbb::blocked_range<Index> range, Body &body) {
    return except_handler([range, &body]()->Value {
        tbb::this_task_arena::isolate([range, &body]() {
            tbb::parallel_scan(range, body);
        });
        return body.sum();
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
        has_sum(true)
    {
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
        if( has_sum ) {
            sum().~T();
        }
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
    return __icp_algorithm::parallel_reduce<T>(tbb::blocked_range<Index>(first, last, 3), body);
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
    S scan;
    bool has_sum;             // Put last to minimize size of class
public:
    trans_scan_body(U u_, T init, C combine_, R reduce_, S scan_) :
        brick_reduce(reduce_),
        u(u_),
        combine(combine_),
        scan(scan_),
        has_sum(true)
    {
        new(sum_storage) T(init);
    }

    trans_scan_body( trans_scan_body& b, tbb::split ) :
        brick_reduce(b.brick_reduce),
        u(b.u),
        combine(b.combine),
        scan(b.scan),
        has_sum(false)
    {}

    ~trans_scan_body() {
        // 17.6.5.12 tells us to not worry about catching exceptions from destructors.
        if( has_sum ) {
            sum().~T();
        }
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
        sum() = scan(range.begin(), range.end(), sum());
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

template<class Index, class U, class T, class C, class R, class S>
T parallel_transform_scan(Index n, U u, T init, C combine, R brick_reduce, S scan) {
    if(n) {
        trans_scan_body<Index, U, T, C, R, S> body(u, init, combine, brick_reduce, scan);
        return __icp_algorithm::parallel_scan<T>(tbb::blocked_range<Index>(0, n), body);
    }
    else
        return init;
}

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
        if( m==1 )
            r[0] = reduce(i*tilesize, lastsize);
        else {
            Index k = split(m);
            tbb::parallel_invoke(
               [=]{upsweep( i, k, tilesize, r, tilesize, reduce, combine );},
               [=]{upsweep( i+k, m-k, tilesize, r+k, lastsize, reduce, combine );}
            );
            if( m==2*k )
                r[m-1] = combine(r[k-1], r[m-1]);
        }
}

template<typename Index, typename T, typename C, typename S>
void downsweep(Index i, Index m, Index tilesize, T* r, Index lastsize, T initial, C combine, S scan) {
        if( m==1 ) {
            scan(i*tilesize, lastsize, initial );
        } else {
            Index k = split(m);
            tbb::parallel_invoke(
                [=]{downsweep(i, k, tilesize, r, tilesize, initial, combine, scan);},
                // Assumes that combine never throws.
                [=]{downsweep(i+k, m-k, tilesize, r+k, lastsize, combine(initial, r[k-1]), combine, scan);}
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
void parallel_strict_scan( Index n, T initial, R reduce, C combine, S scan, A apex ) {
    except_handler([=]() {
        tbb::this_task_arena::isolate([=](){
            if( n>1 ) {
                Index p = __PSTL_get_workers_num();
                const Index slack = 4;
                Index tilesize = (n-1)/(slack*p) + 1;
                Index m = (n-1)/tilesize;
                raw_buffer buf((m+1)*sizeof(T));
                if( buf ) {
                    T* r = static_cast<T*>(buf.get());
                    upsweep(Index(0), Index(m+1), tilesize, r, n-m*tilesize, reduce, combine);
                    // When apex is a no-op and combine has no side effects, a good optimizer
                    // should be able to eliminate all code between here and apex.
                    // Alternatively, provide a default value for apex that can be
                    // recognized by metaprogramming that conditionlly executes the following.
                    size_t k = m+1;
                    T t = r[k-1];
                    while( (k&=k-1) )
                        t = combine(r[k-1],t);
                    apex(combine(initial,t));
                    downsweep(Index(0), Index(m+1), tilesize, r, n-m*tilesize, initial, combine, scan);
                    return;
                }
            }
            // Fewer than 2 elements in sequence, or out of memory.  Handle has single block.
            T sum = initial;
            if(n)
                sum = combine(sum, reduce(Index(0), n));
            apex(sum);
            if(n)
                scan(Index(0), n, initial);
        });
    });
}

//------------------------------------------------------------------------
// parallel_or
//------------------------------------------------------------------------

//! Return true if brick f[i,j) returns true for some subrange [i,j) of [first,last)
template<class Index, class Brick>
bool parallel_or( Index first, Index last, Brick f ) {
    return except_handler([=]() -> bool {
    std::atomic<bool> found(false);
    __icp_algorithm::parallel_for( first, last, [f,&found]( Index i, Index j ) {
            if( f(i, j) ) {
                found = true;
                tbb::task::self().group()->cancel_group_execution();
            }});
    return found;
    });
}

//------------------------------------------------------------------------
// parallel_first
//------------------------------------------------------------------------

/** Return minimum value returned by brick f[i,j) for subranges [i,j) of [first,last)
    Each f[i,j) must return a value in [i,j). */
template<class Index, class Brick>
Index parallel_first( Index first, Index last, Brick f ) {
    return except_handler([=]() -> Index {
    typedef typename std::iterator_traits<Index>::difference_type difference_type;
    difference_type n = last-first;
    std::atomic<difference_type> minimum( last-first );
    __icp_algorithm::parallel_for(first, last, [f, first, &minimum](Index i, Index j) {
        // See "Reducing Contention Through Priority Updates", PPoPP '13, for discussion of 
        // why using a shared variable scales fairly well in this situation.
        if (i - first < minimum) {
            Index res = f(i, j);
            // If not 'last' returned then we found what we want so put this to minimum
            if (res != j) {
                difference_type k = res - first;
                for (difference_type old = minimum; k < old; old = minimum) {
                    minimum.compare_exchange_weak(old, k);
                }
            }
        }
    });
    return first + minimum;
    });
}

//------------------------------------------------------------------------
// parallel_stable_sort
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// stable_sort utilities
//
// These are used by parallel implementations but do not depend on them.
//------------------------------------------------------------------------

//! Destroy sequence [xs,xe)
template<class RandomAccessIterator>
void serial_destroy(RandomAccessIterator zs, RandomAccessIterator ze) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
    while(zs!=ze) {
        --ze;
        (*ze).~T();
    }
}

//! Merge sequences [xs,xe) and [ys,ye) to output sequence [zs,(xe-xs)+(ye-ys)), using std::move
template<class RandomAccessIterator1, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
void serial_move_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 ys, RandomAccessIterator2 ye, RandomAccessIterator3 zs, Compare comp) {
    if(xs!=xe) {
        if(ys!=ye) {
            for(;;)
                if(comp(*ys, *xs)) {
                    *zs = std::move(*ys);
                    ++zs;
                    if(++ys==ye) break;
                }
                else {
                    *zs = std::move(*xs);
                    ++zs;
                    if(++xs==xe) goto movey;
                }
        }
        ys = xs;
        ye = xe;
    }
movey:
    std::move(ys, ye, zs);
}

template<typename RandomAccessIterator1, typename RandomAccessIterator2>
void merge_sort_init_temp_buf(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 zs, bool inplace) {
    RandomAccessIterator2 ze = zs + (xe-xs);
    typedef typename std::iterator_traits<RandomAccessIterator2>::value_type T;
    if(inplace)
        // Initialize the temporary buffer
        for(; zs!=ze; ++zs)
            new(&*zs) T;
    else
        // Initialize the temporary buffer and move keys to it.
        for(; zs!=ze; ++xs, ++zs)
            new(&*zs) T(std::move(*xs));
}

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
class merge_task: public tbb::task {
    /*override*/tbb::task* execute();
    RandomAccessIterator1 xs, xe;
    RandomAccessIterator2 ys, ye;
    RandomAccessIterator3 zs;
    Compare comp;
    bool destroy;
public:
    merge_task( RandomAccessIterator1 xs_, RandomAccessIterator1 xe_,
                RandomAccessIterator2 ys_, RandomAccessIterator2 ye_,
                RandomAccessIterator3 zs_,
                bool destroy_, Compare comp_) :
        xs(xs_), xe(xe_), ys(ys_), ye(ye_), zs(zs_), destroy(destroy_), comp(comp_)
    {}
};

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
tbb::task* merge_task<RandomAccessIterator1, RandomAccessIterator2, RandomAccessIterator3, Compare>::execute() {
    const size_t MERGE_CUT_OFF = 2000;
    auto n = (xe-xs) + (ye-ys);
    if(n <= MERGE_CUT_OFF) {
        serial_move_merge(xs, xe, ys, ye, zs, comp);
        if(destroy) {
            serial_destroy(xs, xe);
            serial_destroy(ys, ye);
        }
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
        RandomAccessIterator3 zm = zs + ((xm-xs) + (ym-ys));
        tbb::task* right = new(allocate_additional_child_of(*parent()))
            merge_task(xm, xe, ym, ye, zm, destroy, comp);
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
    int inplace;
public:
    stable_sort_task(RandomAccessIterator1 xs_, RandomAccessIterator1 xe_, RandomAccessIterator2 zs_, int inplace_, Compare comp_, LeafSort leaf_sort_ ) :
        xs(xs_), xe(xe_), zs(zs_), inplace(inplace_), comp(comp_), leaf_sort(leaf_sort_)
    {}
};

const size_t STABLE_SORT_CUT_OFF = 500;

template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare, typename LeafSort>
tbb::task* stable_sort_task<RandomAccessIterator1, RandomAccessIterator2, Compare, LeafSort>::execute() {
        if( xe - xs <= STABLE_SORT_CUT_OFF ) {
            leaf_sort(xs, xe, comp);
            if( inplace!=2 )
                merge_sort_init_temp_buf(xs, xe, zs, inplace!=0);
            return NULL;
        } else {
            RandomAccessIterator1 xm = xs + (xe - xs) / 2;
            RandomAccessIterator2 zm = zs + (xm - xs);
            RandomAccessIterator2 ze = zs + (xe - xs);
            task* m;
            if (inplace)
                m = new (allocate_continuation()) merge_task<RandomAccessIterator2,RandomAccessIterator2,RandomAccessIterator1,Compare>(zs, zm, zm, ze, xs, inplace==2, comp);
            else
                m = new (allocate_continuation()) merge_task<RandomAccessIterator1,RandomAccessIterator1,RandomAccessIterator2,Compare>(xs, xm, xm, xe, zs, false, comp);
            m->set_ref_count(2);
            task* right = new(m->allocate_child()) stable_sort_task(xm,xe,zm,!inplace, comp, leaf_sort);
            spawn(*right);
            recycle_as_child_of(*m);
            xe=xm;
            inplace=!inplace;
        }
    return this;
}

template<typename RandomAccessIterator, typename Compare, typename LeafSort>
void parallel_stable_sort( RandomAccessIterator xs, RandomAccessIterator xe, Compare comp, LeafSort leaf_sort ) {
    except_handler([=]() {
        tbb::this_task_arena::isolate([=](){
            typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
            if( xe-xs > STABLE_SORT_CUT_OFF ) {
                __icp_algorithm::raw_buffer buf( sizeof(T)*(xe-xs) );
                if( buf ) {
                    using tbb::task;
                    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
                    task::spawn_root_and_wait(*new( task::allocate_root() ) __icp_algorithm::stable_sort_task<RandomAccessIterator,T*,Compare,LeafSort>( xs, xe, (T*)buf.get(), 2, comp, leaf_sort ));
                    return;
                }
            }
            // Not enough memory available or sort too small - fall back on serial sort
            leaf_sort( xs, xe, comp );
        });
    });
}

//------------------------------------------------------------------------
// parallel_equal
//------------------------------------------------------------------------
template<class InputIterator1, class InputIterator2, class Pred, class IsVector>
bool parallel_equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, Pred p, IsVector vec) {
    return __icp_algorithm::parallel_reduce(
        tbb::blocked_range<InputIterator1>(first1, last1, 500), true,
        [=](const tbb::blocked_range<InputIterator1>& r, bool is_equal_local) -> bool {
        return is_equal_local && brick_equal(r.begin(), r.end(), first2+(r.begin()-first1), p, vec);
    },
        [=](const bool is_equal_local1, const bool is_equal_local2) -> bool {
        return is_equal_local1 && is_equal_local2;
    }
    );
}

template<class InputIterator1, class InputIterator2, class BinaryPredicate, class IsVector>
bool pattern_equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate p, IsVector vec, /*is_parallel=*/std::true_type) {
    return parallel_equal(first1, last1, first2, p, vec);
}

//------------------------------------------------------------------------
// count
//------------------------------------------------------------------------
template<class InputIterator, class Predicate, class IsVector>
typename std::iterator_traits<InputIterator>::difference_type
pattern_count(InputIterator first, InputIterator last, Predicate pred, /* is_parallel */ std::true_type, IsVector vec) {
    typedef typename std::iterator_traits<InputIterator>::difference_type size_type;
    return __icp_algorithm::parallel_reduce(tbb::blocked_range<InputIterator>(first, last), size_type(0),
        [pred, vec](const tbb::blocked_range<InputIterator>& r, size_type value)->size_type {
        return value+brick_count(r.begin(), r.end(), pred, vec);
    },
        std::plus<size_type>()
    );
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------
template<class Index, class BinaryPredicate, class IsVector>
Index pattern_adjacent_find(Index first, Index last, BinaryPredicate pred, /* is_parallel */ std::true_type, IsVector vec, bool or_semantic) {
    if(last-first<2)
        return last;

    return __icp_algorithm::parallel_reduce(tbb::blocked_range<Index>(first, last), last,
        [last, pred, vec, or_semantic](const tbb::blocked_range<Index>& r, Index value)->Index {

        if(or_semantic && value<last) {//found
            tbb::task::self().group()->cancel_group_execution();
            return value;
        }

        if(value>r.begin()) {
            //modify local_last to check the predicate on the boundary values; //TODO: to use a custom tbb::blocked_range with boundaries overlapping
            Index local_last = r.end();
            if(local_last!=last)
                ++local_last;

            //correct the global result iterator if the "brick" returns a local "last"
            const Index res = brick_adjacent_find(r.begin(), local_last, pred, vec, or_semantic);
            if(res<local_last)
                value = res;
        }
        return value;
    },
        [](Index x, Index y)->Index { return x<y ? x : y; } //reduce a value
    );
}

//------------------------------------------------------------------------
// fill, fill_n
//------------------------------------------------------------------------
template<class Index, class T, class IsVector>
Index pattern_fill(Index first, Index last, const T& value, /*is_parallel=*/std::true_type, IsVector vec) {
    __icp_algorithm::parallel_for(first, last, [&value, vec](Index begin, Index end) {
        brick_fill(begin, end, value, vec); });
    return last;
}

template<class Index, class Size, class T, class IsVector>
Index pattern_fill_n(Index first, Size count, const T& value, /*is_parallel=*/std::true_type, IsVector vec) {
    return pattern_fill(first, first + count, value, std::true_type(), vec);
}

//------------------------------------------------------------------------
// generate, generate_n
//------------------------------------------------------------------------
template<class Index, class Generator, class IsVector>
Index pattern_generate(Index first, Index last, Generator g, /*is_parallel=*/std::true_type, IsVector vec) {
    __icp_algorithm::parallel_for(first, last, [g, vec](Index begin, Index end) {
        brick_generate(begin, end, g, vec); });
    return last;
}

template<class Index, class Size, class Generator, class IsVector>
Index pattern_generate_n(Index first, Size count, Generator g, /*is_parallel=*/std::true_type, IsVector vec) {
    return pattern_generate(first, first + count, g, std::true_type(), vec);
}

} // namespace __icp_algorithm

#endif /* __PSTL_parallel_impl_tbb_H */
