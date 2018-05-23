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
#include <algorithm>

#include "parallel_backend_utils.h"

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

namespace __pstl {
namespace par_backend {

//! Raw memory buffer with automatic freeing and no exceptions.
/** Some of our algorithms need to start with raw memory buffer,
not an initialize array, because initialization/destruction
would make the span be at least O(N). */
template<typename _Tp>
class buffer {
    _Tp* _M_ptr;
    buffer(const buffer&) = delete;
    void operator=(const buffer&) = delete;
public:
    //! Try to obtain buffer of given size to store objects of T type
    buffer(size_t __n): _M_ptr(static_cast<_Tp*>(operator new(__n*sizeof(_Tp), std::nothrow))) {}
    //! True if buffer was successfully obtained, zero otherwise.
    operator bool() const { return _M_ptr != NULL; }
    //! Return pointer to buffer, or  NULL if buffer could not be obtained.
    _Tp* get() const { return _M_ptr; }
    //! Destroy buffer
    ~buffer() { operator delete(_M_ptr); }
};

// Wrapper for tbb::task
inline void cancel_execution() {
    tbb::task::self().group()->cancel_group_execution();
}

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

template <class _Index, class _RealBody>
class parallel_for_body {
public:
    parallel_for_body( const _RealBody& __body) : _M_body( __body ) { }
    parallel_for_body(const parallel_for_body& __body): _M_body(__body._M_body) { }
    void operator()(const tbb::blocked_range<_Index>& __range) const {
        _M_body(__range.begin(), __range.end());
    }
private:
    _RealBody _M_body;
};

//! Evaluation of brick f[i,j) for each subrange [i,j) of [first,last)
// wrapper over tbb::parallel_for
template<class _Index, class _Fp>
void parallel_for(_Index __first, _Index __last, _Fp __f) {
    tbb::this_task_arena::isolate([=]() {
        tbb::parallel_for(tbb::blocked_range<_Index>(__first, __last), parallel_for_body<_Index, _Fp>(__f));
    });
}

//! Evaluation of brick f[i,j) for each subrange [i,j) of [first,last)
// wrapper over tbb::parallel_reduce
template<class _Value, class _Index, typename _RealBody, typename _Reduction>
_Value parallel_reduce(_Index __first, _Index __last, const _Value& __identity, const _RealBody& __real_body,
                       const _Reduction& __reduction) {
    return tbb::this_task_arena::isolate([__first, __last, &__identity, &__real_body, &__reduction]() -> _Value {
        return tbb::parallel_reduce(tbb::blocked_range<_Index>(__first, __last), __identity,
            [__real_body](const tbb::blocked_range<_Index>& __r, const _Value& __value)-> _Value {
            return __real_body(__r.begin(),__r.end(), __value);
        },
        __reduction);
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

template<class _Index, class _Up, class _Tp, class _Cp, class _Rp>
struct par_trans_red_body {
    alignas(_Tp) char _M_sum_storage[sizeof(_Tp)]; // Holds generalized non-commutative sum when has_sum==true
    _Rp _M_brick_reduce;                 // Most likely to have non-empty layout
    _Up _M_u;
    _Cp _M_combine;
    bool _M_has_sum;             // Put last to minimize size of class
    _Tp& sum() {
        __TBB_ASSERT(_M_has_sum, "sum expected");
        return *(_Tp*)_M_sum_storage;
    }
    par_trans_red_body( _Up __u, _Tp __init, _Cp __c, _Rp __r)
        : _M_brick_reduce(__r)
        , _M_u(__u)
        , _M_combine(__c)
        , _M_has_sum(true)
    { new(_M_sum_storage) _Tp(__init); }

    par_trans_red_body( par_trans_red_body& __left, tbb::split )
      : _M_brick_reduce(__left._M_brick_reduce)
      , _M_u(__left._M_u)
      , _M_combine(__left._M_combine)
      , _M_has_sum(false)
    { }

    ~par_trans_red_body() {
        // 17.6.5.12 tells us to not worry about catching exceptions from destructors.
        if( _M_has_sum )
            sum().~_Tp();
    }

    void join(par_trans_red_body& __rhs) {
        sum() = _M_combine(sum(), __rhs.sum());
    }

    void operator()(const tbb::blocked_range<_Index>& __range) {
        _Index __i = __range.begin();
        _Index __j = __range.end();
        if(!_M_has_sum) {
        __TBB_ASSERT(__range.size() > 1,"there should be at least 2 elements");
        new(&_M_sum_storage) _Tp(_M_combine(_M_u(__i), _M_u(__i+1))); // The condition i+1 < j is provided by the grain size of 3
            _M_has_sum = true;
            std::advance(__i,2);
            if(__i == __j)
                return;
        }
        sum() = _M_brick_reduce(__i, __j, sum());
    }
};

template<class _Index, class _Up, class _Tp, class _Cp, class _Rp>
_Tp parallel_transform_reduce( _Index __first, _Index __last, _Up __u, _Tp __init, _Cp __combine, _Rp __brick_reduce) {
    par_trans_red_body<_Index, _Up, _Tp, _Cp, _Rp> __body(__u, __init, __combine, __brick_reduce);
    // The grain size of 3 is used in order to provide mininum 2 elements for each body
    tbb::this_task_arena::isolate([__first, __last, &__body]() {
        tbb::parallel_reduce(tbb::blocked_range<_Index>(__first, __last, 3), __body);
    });
    return __body.sum();
}

//------------------------------------------------------------------------
// parallel_scan
//------------------------------------------------------------------------

template <class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
class trans_scan_body {
    alignas(_Tp) char _M_sum_storage[sizeof(_Tp)]; // Holds generalized non-commutative sum when has_sum==true
    _Rp _M_brick_reduce;                 // Most likely to have non-empty layout
    _Up _M_u;
    _Cp _M_combine;
    _Sp _M_scan;
    bool _M_has_sum;             // Put last to minimize size of class
public:
    trans_scan_body(_Up __u, _Tp __init, _Cp __combine, _Rp __reduce, _Sp __scan)
        : _M_brick_reduce(__reduce)
        , _M_u(__u)
        , _M_combine(__combine)
        , _M_scan(__scan)
        , _M_has_sum(true)
    { new(_M_sum_storage) _Tp(__init); }

    trans_scan_body( trans_scan_body& __b, tbb::split )
        : _M_brick_reduce(__b._M_brick_reduce)
        , _M_u(__b._M_u)
        , _M_combine(__b._M_combine)
        , _M_scan(__b._M_scan)
        , _M_has_sum(false) {}

    ~trans_scan_body() {
        // 17.6.5.12 tells us to not worry about catching exceptions from destructors.
        if( _M_has_sum )
            sum().~_Tp();
    }

    _Tp& sum() const {
        __TBB_ASSERT(_M_has_sum,"sum expected");
        return *(_Tp*)_M_sum_storage;
    }

    void operator()(const tbb::blocked_range<_Index>& __range, tbb::pre_scan_tag) {
        _Index __i = __range.begin();
        _Index __j = __range.end();
        if(!_M_has_sum) {
            new(&_M_sum_storage) _Tp(_M_u(__i));
            _M_has_sum = true;
            ++__i;
            if(__i == __j)
                return;
        }
        sum() = _M_brick_reduce(__i, __j, sum());
    }

    void operator()(const tbb::blocked_range<_Index>& __range, tbb::final_scan_tag) {
        sum() = _M_scan(__range.begin(), __range.end(), sum());
    }

    void reverse_join(trans_scan_body& __a) {
        if(_M_has_sum) {
            sum() = _M_combine(__a.sum(), sum());
        }
        else {
            new(&_M_sum_storage) _Tp(__a.sum());
            _M_has_sum = true;
        }
    }

    void assign(trans_scan_body& __b) {
        sum() = __b.sum();
    }
};

template<class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
_Tp parallel_transform_scan(_Index __n, _Up __u, _Tp __init, _Cp __combine, _Rp __brick_reduce, _Sp __scan) {
    if (__n <= 0)
        return __init;

    trans_scan_body<_Index, _Up, _Tp, _Cp, _Rp, _Sp> __body(__u, __init, __combine, __brick_reduce, __scan);
    auto __range = tbb::blocked_range<_Index>(0, __n);
    tbb::this_task_arena::isolate([__range, &__body]() {
        tbb::parallel_scan(__range, __body);
    });
    return __body.sum();
}

template<typename _Index>
_Index split(_Index __m) {
    _Index __k = 1;
    while( 2*__k < __m ) __k*=2;
    return __k;
}

//------------------------------------------------------------------------
// parallel_strict_scan
//------------------------------------------------------------------------


template<typename _Index, typename _Tp, typename _Rp, typename _Cp>
void upsweep(_Index __i, _Index __m, _Index __tilesize, _Tp* __r, _Index __lastsize, _Rp __reduce, _Cp __combine) {
    if (__m == 1)
        __r[0] = __reduce(__i*__tilesize, __lastsize);
    else {
        _Index __k = split(__m);
        tbb::parallel_invoke(
                [=] {par_backend::upsweep(__i, __k, __tilesize, __r, __tilesize, __reduce, __combine); },
                [=] {par_backend::upsweep(__i + __k, __m - __k, __tilesize, __r + __k, __lastsize, __reduce, __combine); }
        );
        if (__m == 2 * __k)
            __r[__m - 1] = __combine(__r[__k - 1], __r[__m - 1]);
    }
}

template<typename _Index, typename _Tp, typename _Cp, typename _Sp>
void downsweep(_Index __i, _Index __m, _Index __tilesize, _Tp* __r, _Index __lastsize, _Tp __initial, _Cp __combine, _Sp __scan) {
    if (__m == 1)
        __scan(__i*__tilesize, __lastsize, __initial);
    else {
        const _Index __k = par_backend::split(__m);
        tbb::parallel_invoke(
                [=] { par_backend::downsweep(__i, __k, __tilesize, __r, __tilesize, __initial, __combine, __scan); },
                // Assumes that __combine never throws.
                [=] { par_backend::downsweep(__i + __k, __m - __k, __tilesize, __r + __k, __lastsize,
                                            __combine(__initial, __r[__k - 1]), __combine, __scan); }
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
template<typename _Index, typename _Tp, typename _Rp, typename _Cp, typename _Sp, typename _Ap>
void parallel_strict_scan(_Index __n, _Tp __initial, _Rp __reduce, _Cp __combine, _Sp __scan, _Ap __apex) {
    tbb::this_task_arena::isolate([=]() {
        if (__n > 1) {
            _Index __p = tbb::this_task_arena::max_concurrency();
            const _Index __slack = 4;
            _Index __tilesize = (__n - 1) / (__slack * __p) + 1;
            _Index __m = (__n - 1) / __tilesize;
            buffer<_Tp> __buf(__m + 1);
            if (__buf) {
                _Tp* __r = __buf.get();
                par_backend::upsweep(_Index(0), _Index(__m + 1), __tilesize, __r, __n - __m * __tilesize, __reduce, __combine);
                // When __apex is a no-op and __combine has no side effects, a good optimizer
                // should be able to eliminate all code between here and __apex.
                // Alternatively, provide a default value for __apex that can be
                // recognized by metaprogramming that conditionlly executes the following.
                size_t __k = __m + 1;
                _Tp __t = __r[__k - 1];
                while ((__k &= __k - 1))
                    __t = __combine(__r[__k - 1], __t);
                __apex(__combine(__initial, __t));
                par_backend::downsweep(_Index(0), _Index(__m + 1), __tilesize, __r, __n - __m * __tilesize, __initial, __combine, __scan);
                return;
            }
        }
        // Fewer than 2 elements in sequence, or out of memory.  Handle has single block.
        _Tp __sum = __initial;
        if (__n)
            __sum = __combine(__sum, __reduce(_Index(0), __n));
        __apex(__sum);
        if (__n)
            __scan(_Index(0), __n, __initial);
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

template<typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessIterator3, typename _Compare, typename _Cleanup, typename _LeafMerge>
class merge_task: public tbb::task {
    /*override*/tbb::task* execute();
    _RandomAccessIterator1 _M_xs, _M_xe;
    _RandomAccessIterator2 _M_ys, _M_ye;
    _RandomAccessIterator3 _M_zs;
    _Compare _M_comp;
    _Cleanup _M_cleanup;
    _LeafMerge _M_leaf_merge;
public:
    merge_task( _RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe,
                _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye,
                _RandomAccessIterator3 __zs,
                _Compare __comp, _Cleanup __cleanup, _LeafMerge __leaf_merge)
      : _M_xs(__xs), _M_xe(__xe), _M_ys(__ys), _M_ye(__ye), _M_zs(__zs)
      , _M_comp(__comp), _M_cleanup(__cleanup), _M_leaf_merge(__leaf_merge)
    {}
};

const size_t __PSTL_MERGE_CUT_OFF = 2000;
template<typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessIterator3, typename __M_Compare, typename _Cleanup, typename _LeafMerge>
tbb::task* merge_task<_RandomAccessIterator1, _RandomAccessIterator2, _RandomAccessIterator3, __M_Compare, _Cleanup, _LeafMerge>::execute() {
    const auto __n = (_M_xe-_M_xs) + (_M_ye-_M_ys);
    if(__n <= __PSTL_MERGE_CUT_OFF) {
        _M_leaf_merge(_M_xs, _M_xe, _M_ys, _M_ye, _M_zs, _M_comp);

        //we clean the buffer one time on last step of the sort
        _M_cleanup(_M_xs, _M_xe);
        _M_cleanup(_M_ys, _M_ye);
        return nullptr;
    }
    else {
        _RandomAccessIterator1 __xm;
        _RandomAccessIterator2 __ym;
        if(_M_xe-_M_xs < _M_ye-_M_ys) {
            __ym = _M_ys+(_M_ye-_M_ys)/2;
            __xm = std::upper_bound(_M_xs, _M_xe, *__ym, _M_comp);
        }
        else {
            __xm = _M_xs+(_M_xe-_M_xs)/2;
            __ym = std::lower_bound(_M_ys, _M_ye, *__xm, _M_comp);
        }
        const _RandomAccessIterator3 __zm = _M_zs + ((__xm - _M_xs) + (__ym - _M_ys));
        tbb::task* __right = new(tbb::task::allocate_additional_child_of(*parent()))
            merge_task(__xm, _M_xe, __ym, _M_ye, __zm, _M_comp, _M_cleanup, _M_leaf_merge);
        tbb::task::spawn(*__right);
        tbb::task::recycle_as_continuation();
        _M_xe = __xm;
        _M_ye = __ym;
    }
    return this;
}

template<typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Compare, typename _LeafSort>
class stable_sort_task : public tbb::task {
    /*override*/tbb::task* execute();
    _RandomAccessIterator1 _M_xs, _M_xe;
    _RandomAccessIterator2 _M_zs;
    _Compare _M_comp;
    _LeafSort _M_leaf_sort;
    int32_t _M_inplace;
public:
    stable_sort_task(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _RandomAccessIterator2 __zs,
                     int32_t __inplace, _Compare __comp, _LeafSort __leaf_sort)
      : _M_xs(__xs)
      , _M_xe(__xe)
      , _M_zs(__zs)
      , _M_inplace(__inplace)
      , _M_comp(__comp)
      , _M_leaf_sort(__leaf_sort)
    {}
};

//! Binary operator that does nothing
struct binary_no_op {
    template<typename T>
    void operator()(T, T) {}
};

const size_t __PSTL_STABLE_SORT_CUT_OFF = 500;

template<typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Compare, typename _LeafSort>
tbb::task* stable_sort_task<_RandomAccessIterator1, _RandomAccessIterator2, _Compare, _LeafSort>::execute() {
        if( _M_xe - _M_xs <= __PSTL_STABLE_SORT_CUT_OFF ) {
            _M_leaf_sort(_M_xs, _M_xe, _M_comp);
            if( _M_inplace!=2 )
              par_backend::init_buf(_M_xs, _M_xe, _M_zs, _M_inplace == 0);
            return NULL;
        } else {
            const _RandomAccessIterator1 __xm = _M_xs + (_M_xe - _M_xs) / 2;
            const _RandomAccessIterator2 __zm = _M_zs + (__xm - _M_xs);
            const _RandomAccessIterator2 __ze = _M_zs + (_M_xe - _M_xs);
            tbb::task* __m;
            if (_M_inplace == 2)
                __m = new (allocate_continuation()) par_backend::merge_task< _RandomAccessIterator2, _RandomAccessIterator2,
                                                                             _RandomAccessIterator1, _Compare,
                                                                             par_backend::serial_destroy,
                                                                             par_backend::serial_move_merge
                                                                            >( _M_zs, __zm, __zm, __ze,
                                                                               _M_xs, _M_comp,
                                                                               par_backend::serial_destroy(),
                                                                               par_backend::serial_move_merge() );
            else if (_M_inplace)
                __m = new (allocate_continuation()) par_backend::merge_task< _RandomAccessIterator2, _RandomAccessIterator2,
                                                                             _RandomAccessIterator1, _Compare,
                                                                             binary_no_op, serial_move_merge
                                                                            >(_M_zs, __zm, __zm, __ze,
                                                                              _M_xs, _M_comp,
                                                                              par_backend::binary_no_op(),
                                                                              par_backend::serial_move_merge());
            else
                __m = new (allocate_continuation()) par_backend::merge_task< _RandomAccessIterator1, _RandomAccessIterator1,
                                                                             _RandomAccessIterator2, _Compare,
                                                                             par_backend::binary_no_op,
                                                                             par_backend::serial_move_merge
                                                                            >(_M_xs, __xm, __xm, _M_xe, _M_zs, _M_comp,
                                                                              par_backend::binary_no_op(),
                                                                              par_backend::serial_move_merge());
            __m->set_ref_count(2);
            tbb::task* __right = new(__m->allocate_child()) stable_sort_task(__xm, _M_xe, __zm, !_M_inplace, _M_comp, _M_leaf_sort);
            task::spawn(*__right);
            recycle_as_child_of(*__m);
            _M_xe = __xm;
            _M_inplace= !_M_inplace;
        }
    return this;
}

template<typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void parallel_stable_sort(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp, _LeafSort __leaf_sort) {
    tbb::this_task_arena::isolate([=]() {
        //sorting based on task tree and parallel merge
        typedef typename std::iterator_traits<_RandomAccessIterator>::value_type _value_type;
        if (__xe - __xs > __PSTL_STABLE_SORT_CUT_OFF) {
          par_backend::buffer<_value_type> __buf(__xe - __xs);
            if (__buf) {
              typedef stable_sort_task<_RandomAccessIterator, _value_type*, _Compare, _LeafSort> _task_type;
              tbb::task::spawn_root_and_wait(*new(tbb::task::allocate_root()) _task_type(__xs, __xe, (_value_type*)__buf.get(), 2,
                                                                                         __comp, __leaf_sort));
                return;
            }
        }
        // Not enough memory available or sort too small - fall back on serial sort
        __leaf_sort(__xs, __xe, __comp);
    });
}

//------------------------------------------------------------------------
// parallel_merge
//------------------------------------------------------------------------

template<typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessIterator3, typename _Compare,
         typename _LeafMerge>
void parallel_merge(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp, _LeafMerge __leaf_merge) {
    if ((__xe - __xs) + (__ye - __ys) <= __PSTL_MERGE_CUT_OFF) {
        // Fall back on serial merge
        __leaf_merge(__xs, __xe, __ys, __ye, __zs, __comp);
    }
    else {
        tbb::this_task_arena::isolate([=]() {
            typedef merge_task<_RandomAccessIterator1, _RandomAccessIterator2, _RandomAccessIterator3, _Compare,
                               par_backend::binary_no_op, _LeafMerge> _task_type;
            tbb::task::spawn_root_and_wait(*new(tbb::task::allocate_root()) _task_type(__xs, __xe, __ys, __ye, __zs,
                                                                                       __comp, par_backend::binary_no_op(),
                                                                                       __leaf_merge));
        });
    }
}

template<typename DifferenceType, typename IteratorType>
struct partial_sort_range {
    DifferenceType _M_beg, _M_end;
    bool _M_buf_flag;
    IteratorType _M_buf_0;
    typename std::iterator_traits<IteratorType>::value_type* _M_buf_1;
    DifferenceType _M_mid;
};

template<typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessIterator3>
struct partial_ranges {
    _RandomAccessIterator1 _M_xs, _M_xe;
    _RandomAccessIterator2 _M_ys, _M_ye;
    _RandomAccessIterator3 _M_zs;
    _RandomAccessIterator1 _M_xm;
    _RandomAccessIterator2 _M_ym;
};

template<typename _DifferenceType>
struct range_move_t {
    _DifferenceType _M_xs, _M_xe;
    _DifferenceType _M_zs;
};

//partial sorting based on parallel reduce and parallel merge
template<typename _RandomAccessIterator, typename _Compare>
void parallel_partial_sort(_RandomAccessIterator __xs, _RandomAccessIterator __xm, _RandomAccessIterator __xe, _Compare __comp) {

    //assumption that parallelization overhead affects till number of sorting elements up to 1000
    const size_t __sort_cut_off = 1000;

    const auto __n = __xe - __xs;

    //partial sorting cut off
    if (__n <= __sort_cut_off*2) {
        std::partial_sort(__xs, __xm, __xe, __comp);
        return;
    }

    //trying to request additional memory
    typedef typename std::iterator_traits<_RandomAccessIterator>::value_type _value_type;
    buffer<_value_type> __buf(__n);
    if (!__buf) {
        std::partial_sort(__xs, __xm, __xe, __comp);
        return;
    }

    //prepare subranges to call partial_sort in parallel mode
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _DifferenceType;
    typedef par_backend::partial_sort_range<_DifferenceType, _RandomAccessIterator> _range_t;
    typedef par_backend::range_move_t<_DifferenceType> _rmove_t;

    const auto __n1 = __xm - __xs;
    const auto __n2 = __xe - __xm;

    assert(__n1 + __n2 == __n);

    const auto __m = std::max(__n1, __n2);
    const auto __n_range = __m / __sort_cut_off;

    assert(__n_range >= 1);

    const double __fpart = 1. / __n_range;
    const double __fpart1 = __fpart * __n1;
    const double __fpart2 = __fpart * __n2;

    _RandomAccessIterator __a = __xs; //a - source container
    _value_type* __b = __buf.get(); //b - buffer for merging partial sorted arrays

    //calculation indices for doing subranges for parallel partial sorting
    par_backend::stack <par_backend::buffer<_range_t>> __ranges(__n_range);
    par_backend::stack <par_backend::buffer<_rmove_t>> __ranges_move(__n_range * 2);

    _DifferenceType __x1 = 0, __y1 = __n1, __z = 0, __z1 = 0, __z2 = 0, __zm = 0;
    for (_DifferenceType __i = 1; __i <= __n_range; ++__i) {

        //create subrange indices for dividing original arrays into a couple of range sets
        const _DifferenceType __x2 = __i < __n_range ? __fpart1 * __i : __n1;
        const _DifferenceType __y2 = __i < __n_range ? __n1 + __fpart2 * __i : __n1 + __n2;
        __z1 = __z;
        __ranges_move.push(_rmove_t{__x1, __x2, __z });
        __z += __x2 - __x1;
        __zm = __z;
        __ranges_move.push(_rmove_t{__y1, __y2, __z });
        __z += __y2-__y1;
        __z2 = __z;
        __x1 = __x2, __y1 = __y2;

        //create subrange indices for partial sort
        __ranges.push(_range_t{ __z1, __z2, true, __a, __b, __zm});
    }

    assert(__z2 == __n1 + __n2);

    //init buffer -moving from two array into one
    par_backend::parallel_for(__ranges_move.buffer().get(), __ranges_move.buffer().get() + __ranges_move.size(),
        [&__a, &__b](_rmove_t* __i, _rmove_t* __j) {
            for (; __i < __j; ++__i) {
                const auto& __r = *__i;
                par_backend::init_buf(__a + __r._M_xs, __a + __r._M_xe, __b + __r._M_zs, true);
            }
        }
    );

    auto res_range = tbb::parallel_deterministic_reduce(tbb::blocked_range<_range_t*>(__ranges.buffer().get(),
                                                                                      __ranges.buffer().get() + __ranges.size(), 1),
                                                        _range_t{ 0, 0, false , __a, __b, 0},
        [__a, __b, &__comp](tbb::blocked_range<_range_t*>& __r, const _range_t&)-> _range_t {
            assert(__r.end() - __r.begin() == 1);

            auto& __sr = *__r.begin();
            std::partial_sort(__b + __sr._M_beg, __b + __sr._M_mid, __b + __sr._M_end, __comp);
            return __sr;
        },
        [&__comp](_range_t __l, _range_t __r) -> _range_t {

            assert(__l._M_end - __l._M_beg > 0);
            assert(__r._M_end - __r._M_beg > 0);
            assert(__l._M_end == __r._M_beg);

            //move the subrange to the paired buffer and revert the buffer flag
            if (__l._M_buf_flag != __r._M_buf_flag) {
                if (__l._M_buf_flag) {
                    std::move(__l._M_buf_1 + __l._M_beg, __l._M_buf_1 + __l._M_end, __l._M_buf_0 + __l._M_beg);
                    __l._M_buf_flag = false;
                }
                else {
                    assert(__r._M_buf_flag);

                    std::move(__r._M_buf_1 + __r._M_beg, __r._M_buf_1 + __r._M_end, __r._M_buf_0 + __r._M_beg);
                    __r._M_buf_flag = false;
                }
            }

            assert(__l._M_buf_flag == __r._M_buf_flag);

            //merge two partial sorted ranges
            const auto __n1 = __l._M_mid - __l._M_beg;
            const auto __n2 = __r._M_mid - __r._M_beg;
            const auto __res = _range_t{ __l._M_beg, __r._M_end, !__l._M_buf_flag, __l._M_buf_0, __l._M_buf_1,
                                         __l._M_beg + __n1 + __n2}; //get new range and switch memory buffer
            if (__l._M_buf_flag) {
                par_backend::parallel_merge(__l._M_buf_1 + __l._M_beg, __l._M_buf_1 + __l._M_mid, __r._M_buf_1 + __r._M_beg,
                                            __r._M_buf_1 + __r._M_mid, __res._M_buf_0 + __res._M_beg, __comp,
                                            par_backend::serial_move_merge());
                auto __res_it = std::move(__l._M_buf_1 + __l._M_mid, __l._M_buf_1 + __l._M_end,
                                          __res._M_buf_0 + __res._M_beg + __n1 + __n2); //moving left unsorted subrange
                std::move(__r._M_buf_1 + __r._M_mid, __r._M_buf_1 + __r._M_end, __res_it); //moving right unsorted subrange
            }
            else {
                par_backend::parallel_merge(__l._M_buf_0 + __l._M_beg, __l._M_buf_0 + __l._M_mid, __r._M_buf_0 + __r._M_beg,
                                            __r._M_buf_0 + __r._M_mid, __res._M_buf_1 + __res._M_beg, __comp,
                                            par_backend::serial_move_merge());
                auto __res_it = std::move(__l._M_buf_0 + __l._M_mid, __l._M_buf_0 + __l._M_end,
                                          __res._M_buf_1 + __res._M_beg + __n1 + __n2);//moving left unsorted subrange
                std::move(__r._M_buf_0 + __r._M_mid, __r._M_buf_0 + __r._M_end, __res_it); //moving right unsorted subrange
            }

            return __res;
        },
        tbb::simple_partitioner()
    );

    //move the result into source container
    if (res_range._M_buf_flag) {
        std::move(__b, __b + __n, __a);
    }

    serial_destroy()(__b, __b + __n); //cleanup
}

} // namespace par_backend
} // namespace __pstl

#endif /* __PSTL_parallel_backend_tbb_H */
