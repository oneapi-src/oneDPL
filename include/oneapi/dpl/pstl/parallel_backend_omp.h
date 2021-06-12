// -*- C++ -*-
// -*-===----------------------------------------------------------------------===//
//
// Copyright (C) Christopher Nelson
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_OMP_H
#define _ONEDPL_PARALLEL_BACKEND_OMP_H

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>
#include <omp.h>

#include "parallel_backend_serial.h"
#include "unseq_backend_simd.h"
#include "utils.h"

#if !defined(_OPENMP)
#    error _OPENMP not defined.
#endif

// Portability "#pragma" definition
#ifdef _MSC_VER
#    define _PSTL_PRAGMA(x) __pragma(x)
#else
#    define _PSTL_PRAGMA(x) _Pragma(#    x)
#endif

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

//------------------------------------------------------------------------
// use to cancel execution
//------------------------------------------------------------------------
inline void
__cancel_execution()
{
}

//------------------------------------------------------------------------
// raw buffer
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Tp>
class __buffer
{
    std::allocator<_Tp> __allocator_;
    _Tp* __ptr_;
    const std::size_t __buf_size_;
    __buffer(const __buffer&) = delete;
    void
    operator=(const __buffer&) = delete;

  public:
    __buffer(std::size_t __n) : __allocator_(), __ptr_(__allocator_.allocate(__n)), __buf_size_(__n) {}

    operator bool() const { return __ptr_ != nullptr; }

    _Tp*
    get() const
    {
        return __ptr_;
    }
    ~__buffer() { __allocator_.deallocate(__ptr_, __buf_size_); }
};

// Preliminary size of each chunk: requires further discussion
constexpr std::size_t __default_chunk_size = 512;

// The iteration space partitioner according to __requested_chunk_size
template <class _RandomAccessIterator, class _Size>
void
__chunk_partitioner(_RandomAccessIterator __first, _RandomAccessIterator __last, _Size& __n_chunks, _Size& __chunk_size,
                    _Size& __first_chunk_size, _Size __requested_chunk_size = __default_chunk_size)
{
    /*
   * This algorithm improves distribution of elements in chunks by avoiding
   * small tail chunks. The leftover elements that do not fit neatly into
   * the chunk size are redistributed to early chunks. This improves
   * utilization of the processor's prefetch and reduces the number of
   * tasks needed by 1.
   */

    const _Size __n = __last - __first;
    if (__n < __requested_chunk_size)
    {
        __chunk_size = __n;
        __first_chunk_size = __n;
        __n_chunks = 1;
        return;
    }

    __n_chunks = (__n / __requested_chunk_size) + 1;
    __chunk_size = __n / __n_chunks;
    const _Size __n_leftover_items = __n % __chunk_size;

    if (__n_leftover_items == 0)
    {
        __first_chunk_size = __chunk_size;
        return;
    }

    const _Size __n_extra_items_per_chunk = __n_leftover_items / __n_chunks;
    const _Size __n_final_leftover_items = __n_leftover_items % __n_chunks;

    __chunk_size += __n_extra_items_per_chunk;
    __first_chunk_size = __chunk_size + __n_final_leftover_items;
}

//------------------------------------------------------------------------
// parallel_invoke
//------------------------------------------------------------------------

template <typename _F1, typename _F2>
void
__parallel_invoke_body(_F1&& __f1, _F2&& __f2)
{
    _PSTL_PRAGMA(omp taskgroup)
    {
        _PSTL_PRAGMA(omp task) { std::forward<_F1>(__f1)(); }
        _PSTL_PRAGMA(omp task) { std::forward<_F2>(__f2)(); }
    }
}

template <class _ExecutionPolicy, typename _F1, typename _F2>
void
__parallel_invoke(_ExecutionPolicy&&, _F1&& __f1, _F2&& __f2)
{
    if (omp_in_parallel())
    {
        __parallel_invoke_body(std::forward<_F1>(__f1), std::forward<_F2>(__f2));
    }
    else
    {
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        __parallel_invoke_body(std::forward<_F1>(__f1), std::forward<_F2>(__f2));
    }
}

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

template <class _RandomAccessIterator, class _Fp>
void
__parallel_for_body(_RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f)
{
    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __chunk_partitioner(__first, __last, __n_chunks, __chunk_size, __first_chunk_size);

    // To avoid over-subscription we use taskloop for the nested parallelism
    _PSTL_PRAGMA(omp taskloop)
    for (std::size_t __chunk = 0; __chunk < __n_chunks; ++__chunk)
    {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = __first + __index;
        auto __end = __begin + __this_chunk_size;
        __f(__begin, __end);
    }
}

//------------------------------------------------------------------------
// Notation:
// Evaluation of brick f[i,j) for each subrange [i,j) of [first, last)
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Fp>
void
__parallel_for(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f)
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        dpl::__omp_backend::__parallel_for_body(__first, __last, __f);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single) { dpl::__omp_backend::__parallel_for_body(__first, __last, __f); }
    }
}

//------------------------------------------------------------------------
// parallel_for_each
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
void
__parallel_for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Fp __f)
{
    dpl::__omp_backend::__parallel_for(std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
}

//------------------------------------------------------------------------
// parallel_reduce
//------------------------------------------------------------------------

template <class _Value, typename _ChunkReducer, typename _Reduction>
auto
__parallel_reduce_chunks(std::uint32_t start, std::uint32_t end, _ChunkReducer __reduce_chunk, _Reduction __reduce)
    -> _Value
{
    _Value v1, v2;

    if (end - start == 1)
    {
        return __reduce_chunk(start);
    }
    else if (end - start == 2)
    {
        _PSTL_PRAGMA(omp task shared(v1))
        v1 = __reduce_chunk(start);

        _PSTL_PRAGMA(omp task shared(v2))
        v2 = __reduce_chunk(start + 1);
    }
    else
    {
        auto middle = start + ((end - start) / 2);

        _PSTL_PRAGMA(omp task shared(v1))
        v1 = __parallel_reduce_chunks<_Value>(start, middle, __reduce_chunk, __reduce);

        _PSTL_PRAGMA(omp task shared(v2))
        v2 = __parallel_reduce_chunks<_Value>(middle, end, __reduce_chunk, __reduce);
    }

    _PSTL_PRAGMA(omp taskwait)
    return __reduce(v1, v2);
}

template <class _RandomAccessIterator, class _Value, typename _RealBody, typename _Reduction>
_Value
__parallel_reduce_body(_RandomAccessIterator __first, _RandomAccessIterator __last, _Value __identity,
                       _RealBody __real_body, _Reduction __reduction)
{

    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __omp_backend::__chunk_partitioner(__first, __last, __n_chunks, __chunk_size, __first_chunk_size);

    auto __reduce_chunk = [&](std::uint32_t __chunk)
    {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = __first + __index;
        auto __end = __begin + __this_chunk_size;

        //IMPORTANT: __real_body call does a serial reduction based on an initial value;
        //in case of passing an identity value, a partial result should be explicitly combined
        //with the previous partial reduced value.

        return __real_body(__begin, __end, __identity);
    };

    return __parallel_reduce_chunks<_Value>(0, __n_chunks, __reduce_chunk, __reduction);
}

//------------------------------------------------------------------------
// Notation:
//      r(i,j,init) returns reduction of init with reduction over [i,j)
//      c(x,y) combines values x and y that were the result of r
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Value, typename _RealBody, typename _Reduction>
_Value
__parallel_reduce(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Value __identity,
                  _RealBody __real_body, _Reduction __reduction)
{
    if (__first == __last)
        return __identity;

    // Don't bother parallelizing the work if the size is too small.
    if (__last - __first < static_cast<long>(__default_chunk_size))
    {
        return __real_body(__first, __last, __identity);
    }

    // We don't create a nested parallel region in an existing parallel region:
    // just create tasks.
    if (omp_in_parallel())
    {
        return dpl::__omp_backend::__parallel_reduce_body(__first, __last, __identity, __real_body, __reduction);
    }

    // In any case (nested or non-nested) one parallel region is created and only
    // one thread creates a set of tasks.
    _Value __res = __identity;

    _PSTL_PRAGMA(omp parallel)
    _PSTL_PRAGMA(omp single)
    {
        __res = dpl::__omp_backend::__parallel_reduce_body(__first, __last, __identity, __real_body, __reduction);
    }

    return __res;
}

//------------------------------------------------------------------------
// parallel_transform_reduce
//
// Notation:
//      r(i,j,init) returns reduction of init with reduction over [i,j)
//      u(i) returns f(i,i+1,identity) for a hypothetical left identity element
//      of r c(x,y) combines values x and y that were the result of r or u
//------------------------------------------------------------------------

template <class _RandomAccessIterator, class _UnaryOp, class _Value, class _Combiner, class _Reduction>
auto
__transform_reduce_body(_RandomAccessIterator __first, _RandomAccessIterator __last, _UnaryOp __unary_op, _Value __init,
                        _Combiner __combiner, _Reduction __reduction)
{
    using _Size = std::size_t;
    const _Size __num_threads = omp_get_num_threads();
    const _Size __n = __last - __first;

    if (__n >= __num_threads)
    {
        // Here, we cannot use OpenMP UDR because we must store the init value in
        // the combiner and it will be used several times. Although there should be
        // the only one; we manually generate the identity elements for each thread.
        alignas(_Value) char __accums_storage[__num_threads * sizeof(_Value)];
        _Value* __accums = reinterpret_cast<_Value*>(__accums_storage);

        // initialize accumulators for all threads
        for (_Size __i = 0; __i < __num_threads; ++__i)
        {
            ::new (__accums + __i) _Value(__unary_op(__first + __i));
        }

        // initial partition of the iteration space into chunks
        std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
        __omp_backend::__chunk_partitioner(__first + __num_threads, __last, __n_chunks, __chunk_size,
                                           __first_chunk_size);

        // main loop
        _PSTL_PRAGMA(omp taskloop)
        for (std::size_t __chunk = 0; __chunk < __n_chunks; ++__chunk)
        {
            auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
            auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
            auto __begin = __first + __index + __num_threads;
            auto __end = __begin + __this_chunk_size;

            auto __thread_num = omp_get_thread_num();
            __accums[__thread_num] = __reduction(__begin, __end, __accums[__thread_num]);
        }

        // combine by accumulators
        for (_Size __i = 0; __i < __num_threads; ++__i)
        {
            __init = __combiner(__init, __accums[__i]);
        }

        // destroy accumulators
        for (_Size __i = 0; __i < __num_threads; ++__i)
        {
            __accums[__i].~_Value();
        }
    }
    else
    { // if the number of elements is less than the number of threads, we
        // process them sequentially
        for (_Size __i = 0; __i < __n; ++__i)
        {
            __init = __combiner(__init, __unary_op(__first + __i));
        }
    }

    return __init;
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryOp, class _Value, class _Combiner,
          class _Reduction>
_Value
__parallel_transform_reduce(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last,
                            _UnaryOp __unary_op, _Value __init, _Combiner __combiner, _Reduction __reduction)
{

    if (__first == __last)
    {
        return __init;
    }

    _Value __result = __init;
    if (omp_in_parallel())
    {
        // We don't create a nested parallel region in an existing parallel
        // region: just create tasks
        __result = dpl::__omp_backend::__transform_reduce_body(__first, __last, __unary_op, __init, __combiner,
                                                                  __reduction);
    }
    else
    {
        // Create a parallel region, and a single thread will create tasks
        // for the region.
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        {
            __result = dpl::__omp_backend::__transform_reduce_body(__first, __last, __unary_op, __init, __combiner,
                                                                      __reduction);
        }
    }

    return __result;
}

//------------------------------------------------------------------------
// parallel_scan
//------------------------------------------------------------------------

template <typename _Index>
_Index
__split(_Index __m)
{
    _Index __k = 1;
    while (2 * __k < __m)
        __k *= 2;
    return __k;
}

template <typename _Index, typename _Tp, typename _Rp, typename _Cp>
void
__upsweep(_Index __i, _Index __m, _Index __tilesize, _Tp* __r, _Index __lastsize, _Rp __reduce, _Cp __combine)
{
    if (__m == 1)
        __r[0] = __reduce(__i * __tilesize, __lastsize);
    else
    {
        _Index __k = __split(__m);
        __omp_backend::__parallel_invoke_body(
            [=] { __omp_backend::__upsweep(__i, __k, __tilesize, __r, __tilesize, __reduce, __combine); },
            [=] {
                __omp_backend::__upsweep(__i + __k, __m - __k, __tilesize, __r + __k, __lastsize, __reduce, __combine);
            });
        if (__m == 2 * __k)
            __r[__m - 1] = __combine(__r[__k - 1], __r[__m - 1]);
    }
}

template <typename _Index, typename _Tp, typename _Cp, typename _Sp>
void
__downsweep(_Index __i, _Index __m, _Index __tilesize, _Tp* __r, _Index __lastsize, _Tp __initial, _Cp __combine,
            _Sp __scan)
{
    if (__m == 1)
        __scan(__i * __tilesize, __lastsize, __initial);
    else
    {
        const _Index __k = __split(__m);
        __omp_backend::__parallel_invoke_body(
            [=] { __omp_backend::__downsweep(__i, __k, __tilesize, __r, __tilesize, __initial, __combine, __scan); },
            // Assumes that __combine never throws.
            // TODO: Consider adding a requirement for user functors to be constant.
            [=, &__combine]
            {
                __omp_backend::__downsweep(__i + __k, __m - __k, __tilesize, __r + __k, __lastsize,
                                           __combine(__initial, __r[__k - 1]), __combine, __scan);
            });
    }
}

template <typename _ExecutionPolicy, typename _Index, typename _Tp, typename _Rp, typename _Cp, typename _Sp, typename _Ap>
void
__parallel_strict_scan_body(_ExecutionPolicy&&, _Index __n, _Tp __initial, _Rp __reduce, _Cp __combine, _Sp __scan, _Ap __apex)
{
    _Index __p = omp_get_num_threads();
    const _Index __slack = 4;
    _Index __tilesize = (__n - 1) / (__slack * __p) + 1;
    _Index __m = (__n - 1) / __tilesize;
    __buffer<_ExecutionPolicy, _Tp> __buf(__m + 1);
    _Tp* __r = __buf.get();

    __omp_backend::__upsweep(_Index(0), _Index(__m + 1), __tilesize, __r, __n - __m * __tilesize, __reduce, __combine);

    std::size_t __k = __m + 1;
    _Tp __t = __r[__k - 1];
    while ((__k &= __k - 1))
    {
        __t = __combine(__r[__k - 1], __t);
    }

    __apex(__combine(__initial, __t));
    __omp_backend::__downsweep(_Index(0), _Index(__m + 1), __tilesize, __r, __n - __m * __tilesize, __initial,
                               __combine, __scan);
}

template <class _ExecutionPolicy, typename _Index, typename _Tp, typename _Rp, typename _Cp, typename _Sp, typename _Ap>
void
__parallel_strict_scan(_ExecutionPolicy&& __exec, _Index __n, _Tp __initial, _Rp __reduce, _Cp __combine, _Sp __scan,
                       _Ap __apex)
{
    if (__n <= 1)
    {
        __serial_backend::__parallel_strict_scan(std::forward<_ExecutionPolicy>(__exec), __n, __initial, __reduce,
                                                 __combine, __scan, __apex);
        return;
    }

    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        dpl::__omp_backend::__parallel_strict_scan_body(std::forward<_ExecutionPolicy>(__exec),__n, __initial, __reduce, __combine, __scan, __apex);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        {
            dpl::__omp_backend::__parallel_strict_scan_body(std::forward<_ExecutionPolicy>(__exec),__n, __initial, __reduce, __combine, __scan, __apex);
        }
    }
}

template <class _ExecutionPolicy, class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
_Tp
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Index __n, _Up __u, _Tp __init, _Cp __combine, _Rp __brick_reduce,
                          _Sp __scan)
{

    return __serial_backend::__parallel_transform_scan(std::forward<_ExecutionPolicy>(__exec), __n, __u, __init,
                                                       __combine, __brick_reduce, __scan);
}

//------------------------------------------------------------------------
// parallel_stable_sort
//------------------------------------------------------------------------

template<typename _RandomAccessIterator, typename _Compare>
struct _MinKOp {
    std::vector<_RandomAccessIterator> &__items;
    _Compare __comp;

    _MinKOp(std::vector<_RandomAccessIterator> &__items_, _Compare __comp_)
        : __items(__items_), __comp(__comp_) {}

    auto __it_comp() const {
        return [this](const auto &l, const auto &r) { return __comp(*l, *r); };
    }

    void __keep_smallest_k_items(_RandomAccessIterator __item) {
        // Put the new item on the heap and re-establish the heap invariant.
        __items.push_back(__item);
        std::push_heap(__items.begin(),
                       __items.end(), __it_comp());

        // Pop the largest item off the heap.
        std::pop_heap(__items.begin(),
                      __items.end(), __it_comp());
        __items.pop_back();
    };

    void __merge(std::vector<_RandomAccessIterator> &__other) {
        for (auto __it = std::begin(__other);
             __it != std::end(__other); ++__it) {
            __keep_smallest_k_items(*__it);
        }
    }

    void __initialize(_RandomAccessIterator __first, _RandomAccessIterator __last,
                      std::size_t __k) {
        __items.resize(__k);
        auto __item_it = __first;
        auto __tracking_it = std::begin(__items);
        while (__item_it != __last &&
            __tracking_it != std::end(__items)) {
            *__tracking_it = __item_it;
            ++__item_it;
            ++__tracking_it;
        }
        std::make_heap(__items.begin(),
                       __items.end(), __it_comp());
        for (; __item_it != __last; ++__item_it) {
            __keep_smallest_k_items(__item_it);
        }
    }

    static auto __reduce(std::vector<_RandomAccessIterator> &__v1,
                         std::vector<_RandomAccessIterator> &__v2, _Compare __comp)
    -> std::vector<_RandomAccessIterator> {
        if (__v1.empty()) {
            return __v2;
        }

        if (__v2.empty()) {
            return __v1;
        }

        if (__v1.size() >= __v2.size()) {
            _MinKOp<_RandomAccessIterator, _Compare> __op(__v1, __comp);
            __op.__merge(__v2);
            return __v1;
        }

        _MinKOp<_RandomAccessIterator, _Compare> __op(__v2, __comp);
        __op.__merge(__v1);
        return __v2;
    }
};

template<typename _RandomAccessIterator, typename _Compare>
auto __find_min_k(_RandomAccessIterator __first, _RandomAccessIterator __last,
                  std::size_t __k, _Compare __comp)
-> std::vector<_RandomAccessIterator> {
    std::vector<_RandomAccessIterator> __items;
    _MinKOp<_RandomAccessIterator, _Compare> op(__items, __comp);

    op.__initialize(__first, __last, __k);
    return __items;
}

template<typename _RandomAccessIterator, typename _Compare>
auto __parallel_find_pivot(_RandomAccessIterator __first,
                           _RandomAccessIterator __last, _Compare __comp,
                           std::size_t __nsort) -> _RandomAccessIterator {
    using _Value = std::vector<_RandomAccessIterator>;
    using _Op = _MinKOp<_RandomAccessIterator, _Compare>;

    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __chunk_partitioner(__first, __last, __n_chunks, __chunk_size,
                        __first_chunk_size,
                        std::max(__nsort, __default_chunk_size));
    /*
     * This function creates a vector of iterators to the container being operated
     * on. It splits that container into fixed size chunks, just like other
     * functions in this backend. For each chunk it finds the smallest k items.
     * The chunks are run through a reducer which keeps the smallest k items from
     * each chunk. Finally, the largest item from the merged chunks is returned as
     * the pivot.
     *
     * The chunks are partitioned in such a way that there will always be at least
     * k items in one chunk. The reducer will always produce a chunk merge so that
     * the longest k items list propagates out. So even if some of the chunks are
     * less than __nsort elements, at least one chunk will be and this chunk will
     * end up producing a correctly sized smallest k items list.
     *
     * Note that the case where __nsort == distance(__first, __last) is handled by
     * performing a complete sort of the container, so we don't have to handle
     * that here.
     */

    auto __reduce_chunk = [&](std::uint32_t __chunk) {
      auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
      auto __index = __chunk == 0 ? 0
                                  : (__chunk * __chunk_size) +
              (__first_chunk_size - __chunk_size);
      auto __begin = std::next(__first, __index);
      auto __end = std::next(__begin, __this_chunk_size);

      return __find_min_k(__begin, __end, __nsort, __comp);
    };

    auto __reduce_value = [&](auto &__v1, auto &__v2) {
      return _Op::__reduce(__v1, __v2, __comp);
    };
    auto __result = __parallel_reduce_chunks<_Value>(
        0, __n_chunks, __reduce_chunk, __reduce_value);

    // Return largest item
    return __result.front();
}

template <typename _RandomAccessIterator, typename _Compare>
void
__parallel_partition(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _RandomAccessIterator __pivot,
                     _Compare __comp, std::size_t __nsort)
{
    auto __size = static_cast<std::size_t>(std::distance(__xs, __xe));
    std::atomic_bool* __status = new std::atomic_bool[__size];

    /*
     * First, walk through the entire array and mark items that are on the
     * correct side of the pivot as true, and the others as false.
     */
    _PSTL_PRAGMA(omp taskloop shared(__status))
    for (std::size_t __index = 0U; __index < __size; ++__index)
    {
        auto __item = std::next(__xs, __index);
        if (__index < __nsort)
        {
            __status[__index].store(__comp(*__item, *__pivot));
        }
        else
        {
            __status[__index].store(__comp(*__pivot, *__item));
        }
    }

    /*
     * Second, walk through the first __nsort items of the array and move
     * any items that are not in the right place. The status array is used
     * to locate places outside the partition where values can be safely
     * swapped.
     */
    _PSTL_PRAGMA(omp taskloop shared(__status))
    for (std::size_t __index = 0U; __index < __nsort; ++__index)
    {
        // If the item is already in the right place, move along.
        if (__status[__index].load())
        {
            continue;
        }

        // Otherwise, find the an item that can be moved into this
        // spot safely.
        for (std::size_t __swap_index = __nsort; __swap_index < __size; ++__swap_index)
        {
            // Try to capture this slot by using compare and exchange. If we
            // are able to capture the slot then perform a swap and exit this
            // loop.
            if (__status[__swap_index].load() == false && __status[__swap_index].exchange(true) == false)
            {
                auto __current_item = std::next(__xs, __index);
                auto __swap_item = std::next(__xs, __swap_index);
                std::iter_swap(__current_item, __swap_item);
                break;
            }
        }
    }

    delete[] __status;
}

template <typename _RandomAccessIterator, typename _Compare>
void
__parallel_stable_sort_body(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp)
{
    std::size_t __size = std::distance(__xs, __xe);
    if (__size == 0)
    {
        return;
    }

    auto __left_it = __xs;
    auto __right_it = __xe;
    bool __is_swapped_left = false, __is_swapped_right = false;
    auto __pivot = *__xs;

    auto __forward_it = __xs + 1;
    while (__forward_it <= __right_it)
    {
        if (__comp(*__forward_it, __pivot))
        {
            __is_swapped_left = true;
            std::iter_swap(__left_it, __forward_it);
            __left_it++;
            __forward_it++;
        }
        else if (__comp(__pivot, *__forward_it))
        {
            __is_swapped_right = true;
            std::iter_swap(__right_it, __forward_it);
            __right_it--;
        }
        else
        {
            __forward_it++;
        }
    }

    if (__size >= __default_chunk_size)
    {
        _PSTL_PRAGMA(omp taskgroup)
        {
            _PSTL_PRAGMA(omp task untied mergeable)
            {
                if (std::distance(__xs, __left_it) > 0 && __is_swapped_left)
                {
                    __parallel_stable_sort_body(__xs, __left_it - 1, __comp);
                }
            }

            _PSTL_PRAGMA(omp task untied mergeable)
            {
                if (std::distance(__right_it, __xe) && __is_swapped_right)
                {
                    __parallel_stable_sort_body(__right_it + 1, __xe, __comp);
                }
            }
        }
    }
    else
    {
        _PSTL_PRAGMA(omp task untied mergeable)
        {
            if (std::distance(__xs, __left_it) > 0 && __is_swapped_left)
            {
                __parallel_stable_sort_body(__xs, __left_it - 1, __comp);
            }

            if (std::distance(__right_it, __xe) && __is_swapped_right)
            {
                __parallel_stable_sort_body(__right_it + 1, __xe, __comp);
            }
        }
    }
}

template <typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_partial_sort(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp,
                               _LeafSort __leaf_sort, std::size_t __nsort)
{
    auto __pivot = __parallel_find_pivot(__xs, __xe, __comp, __nsort);
    __parallel_partition(__xs, __xe, __pivot, __comp, __nsort);
    auto __part_end = std::next(__xs, __nsort);

    if (__nsort <= __default_chunk_size)
    {
        __leaf_sort(__xs, __part_end, __comp);
    }
    else
    {
        __parallel_stable_sort_body(__xs, __part_end, __comp);
    }
}

template <class _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __xs, _RandomAccessIterator __xe,
                       _Compare __comp, _LeafSort __leaf_sort, std::size_t __nsort = 0)
{
    if (__xs >= __xe)
    {
        return;
    }

    if (__nsort <= __default_chunk_size)
    {
        __serial_backend::__parallel_stable_sort(std::forward<_ExecutionPolicy>(__exec), __xs, __xe, __comp,
                                                 __leaf_sort, __nsort);
        return;
    }

    std::size_t __count = static_cast<std::size_t>(std::distance(__xs, __xe));

    if (omp_in_parallel())
    {
        if (__count <= __nsort)
        {
            __parallel_stable_sort_body(__xs, __xe, __comp);
        }
        else
        {
            __parallel_stable_partial_sort(__xs, __xe, __comp, __leaf_sort, __nsort);
        }
    }
    else
    {
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        if (__count <= __nsort)
        {
            __parallel_stable_sort_body(__xs, __xe, __comp);
        }
        else
        {
            __parallel_stable_partial_sort(__xs, __xe, __comp, __leaf_sort, __nsort);
        }
    }
}

//------------------------------------------------------------------------
// parallel_merge
//------------------------------------------------------------------------

template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3, typename _Compare, typename _LeafMerge>
void
__parallel_merge(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe,
                 _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp,
                 _LeafMerge __leaf_merge)

{
    __serial_backend::__parallel_merge(std::forward<_ExecutionPolicy>(__exec), __xs, __xe, __ys, __ye, __zs, __comp,
                                       __leaf_merge);
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi

#endif //
