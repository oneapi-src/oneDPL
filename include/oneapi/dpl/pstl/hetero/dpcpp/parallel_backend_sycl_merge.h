// -*- C++ -*-
//===-- parallel_backend_sycl_merge.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H

#include <limits>    // std::numeric_limits
#include <cassert>   // assert
#include <cstdint>   // std::uint8_t, ...
#include <utility>   // std::make_pair, std::forward
#include <algorithm> // std::min, std::lower_bound
#include <tuple>     // std::tuple

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"

//#define DUMP_DATA_LOADING 1

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename _Index>
using _split_point_t = std::pair<_Index, _Index>;

template <typename _Index>
constexpr _split_point_t<_Index> __zero_split_point{0, 0};

//Searching for an intersection of a merge matrix (n1, n2) diagonal with the Merge Path to define sub-ranges
//to serial merge. For example, a merge matrix for [0,1,1,2,3] and [0,0,2,3] is shown below:
//     0   1  1  2   3
//    ------------------
//   |--->
// 0 | 0 | 1  1  1   1
//   |   |
// 0 | 0 | 1  1  1   1
//   |   ---------->
// 2 | 0   0  0  0 | 1
//   |             ---->
// 3 | 0   0  0  0   0 |
template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
_split_point_t<_Index>
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __i_elem, const _Index __n1,
                   const _Index __n2, _Compare __comp)
{
    constexpr _Index __rng1_from = 0;
    constexpr _Index __rng2_from = 0;

    const _Index __rng1_to = __n1;
    const _Index __rng2_to = __n2;

    assert(__rng1_from <= __rng1_to);
    assert(__rng2_from <= __rng2_to);

    assert(__rng1_to > 0 || __rng2_to > 0);

    if constexpr (!std::is_pointer_v<_Rng1>)
        assert(__rng1_to <= __rng1.size());
    if constexpr (!std::is_pointer_v<_Rng2>)
        assert(__rng2_to <= __rng2.size());

    assert(__i_elem >= 0);

    // ----------------------- EXAMPLE ------------------------
    // Let's consider the following input data:
    //    rng1.size() = 10
    //    rng2.size() = 6
    //    i_diag = 9
    // Let's define the following ranges for processing:
    //    rng1: [3, ..., 9) -> __rng1_from = 3, __rng1_to = 9
    //    rng2: [1, ..., 4) -> __rng2_from = 1, __rng2_to = 4
    //
    // The goal: required to process only X' items of the merge matrix
    //           as intersection of rng1[3, ..., 9) and rng2[1, ..., 4)
    //
    // --------------------------------------------------------
    //
    //         __diag_it_begin(rng1)            __diag_it_end(rng1)
    //      (init state) (dest state)          (init state, dest state)
    //            |          |                       |
    //            V          V                       V
    //                       +   +   +   +   +   +
    //    \ rng1  0  1   2   3   4   5   6   7   8   9
    //   rng2   +--------------------------------------+
    //    0     |                    ^   ^   ^   X     |     <--- __diag_it_end(rng2) (init state)
    // +  1     | <----------------- +   +   X'2 ^     |     <--- __diag_it_end(rng2) (dest state)
    // +  2     | <----------------- +   X'1     |     |
    // +  3     | <----------------- X'0         |     |     <--- __diag_it_begin(rng2) (dest state)
    //    4     |                X   ^           |     |
    //    5     |            X       |           |     |     <--- __diag_it_begin(rng2) (init state)
    //          +-------AX-----------+-----------+-----+
    //              AX               |           |
    //           AX                  |           |
    //              Run lower_bound:[from = 5,   to = 8)
    //
    //  AX - absent items in rng2
    //
    //  We have three points on diagonal for call comparison:
    //      X'0 : call __comp(rng1[5], rng2[3])             // 5 + 3 == 9 - 1 == 8
    //      X'1 : call __comp(rng1[6], rng2[2])             // 6 + 2 == 9 - 1 == 8
    //      X'3 : call __comp(rng1[7], rng2[1])             // 7 + 1 == 9 - 1 == 8
    //   - where for every comparing pairs idx(rng1) + idx(rng2) == i_diag - 1

    ////////////////////////////////////////////////////////////////////////////////////
    // Process the corner case: for the first diagonal with the index 0 split point
    // is equal to (0, 0) regardless of the size and content of the data.
    if (__i_elem > 0)
    {
        ////////////////////////////////////////////////////////////////////////////////////
        // Taking into account the specified constraints of the range of processed data
        const auto __index_sum = __i_elem - 1;

        using _IndexSigned = std::make_signed_t<_Index>;

        _IndexSigned idx1_from = __rng1_from;
        _IndexSigned idx1_to = __rng1_to;
        assert(idx1_from <= idx1_to);

        _IndexSigned idx2_from = __index_sum - (__rng1_to - 1);
        _IndexSigned idx2_to = __index_sum - __rng1_from + 1;
        assert(idx2_from <= idx2_to);

        const _IndexSigned idx2_from_diff =
            idx2_from < (_IndexSigned)__rng2_from ? (_IndexSigned)__rng2_from - idx2_from : 0;
        const _IndexSigned idx2_to_diff = idx2_to > (_IndexSigned)__rng2_to ? idx2_to - (_IndexSigned)__rng2_to : 0;

        idx1_to -= idx2_from_diff;
        idx1_from += idx2_to_diff;

        idx2_from = __index_sum - (idx1_to - 1);
        idx2_to = __index_sum - idx1_from + 1;

        assert(idx1_from <= idx1_to);
        assert(__rng1_from <= idx1_from && idx1_to <= __rng1_to);

        assert(idx2_from <= idx2_to);
        assert(__rng2_from <= idx2_from && idx2_to <= __rng2_to);

        ////////////////////////////////////////////////////////////////////////////////////
        // Run search of split point on diagonal

        using __it_t = oneapi::dpl::counting_iterator<_Index>;

        __it_t __diag_it_begin(idx1_from);
        __it_t __diag_it_end(idx1_to);

        constexpr int kValue = 1;
        const __it_t __res =
            std::lower_bound(__diag_it_begin, __diag_it_end, kValue, [&](_Index __idx, const auto& __value) {
                const auto __rng1_idx = __idx;
                const auto __rng2_idx = __index_sum - __idx;

                assert(__rng1_from <= __rng1_idx && __rng1_idx < __rng1_to);
                assert(__rng2_from <= __rng2_idx && __rng2_idx < __rng2_to);
                assert(__rng1_idx + __rng2_idx == __index_sum);

                const auto __zero_or_one = __comp(__rng2[__rng2_idx], __rng1[__rng1_idx]);
                return __zero_or_one < kValue;
            });

        const _split_point_t<_Index> __result = std::make_pair(*__res, __index_sum - *__res + 1);
        assert(__result.first + __result.second == __i_elem);

        assert(__rng1_from <= __result.first && __result.first <= __rng1_to);
        assert(__rng2_from <= __result.second && __result.second <= __rng2_to);

        return __result;
    }

    return __zero_split_point<_Index>;
}

// Do serial merge of the data from rng1 (starting from start1) and rng2 (starting from start2) and writing
// to rng3 (starting from start3) in 'chunk' steps, but do not exceed the total size of the sequences (n1 and n2)
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index, typename _Compare>
void
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3, _Index __start1, _Index __start2,
               const _Index __start3, const std::uint8_t __chunk, const _Index __n1, const _Index __n2, _Compare __comp)
{
    if (__start1 >= __n1)
    {
        //copying a residual of the second seq
        const _Index __n = std::min<_Index>(__n2 - __start2, __chunk);
        for (_Index __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng2[__start2 + __i];
    }
    else if (__start2 >= __n2)
    {
        //copying a residual of the first seq
        const _Index __n = std::min<_Index>(__n1 - __start1, __chunk);
        for (_Index __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng1[__start1 + __i];
    }
    else
    {
        for (_Index __i = 0; __i < __chunk && __start1 < __n1 && __start2 < __n2; ++__i)
        {
            const auto& __val1 = __rng1[__start1];
            const auto& __val2 = __rng2[__start2];
            if (__comp(__val2, __val1))
            {
                __rng3[__start3 + __i] = __val2;
                if (++__start2 == __n2)
                {
                    //copying a residual of the first seq
                    for (++__i; __i < __chunk && __start1 < __n1; ++__i, ++__start1)
                        __rng3[__start3 + __i] = __rng1[__start1];
                }
            }
            else
            {
                __rng3[__start3 + __i] = __val1;
                if (++__start1 == __n1)
                {
                    //copying a residual of the second seq
                    for (++__i; __i < __chunk && __start2 < __n2; ++__i, ++__start2)
                        __rng3[__start3 + __i] = __rng2[__start2];
                }
            }
        }
    }
}

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _IdType, typename _MergeKernelName>
struct __parallel_merge_submitter;

template <typename _IdType, typename... _MergeKernelName>
struct __parallel_merge_submitter<_IdType, __internal::__optional_kernel_name<_MergeKernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        const _IdType __n = __n1 + __n2;

        assert(__n1 > 0 || __n2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // Empirical number of values to process per work-item
        const std::uint8_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : 4;

        const _IdType __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            __cgh.parallel_for<_MergeKernelName...>(
                sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                const _IdType __i_elem = __item_id.get_linear_id() * __chunk;
                    const _split_point_t<_IdType> __start = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __chunk, __n1, __n2,
                               __comp);
            });
        });
        return __future(__event);
    }
};

template <typename _IdType, typename _CustomName, typename _DiagonalsKernelName, typename _MergeKernelName>
struct __parallel_merge_submitter_large;

template <typename _IdType, typename _CustomName, typename... _DiagonalsKernelName, typename... _MergeKernelName>
struct __parallel_merge_submitter_large<_IdType, _CustomName,
                                        __internal::__optional_kernel_name<_DiagonalsKernelName...>,
                                        __internal::__optional_kernel_name<_MergeKernelName...>>
{
#if DUMP_DATA_LOADING
    template <typename _Range, typename _Index, typename _Data>
    static void
    __load_item_into_slm(_Range&& __rng, _Index __idx_from, _Data* __slm, _Index __idx_to, std::size_t __range_index,
                         bool __b_check, std::size_t __group_linear_id, std::size_t __local_id)
    {
        // BP
        //  condition: __b_check
        //  action: __range_index = {__range_index}, __rng[{__idx_from}] -> __slm[{__idx_to}], __group_linear_id = {__group_linear_id}, __local_id = {__local_id}
        //  action: {__range_index}, {__idx_from}, {__idx_to}, {__group_linear_id}, {__local_id}
        __slm[__idx_to] = __rng[__idx_from];
    }
#endif

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        using _Range1ValueType = typename std::iterator_traits<decltype(__rng1.begin())>::value_type;
        using _Range2ValueType = typename std::iterator_traits<decltype(__rng2.begin())>::value_type;
        static_assert(std::is_same_v<_Range1ValueType, _Range2ValueType>, "In this implementation we can merge only data of the same type");

        using _RangeValueType = _Range1ValueType;

        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        const _IdType __n = __n1 + __n2;

#if DUMP_DATA_LOADING
        //const bool __b_check = __n1 == 16144 && __n2 == 8072;
        //const bool __b_check = __n1 == 50716 && __n2 == 25358;      // __wi_in_one_wg = 51 __wg_count = 12
        const bool __b_check = false;

        if (__b_check)
        {
            int i = 0;
            i = i;
        }
#endif

        assert(__n1 > 0 || __n2 > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // Empirical number of values to process per work-item
        const _IdType __chunk = __exec.queue().get_device().is_cpu() ? 128 : 4;
        assert(__chunk > 0);

        // Define SLM bank size
        constexpr std::size_t __slm_bank_size = 32;     // TODO is it correct value? How to get it from hardware?

        // Calculate how many data items we can read into one SLM bank
        constexpr std::size_t __data_items_in_slm_bank = std::max((std::size_t)1, __slm_bank_size / sizeof(_RangeValueType));

        // Pessimistically only use 2/3 of the memory to take into account memory used by compiled kernel
        const auto __slm_adjusted_work_group_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(_RangeValueType));
        const auto __slm_adjusted_work_group_size_x_part = __slm_adjusted_work_group_size * 4 / 5;

        // The amount of data must be a multiple of the chunk size.
        const std::size_t __max_source_data_items_fit_into_slm = __slm_adjusted_work_group_size_x_part - __slm_adjusted_work_group_size_x_part % __chunk;
        assert(__max_source_data_items_fit_into_slm > 0);
        assert(__max_source_data_items_fit_into_slm % __chunk == 0);

        // The amount of items in the each work-group is the amount of diagonals processing between two work-groups + 1 (for the left base diagonal in work-group)
        const std::size_t __wi_in_one_wg = __max_source_data_items_fit_into_slm / __chunk;
        assert(__wi_in_one_wg > 0);

        // The amount of the base diagonals is the amount of the work-groups
        //  - also it's the distance between two base diagonals is equal to the amount of work-items in each work-group
        const std::size_t __wg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk * __wi_in_one_wg);

        // Create storage for save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
        //  - in GLOBAL coordinates
        using __base_diagonals_sp_storage_t = __result_and_scratch_storage<_ExecutionPolicy, _split_point_t<_IdType>>;
        __base_diagonals_sp_storage_t __base_diagonals_sp_global_storage{__exec, 0, __wg_count + 1};

        // 1. Calculate split points on each base diagonal
        //    - one work-item processing one base diagonal
        sycl::event __event = __exec.queue().submit([&](sycl::handler& __cgh) {

            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2);
            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::write>(__cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<_DiagonalsKernelName...>(
                sycl::range</*dim=*/1>(__wg_count + 1), [=](sycl::item</*dim=*/1> __item_id) {

                    const std::size_t __linear_id = __item_id.get_linear_id();

                    _split_point_t<_IdType>* __base_diagonals_sp_global_ptr = __base_diagonals_sp_storage_t::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    // Save top-left split point for first/last base diagonals of merge matrix
                    //  - in GLOBAL coordinates
                    _split_point_t<_IdType> __sp(__linear_id == 0 ? __zero_split_point<std::size_t> : _split_point_t<std::size_t>{__n1, __n2});
                    if (0 < __linear_id && __linear_id < __wg_count)
                        __sp = __find_start_point(__rng1, __rng2, (_IdType)(__linear_id * __wi_in_one_wg * __chunk), __n1, __n2, __comp);

                    __base_diagonals_sp_global_ptr[__linear_id] = __sp;
                });
        });

        // 2. Merge data using split points on each base diagonal
        //    - one work-item processing one diagonal
        //    - work-items grouped to process diagonals between two base diagonals (include left base diagonal and exclude right base diagonal)
        __event = __exec.queue().submit([&](sycl::handler& __cgh) {

            __cgh.depends_on(__event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read>(__cgh);
            auto __base_diagonals_sp_global_ptr = __base_diagonals_sp_storage_t::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

            const std::size_t __slm_cached_data_size = __wi_in_one_wg * __chunk;
            __dpl_sycl::__local_accessor<_RangeValueType> __loc_acc(2 * __slm_cached_data_size, __cgh);

            // Run nd_range parallel_for to process all the data
            // - each work-group caching source data in SLM and processing diagonals between two base diagonals;
            // - each work-item processing one diagonal.
            __cgh.parallel_for<_MergeKernelName...>(
                sycl::nd_range</*dim=*/1>(__wg_count * __wi_in_one_wg, __wi_in_one_wg),
                [=](sycl::nd_item</*dim=*/1> __nd_item)
                {
                    const std::size_t __global_linear_id = __nd_item.get_global_linear_id();    // Merge matrix diagonal's GLOBAL index
                    const std::size_t __local_id = __nd_item.get_local_id(0);                   // Merge sub-matrix LOCAL diagonal's index
                    const std::size_t __group_linear_id = __nd_item.get_group_linear_id();      // Merge matrix base diagonal's GLOBAL index

                    // Split points on left anr right base diagonals
                    //  - in GLOBAL coordinates
                    const auto& __sp_base_left_global  = __base_diagonals_sp_global_ptr[__group_linear_id];
                    const auto& __sp_base_right_global = __base_diagonals_sp_global_ptr[__group_linear_id + 1]; 

                    assert(__sp_base_right_global.first >= __sp_base_left_global.first);
                    assert(__sp_base_right_global.second >= __sp_base_left_global.second);

                    const _IdType __rng1_wg_data_size = __sp_base_right_global.first - __sp_base_left_global.first;
                    const _IdType __rng2_wg_data_size = __sp_base_right_global.second - __sp_base_left_global.second;

                    _RangeValueType* __rng1_cache_slm = std::addressof(__loc_acc[0]);
                    _RangeValueType* __rng2_cache_slm = std::addressof(__loc_acc[0]) + __rng1_wg_data_size;

                    const _IdType __chunk_of_data_reading = std::max(__data_items_in_slm_bank, oneapi::dpl::__internal::__dpl_ceiling_div(__rng1_wg_data_size + __rng2_wg_data_size, __wi_in_one_wg));

                    const _IdType __how_many_wi_reads_rng1 = oneapi::dpl::__internal::__dpl_ceiling_div(__rng1_wg_data_size, __chunk_of_data_reading);
                    const _IdType __how_many_wi_reads_rng2 = oneapi::dpl::__internal::__dpl_ceiling_div(__rng2_wg_data_size, __chunk_of_data_reading);
                    
                    // Calculate the amount of WI for read data from rng1
                    if (__local_id < __how_many_wi_reads_rng1)
                    {
                        const _IdType __idx_begin = __local_id * __chunk_of_data_reading;

                        // Cooperative data load from __rng1 to __rng1_cache_slm
                        if (__idx_begin < __rng1_wg_data_size)
                        {
                            const _IdType __idx_end = std::min(__idx_begin + __chunk_of_data_reading, __rng1_wg_data_size);
                    
                            _ONEDPL_PRAGMA_UNROLL
                            for (_IdType __idx = __idx_begin; __idx < __idx_end; ++__idx)
#if !DUMP_DATA_LOADING
                                __rng1_cache_slm[__idx] = __rng1[__sp_base_left_global.first + __idx];
#else
                                __load_item_into_slm(__rng1, __sp_base_left_global.first + __idx, __rng1_cache_slm, __idx, 1, __b_check, __group_linear_id, __local_id);
#endif
                        }
                    }

                    const std::size_t __first_wi_local_id_for_read_rng2 = __wi_in_one_wg - __how_many_wi_reads_rng2 - 1;
                    if (__local_id >= __first_wi_local_id_for_read_rng2)
                    {
                        const _IdType __idx_begin = (__local_id - __first_wi_local_id_for_read_rng2) * __chunk_of_data_reading;

                        // Cooperative data load from __rng2 to __rng2_cache_slm
                        if (__idx_begin < __rng2_wg_data_size)
                        {
                            const _IdType __idx_end = std::min(__idx_begin + __chunk_of_data_reading, __rng2_wg_data_size);
                    
                            _ONEDPL_PRAGMA_UNROLL
                            for (_IdType __idx = __idx_begin; __idx < __idx_end; ++__idx)
#if !DUMP_DATA_LOADING
                                __rng2_cache_slm[__idx] = __rng2[__sp_base_left_global.second + __idx];
#else
                                __load_item_into_slm(__rng2, __sp_base_left_global.second + __idx, __rng2_cache_slm, __idx, 2, __b_check, __group_linear_id, __local_id);
#endif
                        }
                    }

                    // Wait until all the data is loaded
                    __dpl_sycl::__group_barrier(__nd_item);

#if DUMP_DATA_LOADING
                    if (__local_id == 0)
                    {
                        for (auto i = __sp_base_left_global.first; i < __sp_base_right_global.first; ++i)
                        {
                            auto _idx_slm = i - __sp_base_left_global.first;
                            if (__rng1_cache_slm[_idx_slm] != __rng1[i])
                            {
                                auto __group_linear_id_tmp = __group_linear_id;
                                __group_linear_id_tmp = __group_linear_id_tmp;
                                assert(false);
                            }
                        }

                        for (auto i = __sp_base_left_global.second; i < __sp_base_right_global.second; ++i)
                        {
                            auto _idx_slm = i - __sp_base_left_global.second;
                            if (__rng2_cache_slm[_idx_slm] != __rng2[i])
                            {
                                auto __group_linear_id_tmp = __group_linear_id;
                                __group_linear_id_tmp = __group_linear_id_tmp;
                                assert(false);
                            }
                        }
                    }
#endif

                    // Current diagonal inside of the merge matrix?
                    if (__global_linear_id * __chunk < __n)
                    {
                        // Find split point in LOCAL coordinates
                        //  - bottom-right split point describes the size of current area between two base diagonals.
                        const _split_point_t<_IdType> __sp_local = __find_start_point(
                            __rng1_cache_slm, __rng2_cache_slm,                         // SLM cached copy of merging data
                            (_IdType)(__local_id * __chunk),                            // __i_elem in LOCAL coordinates because __rng1_cache_slm and __rng1_cache_slm is work-group SLM cached copy of source data
                            __rng1_wg_data_size, __rng2_wg_data_size,                   // size of rng1 and rng2
                            __comp);

                        // Merge data for the current diagonal
                        //  - we should have here __sp_global in GLOBAL coordinates
                        __serial_merge(__rng1_cache_slm, __rng2_cache_slm,              // SLM cached copy of merging data
                                       __rng3,                                          // Destination range
                                       __sp_local.first,                                // __start1 in LOCAL coordinates because __rng1_cache_slm is work-group SLM cached copy of source data
                                       __sp_local.second,                               // __start2 in LOCAL coordinates because __rng1_cache_slm is work-group SLM cached copy of source data
                                       (_IdType)(__global_linear_id * __chunk),         // __start3 in GLOBAL coordinates because __rng3 is not cached at all
                                       __chunk,
                                       __rng1_wg_data_size, __rng2_wg_data_size,        // size of rng1 and rng2
                                       __comp);
                    }
                });
        });
        return __future(__event);
    }
};

template <typename... _Name>
class __merge_kernel_name;

template <typename... _Name>
class __diagonals_kernel_name;

template <typename... _Name>
class __merge_kernel_name_large;

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    constexpr std::size_t __starting_size_limit_for_large_submitter = 1 * 1'048'576; // 1 Mb

    using _Range1ValueType = typename std::iterator_traits<decltype(__rng1.begin())>::value_type;
    using _Range2ValueType = typename std::iterator_traits<decltype(__rng2.begin())>::value_type;

    constexpr bool __same_merge_types = std::is_same_v<_Range1ValueType, _Range2ValueType>;

    const std::size_t __n = __rng1.size() + __rng2.size();
    if (__n < __starting_size_limit_for_large_submitter || !__same_merge_types)
    {
        static_assert(__starting_size_limit_for_large_submitter < std::numeric_limits<std::uint32_t>::max());
    
        using _WiIndex = std::uint32_t;
        using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name<_CustomName, _WiIndex>>;
        return __parallel_merge_submitter<_WiIndex, _MergeKernelName>()(
            std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            std::forward<_Range3>(__rng3), __comp);
    }
    else
    {
        if (__n <= std::numeric_limits<std::uint32_t>::max())
        {
            using _WiIndex = std::uint32_t;
            using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __diagonals_kernel_name<_CustomName, _WiIndex>>;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_WiIndex, _CustomName, _DiagonalsKernelName, _MergeKernelName>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __diagonals_kernel_name<_CustomName, _WiIndex>>;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_WiIndex, _CustomName, _DiagonalsKernelName, _MergeKernelName>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
