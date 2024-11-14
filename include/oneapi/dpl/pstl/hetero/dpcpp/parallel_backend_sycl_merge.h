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
    const _Index __rng1_from = 0;
    const _Index __rng1_to = __n1;
    const _Index __rng2_from = 0;
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

    return std::make_pair(0, 0);
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
        for (std::uint8_t __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng2[__start2 + __i];
    }
    else if (__start2 >= __n2)
    {
        //copying a residual of the first seq
        const _Index __n = std::min<_Index>(__n1 - __start1, __chunk);
        for (std::uint8_t __i = 0; __i < __n; ++__i)
            __rng3[__start3 + __i] = __rng1[__start1 + __i];
    }
    else
    {
        for (std::uint8_t __i = 0; __i < __chunk && __start1 < __n1 && __start2 < __n2; ++__i)
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
    // Create local accessors for data cache in SLM:
    //  - one accessor for the first and for the second ranges if _Range1 and _Range2 has the SAME value types;
    //  - two accessors for the first and for the second ranges if _Range1 and _Range2 has DIFFERENT value types.
    struct __merge_slm_helper
    {
        template <typename _Range1, typename _Range2>
        static std::size_t
        get_data_size(_Range1&& __rng1, _Range2&& __rng2)
        {
            using _Range1ValueType = typename std::iterator_traits<decltype(__rng1.begin())>::value_type;
            using _Range2ValueType = typename std::iterator_traits<decltype(__rng2.begin())>::value_type;

            return sizeof(_Range1ValueType) + sizeof(_Range2ValueType);
        }

        template <typename _Range1, typename _Range2>
        static constexpr auto
        create_local_accessors(sycl::handler& __cgh, _Range1&& __rng1, _Range2&& __rng2,
                               std::size_t __slm_cached_data_size)
        {
            using _Range1ValueType = typename std::iterator_traits<decltype(__rng1.begin())>::value_type;
            using _Range2ValueType = typename std::iterator_traits<decltype(__rng2.begin())>::value_type;

            if constexpr (std::is_same_v<_Range1ValueType, _Range2ValueType>)
                return std::tuple<__dpl_sycl::__local_accessor<_Range1ValueType>>(
                    __dpl_sycl::__local_accessor<_Range1ValueType>(2 * __slm_cached_data_size, __cgh));
            else
                return std::tuple<__dpl_sycl::__local_accessor<_Range1ValueType>,
                                  __dpl_sycl::__local_accessor<_Range2ValueType>>(
                    __dpl_sycl::__local_accessor<_Range1ValueType>(__slm_cached_data_size, __cgh),
                    __dpl_sycl::__local_accessor<_Range2ValueType>(__slm_cached_data_size, __cgh));
        }

        template <std::size_t AccessorIdx, typename AccessorsTuple>
        static auto
        get_local_accessor(AccessorsTuple& __acc_tuple, std::size_t __offset = 0)
        {
            static_assert(std::tuple_size_v<AccessorsTuple> == 1 || std::tuple_size_v<AccessorsTuple> == 2);

            if constexpr (std::tuple_size_v<AccessorsTuple> == 1)
                return std::pair(std::get<0>(__acc_tuple), __offset);

            else
                return std::pair(std::get<AccessorIdx>(__acc_tuple), 0);
        }
    };

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
        const _IdType __chunk = __exec.queue().get_device().is_cpu() ? 128 : 4;
        assert(__chunk > 0);

        // Pessimistically only use half of the memory to take into account memory used by compiled kernel
        const std::size_t __max_slm_size_adj =
            oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, __merge_slm_helper::get_data_size(__rng1, __rng2));

        // The amount of data must be a multiple of the chunk size.
        const std::size_t __max_source_data_items_fit_into_slm = __max_slm_size_adj - __max_slm_size_adj % __chunk;
        assert(__max_source_data_items_fit_into_slm > 0);
        assert(__max_source_data_items_fit_into_slm % __chunk == 0);

        // The amount of items in the each work-group is the amount of diagonals processing between two work-groups + 1 (for the left base diagonal in work-group)
        const _IdType __items_in_wg_count = __max_source_data_items_fit_into_slm / __chunk;
        assert(__items_in_wg_count > 0);

        // The amount of the base diagonals is the amount of the work-groups
        //  - also it's the distance between two base diagonals is equal to the amount of work-items in each work-group
        const _IdType __wg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __max_source_data_items_fit_into_slm);

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

                    const std::size_t __global_idx = __item_id.get_linear_id();

                    _split_point_t<_IdType>* __base_diagonals_sp_global_ptr = __base_diagonals_sp_storage_t::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    // Save top-left split point for first/last base diagonals of merge matrix
                    //  - in GLOBAL coordinates
                    _split_point_t<_IdType> __sp(__global_idx == 0 ? __zero_split_point<_IdType> : _split_point_t<_IdType>{__n1, __n2});

                    if (0 < __global_idx && __global_idx < __wg_count)
                    {
                        const _IdType __i_elem = __global_idx * __items_in_wg_count * __chunk;

                        // Save bottom-right split point for current base diagonal of merge matrix
                        //  - in GLOBAL coordinates
                        __sp = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                    }

                    __base_diagonals_sp_global_ptr[__global_idx] = __sp;
                });
        });

        // 2. Merge data using split points on each base diagonal
        //    - one work-item processing one diagonal
        //    - work-items grouped to process diagonals between two base diagonals (include left base diagonal and exclude right base diagonal)
        __event = __exec.queue().submit([&](sycl::handler& __cgh) {

            __cgh.depends_on(__event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read>(__cgh);

            const std::size_t __slm_cached_data_size = __items_in_wg_count * __chunk;
            auto local_accessors = __merge_slm_helper::create_local_accessors(__cgh, __rng1, __rng2, __slm_cached_data_size);

            // Run nd_range parallel_for to process all the data
            __cgh.parallel_for<_MergeKernelName...>(
                sycl::nd_range</*dim=*/1>(__wg_count * __items_in_wg_count, __items_in_wg_count),
                [=](sycl::nd_item</*dim=*/1> __nd_item)
                {
                    // Merge matrix diagonal's GLOBAL index
                    const std::size_t __global_idx = __nd_item.get_global_linear_id();

                    // Merge sub-matrix LOCAL diagonal's index
                    const std::size_t __local_idx = __nd_item.get_local_id(0);

                    // Merge matrix base diagonal's GLOBAL index
                    const std::size_t __wg_id = __nd_item.get_group_linear_id();

                    auto __base_diagonals_sp_global_ptr = __base_diagonals_sp_storage_t::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    // Split points on left anr right base diagonals
                    //  - in GLOBAL coordinates
                    assert(__wg_id + 1 < __wg_count + 1);
                    const _split_point_t<_IdType>& __sp_base_left_global  = __base_diagonals_sp_global_ptr[__wg_id];
                    const _split_point_t<_IdType>& __sp_base_right_global = __base_diagonals_sp_global_ptr[__wg_id + 1]; 

                    auto [__local_accessor_rng1, offset_to_slm1] = __merge_slm_helper::template get_local_accessor<0>(local_accessors);
                    auto [__local_accessor_rng2, offset_to_slm2] = __merge_slm_helper::template get_local_accessor<1>(local_accessors, (std::size_t)(__sp_base_right_global.first -__sp_base_left_global.first));
                    auto __rngs_data_in_slm1 = std::addressof(__local_accessor_rng1[0]) + offset_to_slm1;
                    auto __rngs_data_in_slm2 = std::addressof(__local_accessor_rng2[0]) + offset_to_slm2;

                    // Full amount of work-items may be great then the amount of diagonals in the merge matrix
                    // so we should skip the redundant work-items
                    const bool __out_of_data = __global_idx * __chunk >= __n;
                    if (!__out_of_data)
                    {
                        // Load the current part of merging data placed between two base diagonals into SLM
                        // TODO implement cooperative data load by multiple work-items
                        assert(__items_in_wg_count > 1);
                        if (__local_idx == 0)
                        {
                            _IdType __slm_idx = 0;
                            for (_IdType __idx = __sp_base_left_global.first; __idx < __sp_base_right_global.first; ++__idx, ++__slm_idx)
                            {
                                assert(__slm_idx < __slm_cached_data_size);
                                assert(__idx < __n1);
                                __rngs_data_in_slm1[__slm_idx] = __rng1[__idx];
                            }
                        }

                        if (__local_idx == 1 && __items_in_wg_count > 1 || __local_idx == 0)
                        {
                            _IdType __slm_idx = 0;
                            for (_IdType __idx = __sp_base_left_global.second; __idx < __sp_base_right_global.second; ++__idx, ++__slm_idx)
                            {
                                assert(__slm_idx < __slm_cached_data_size);
                                assert(__idx < __n2);
                                __rngs_data_in_slm2[__slm_idx] = __rng2[__idx];
                            }
                        }
                    }

                    // Wait until all the data is loaded
                    //  - we shouldn't setup this barrier under any conditions!!!
                    __dpl_sycl::__group_barrier(__nd_item);

                    if (!__out_of_data)
                    {
                        // We are between two base diagonals and need to find the start points in the merge matrix area,
                        // limited by split points of the left and right base diagonals.

                        // Find split point in LOCAL coordinates
                        //  - top-left split point is (0, 0);
                        //  - bottom-right split point describes the size of current area between two base diagonals.
                        assert(__sp_base_right_global.first >= __sp_base_left_global.first);
                        assert(__sp_base_right_global.second >= __sp_base_left_global.second);
                        const _split_point_t<_IdType> __sp_local = __find_start_point(
                            __rngs_data_in_slm1, __rngs_data_in_slm2,                                   // SLM cached copy of merging data
                            (_IdType)(__local_idx * __chunk),                                           // __i_elem in LOCAL coordinates because __rngs_data_in_slm1 and __rngs_data_in_slm2 is work-group SLM cached copy of source data
                            (_IdType)(__sp_base_right_global.first - __sp_base_left_global.first),      // size of rng1
                            (_IdType)(__sp_base_right_global.second - __sp_base_left_global.second),    // size of rng2
                            __comp);

                        // Merge data for the current diagonal
                        //  - we should have here __sp_global in GLOBAL coordinates
                        __serial_merge(__rngs_data_in_slm1, __rngs_data_in_slm2,                        // SLM cached copy of merging data
                                       __rng3,                                                          // Destination range
                                       __sp_local.first,                                                // __start1 in LOCAL coordinates because __rngs_data_in_slm1 is work-group SLM cached copy of source data
                                       __sp_local.second,                                               // __start2 in LOCAL coordinates because __rngs_data_in_slm2 is work-group SLM cached copy of source data
                                       (_IdType)(__global_idx * __chunk),                               // __start3 in GLOBAL coordinates because __rng3 is not cached at all
                                       __chunk,
                                        __sp_base_right_global.first - __sp_base_left_global.first,     // size of __rngs_data_in_slm1
                                        __sp_base_right_global.second - __sp_base_left_global.second,   // size of __rngs_data_in_slm2
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

    const auto __n = __rng1.size() + __rng2.size();
    if (__n < 4 * 1'048'576)
    {
        if (__n <= std::numeric_limits<std::uint32_t>::max())
        {
            using _WiIndex = std::uint32_t;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter<_WiIndex, _MergeKernelName>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter<_WiIndex, _MergeKernelName>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
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
