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

#define LOG_SET_SPLIT_POINTS 1
#define LOG_ND_RANGE_PARAMS 1
#define WAIT_IN_IMPL 1
#define FILL_ADDITIONAL_DIAGONALS 1
#if FILL_ADDITIONAL_DIAGONALS
#   define USE_ADDITIONAL_DIAGONALS 1
#endif

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

template <typename _Size1, typename _Value, typename _Compare>
_Size1
__pstl_lower_bound(_Size1 __first, _Size1 __last, const _Value& __value, _Compare __comp)
{
    auto __n = __last - __first;
    auto __cur = __n;
    _Size1 __it;
    while (__n > 0)
    {
        __it = __first;
        __cur = __n / 2;
        __it += __cur;
        if (__comp(__it, __value))
        {
            __n -= __cur + 1;
            __first = ++__it;
        }
        else
        {
            __n = __cur;
        }
    }
    return __first;
}

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
__find_start_point_in(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __i_elem,
                      const _Index __rng1_from, const _Index __rng1_to,
                      const _Index __rng2_from, const _Index __rng2_to,
                      _Compare __comp)
{
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

        constexpr int kValue = 1;
        const auto __res = __pstl_lower_bound(idx1_from, idx1_to, kValue, [&](_Index __idx, const auto& __value) {
                const auto __rng1_idx = __idx;
                const auto __rng2_idx = __index_sum - __idx;

                assert(__rng1_from <= __rng1_idx && __rng1_idx < __rng1_to);
                assert(__rng2_from <= __rng2_idx && __rng2_idx < __rng2_to);
                assert(__rng1_idx + __rng2_idx == __index_sum);

                const auto __zero_or_one = __comp(__rng2[__rng2_idx], __rng1[__rng1_idx]);
                return __zero_or_one < kValue;
            });

        const _split_point_t<_Index> __result = std::make_pair(__res, __index_sum - __res + 1);
        assert(__result.first + __result.second == __i_elem);

        assert(__rng1_from <= __result.first && __result.first <= __rng1_to);
        assert(__rng2_from <= __result.second && __result.second <= __rng2_to);

        return __result;
    }

    return __zero_split_point<_Index>;
}

template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
inline _split_point_t<_Index>
__find_start_point_in(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __i_elem,
                      const _split_point_t<_Index>& __sp_from,
                      const _split_point_t<_Index>& __sp_to, 
                      _Compare __comp)
{
    return __find_start_point_in(__rng1, __rng2, __i_elem,
                                 __sp_from.first, __sp_to.first,
                                 __sp_from.second, __sp_to.second,
                                 __comp);
}

template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
inline _split_point_t<_Index>
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __i_elem, const _Index __n1,
                   const _Index __n2, _Compare __comp)
{
    return __find_start_point_in(__rng1, __rng2, __i_elem, _split_point_t<_Index>{0, 0}, _split_point_t<_Index>{__n1, __n2}, __comp);
}

// Do serial merge of the data from rng1 (starting from start1) and rng2 (starting from start2) and writing
// to rng3 (starting from start3) in 'chunk' steps, but do not exceed the total size of the sequences (n1 and n2)
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index, typename _Compare>
void
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3, _Index __start1, _Index __start2,
               const _Index __start3, const _Index __chunk, const _Index __n1, const _Index __n2, _Compare __comp)
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
        const _IdType __chunk = __exec.queue().get_device().is_cpu() ? 128 : 4;

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
  protected:

    struct nd_range_params
    {
        std::size_t wg_count = 0;                   // Amount of work-groups
        std::size_t wi_in_one_wg = 0;               // Amount of work-items in each work-group
        std::size_t chunk = 0;                      // Diagonal's chunk
        std::size_t diags_per_wi = 1;               // Amount of diagonals for processing in each work-item
        std::size_t additional_diags_inside_wg = 0; // Amount of additional base diagonals inside work-group

        inline std::size_t
        get_diags_in_one_wg() const
        {
            return wi_in_one_wg * diags_per_wi;
        }

        inline std::size_t
        get_wg_data_size() const
        {
            return get_diags_in_one_wg() * chunk;
        }

        inline bool
        have_additional_base_diagonals_inside_one_wg() const
        {
            return additional_diags_inside_wg > 0;
        }

        // Get amount of base diagonals in one wg
        inline std::size_t
        get_base_diagonals_in_one_wg_count() const
        {
            // +1 - to take into account the left bound of work-group
            return additional_diags_inside_wg + 1;
        }

        // Get step (chunk) of base diagonals in one wg
        inline std::size_t
        get_step_of_base_diagonals_in_one_wg_count() const
        {
            return oneapi::dpl::__internal::__dpl_ceiling_div(get_diags_in_one_wg(),
                                                              get_base_diagonals_in_one_wg_count());
        }

        // Get amount of base diagonals
        inline std::size_t
        get_base_diagonals_count_in_all_groups() const
        {
            // +1 - to take into account the right bound for last work-group
            return wg_count * get_base_diagonals_in_one_wg_count() + 1;
        }
    };

    // Calculate nd-range params
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
    nd_range_params
    eval_nd_range_params(_ExecutionPolicy&& __exec, const _Range1& __rng1, const _Range2& __rng2) const
    {
        using namespace oneapi::dpl::__internal;

        using _Range1ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        using _Range2ValueType = oneapi::dpl::__internal::__value_t<_Range2>;
        static_assert(std::is_same_v<_Range1ValueType, _Range2ValueType>,
                      "In this implementation we can merge only data of the same type");

        using _RangeValueType = _Range1ValueType;

        const std::size_t __n = __rng1.size() + __rng2.size();

        // Define SLM bank size
        constexpr std::size_t __slm_bank_size = 16;     // TODO is it correct value? How to get it from hardware?

        const std::size_t __oversubscription = 2;

        // Calculate how many data items we can read into one SLM bank
        constexpr std::size_t __data_items_in_slm_bank = __dpl_ceiling_div(__slm_bank_size, sizeof(_RangeValueType));

        // Empirical number of values to process per work-item
        std::size_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : __data_items_in_slm_bank;
        assert(__chunk > 0);

        // Get the size of local memory arena in bytes.
        const std::size_t __slm_mem_size = __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>() / __oversubscription;

        // Calculate how many items count we may place into SLM memory
        const auto __slm_cached_items_count = __slm_mem_size / sizeof(_RangeValueType);

        // Get the maximum work-group size for the current device
        const std::size_t __max_wg_size = __exec.queue().get_device().template get_info<sycl::info::device::max_work_group_size>();

        // The amount of items in the each work-group is the amount of diagonals processing between two work-groups + 1 (for the left base diagonal in work-group)
        std::size_t __diags_per_wi = 1;
        std::size_t __wi_in_one_wg = __slm_cached_items_count / (__diags_per_wi * __chunk);
        if (__wi_in_one_wg > __max_wg_size)
        {
            __diags_per_wi = __dpl_ceiling_div(__wi_in_one_wg, __max_wg_size);
            __wi_in_one_wg = __slm_cached_items_count / (__diags_per_wi * __chunk);
            assert(__wi_in_one_wg <= __max_wg_size);
        }

        // Check the amount of work-items
        assert(0 < __wi_in_one_wg && __wi_in_one_wg <= __max_wg_size);

        // Check that we have enough SLM to cache all the data
        assert(__wi_in_one_wg * __diags_per_wi * __chunk * sizeof(_RangeValueType) <= __slm_mem_size);

        // The amount of the base diagonals is the amount of the work-groups
        //  - also it's the distance between two base diagonals is equal to the amount of work-items in each work-group
        const std::size_t __wg_count = __dpl_ceiling_div(__n, __wi_in_one_wg * __diags_per_wi * __chunk);

        // Check that we have enough nd-range to process all the data
        assert(__wg_count * __wi_in_one_wg * __diags_per_wi * __chunk >= __n);

        // Calculate the amount of additional base diagonal inside work-group
        std::size_t __additional_diags_inside_wg = 0;
#if FILL_ADDITIONAL_DIAGONALS
        if (__wi_in_one_wg * __diags_per_wi >= 100)
            __additional_diags_inside_wg = __dpl_ceiling_div(__wi_in_one_wg * __diags_per_wi, 10);
#endif

        return nd_range_params{__wg_count, __wi_in_one_wg, __chunk, __diags_per_wi, __additional_diags_inside_wg};
    }

    // Get indexes of left and right base diagonals indexes for specified group
    static std::tuple<std::size_t, std::size_t>
    __get_group_base_diagonals(const nd_range_params& __nd_range_params, const std::size_t __group_linear_id)
    {
        const auto __base_diagonals_in_one_wg_count = __nd_range_params.get_base_diagonals_in_one_wg_count();

        const auto __base_diagonals_in_all_prev_groups = __base_diagonals_in_one_wg_count * __group_linear_id;

        std::tuple<std::size_t, std::size_t> __result{__base_diagonals_in_all_prev_groups,
                                                      __base_diagonals_in_all_prev_groups + __base_diagonals_in_one_wg_count};

        return __result;
    }

#if LOG_SET_SPLIT_POINTS
    // TODO remove debug code
    template <typename Data>
    static void
    __set_base_sp(Data* __base_diagonals_sp_global_ptr, std::size_t __diagonal_idx, const _split_point_t<_IdType>& __sp,
                  const _IdType __n1, const _IdType __n2)
    {
        assert(__sp.first <= __n1);
        assert(__sp.second <= __n2);
        // __base_diagonals_sp_global_ptr[{__diagonal_idx}] = {__sp};
        __base_diagonals_sp_global_ptr[__diagonal_idx] = __sp;
    }
#endif

    // Calculation of split points on each base diagonal
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Compare, typename _Storage>
    sycl::event
    eval_split_points_for_groups(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Compare __comp,
                                 const nd_range_params& __nd_range_params,
                                 _Storage& __base_diagonals_sp_global_storage) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        const _IdType __n = __n1 + __n2;

        sycl::event __event = __exec.queue().submit([&](sycl::handler& __cgh) {

            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2);

            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read_write>(__cgh);
            auto __base_diagonals_sp_global_ptr = _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

            const std::size_t __step_of_base_diagonals_in_one_wg_count = __nd_range_params.get_step_of_base_diagonals_in_one_wg_count();

            const std::size_t __wg_count = __nd_range_params.wg_count + 1;                      // +1 to take into account the last group in which we fill only left base diagonal split point
            const std::size_t __wi_count = __nd_range_params.additional_diags_inside_wg + 1;

            // Each work-group processing left base diagonal and additional internal diagonals between two base diagonals.
            // Each work-item processing one base or additional diagonal inside group.
            __cgh.parallel_for<_DiagonalsKernelName...>(
                sycl::nd_range</*dim=*/1>(__wg_count * __wi_count, __wi_count),
                [=](sycl::nd_item</*dim=*/1> __nd_item)
                {
                    //const std::size_t __global_linear_id = __nd_item.get_global_linear_id();    // Merge matrix diagonal's GLOBAL index
                    const std::size_t __local_id = __nd_item.get_local_id(0);                   // Merge sub-matrix LOCAL diagonal's index
                    const std::size_t __group_linear_id = __nd_item.get_group_linear_id();      // Merge matrix base diagonal's GLOBAL index

                    // Calculate indexes of left and right base diagonals for the current group
                    const auto [__diagonal_idx_left_global, __diagonal_idx_right_global] = __get_group_base_diagonals(__nd_range_params, __group_linear_id);
                    //assert(__local_id < __diagonal_idx_right_global - __diagonal_idx_left_global);

                    const bool __is_first_wi_of_first_group = __group_linear_id == 0 && __local_id == 0;
                    const bool __is_last_group = __group_linear_id == __wg_count - 1;

                    // In the last group we calculate and fill the split point only for the left base diagonal
                    if (!__is_last_group || __local_id == 0) 
                    {
                        _split_point_t<_IdType> __sp = __is_first_wi_of_first_group ? __zero_split_point<_IdType> : _split_point_t<_IdType>{ __n1, __n2 };
                        if (!__is_last_group)
                        {
                            if (!__is_first_wi_of_first_group)
                            {
                                // We calculate some additional diagonal inside of the group
                                const _IdType __i_elem = __group_linear_id * __nd_range_params.get_wg_data_size() + __local_id * __step_of_base_diagonals_in_one_wg_count * __nd_range_params.chunk;

                                // Save top-left split point for first/last base diagonals of merge matrix
                                //  - in GLOBAL coordinates
                                if (__i_elem < __n)
                                    __sp = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                            }
                        }

                        const std::size_t __diagonal_idx_global = __diagonal_idx_left_global + __local_id;

    #if LOG_SET_SPLIT_POINTS
                        // TODO remove debug code
                        __set_base_sp(__base_diagonals_sp_global_ptr, __diagonal_idx_global, __sp, __n1, __n2);
    #else
                        __base_diagonals_sp_global_ptr[__diagonal_idx_global] = __sp;
    #endif
                    }
                });
        });
#if WAIT_IN_IMPL
        // TODO remove debug code
        __event.wait();
#endif

        return __event;
    }

    // Read data into SLM cache
    template <typename _Range, typename _RangeValueType>
    static void
    __read_data_into_slm(const _Range& __rng, _RangeValueType* __rng_cache_slm,
                         const std::size_t __local_id, const std::size_t __chunk_of_data_reading,
                         const std::size_t __rng_wg_data_size,
                         const std::size_t __rng_offset)
    {
        const std::size_t __idx_begin = __local_id * __chunk_of_data_reading;

        // Cooperative data load from __rng1 to __rng1_cache_slm
        if (__idx_begin < __rng_wg_data_size)
        {
            const std::size_t __idx_end = std::min(__idx_begin + __chunk_of_data_reading, __rng_wg_data_size);

            _ONEDPL_PRAGMA_UNROLL
            for (_IdType __idx = __idx_begin; __idx < __idx_end; ++__idx)
                __rng_cache_slm[__idx] = __rng[__rng_offset + __idx];
        }
    }

    static _split_point_t<_IdType>
    __convert_to_local(const _split_point_t<_IdType>& __sp_base_global, const _split_point_t<_IdType>& __sp_global,
                       const std::size_t __global_linear_id,
                       const std::size_t __local_id,
                       const std::size_t __group_linear_id)
    {
        assert(__sp_global.first >= __sp_base_global.first);
        assert(__sp_global.second >= __sp_base_global.second);

        _split_point_t<_IdType> __sp_local{ __sp_global.first - __sp_base_global.first, __sp_global.second - __sp_base_global.second };

        // condition: __sp_base_global.first + __sp_base_global.second > 0
        // action:    __global_linear_id = {__global_linear_id}, __group_linear_id = {__group_linear_id}, __local_id = {__local_id} : __convert_to_local({__sp_base_global}, {__sp_global}) -> {__sp_local}
        return __sp_local;
    }

    // Process merge in nd-range space
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
              typename _Storage>
    sycl::event
    run_parallel_merge(sycl::event __event,
                       _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp,
                       const nd_range_params& __nd_range_params,
                       const _Storage& __base_diagonals_sp_global_storage) const
    {
        using _Range1ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        using _Range2ValueType = oneapi::dpl::__internal::__value_t<_Range2>;
        static_assert(std::is_same_v<_Range1ValueType, _Range2ValueType>,
                      "In this implementation we can merge only data of the same type");

        using _RangeValueType = _Range1ValueType;

        /*const*/ _IdType __n = __rng1.size() + __rng2.size();
        __n = __n;

        return __exec.queue().submit([&](sycl::handler& __cgh) {

            __cgh.depends_on(__event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);

            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read>(__cgh);
            auto __base_diagonals_sp_global_ptr = _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

            const std::size_t __slm_cached_data_size = __nd_range_params.get_wg_data_size();
            __dpl_sycl::__local_accessor<_RangeValueType> __loc_acc(__slm_cached_data_size, __cgh);

            // Run nd_range parallel_for to process all the data
            // - each work-group caching source data in SLM and processing diagonals between two base diagonals;
            // - each work-item processing one one more diagonal.
            __cgh.parallel_for<_MergeKernelName...>(
                sycl::nd_range</*dim=*/1>(__nd_range_params.wg_count * __nd_range_params.wi_in_one_wg, __nd_range_params.wi_in_one_wg),
                [=](sycl::nd_item</*dim=*/1> __nd_item)
                {
                    const std::size_t __global_linear_id = __nd_item.get_global_linear_id();    // Merge matrix diagonal's GLOBAL index
                    const std::size_t __local_id = __nd_item.get_local_id(0);                   // Merge sub-matrix LOCAL diagonal's index
                    const std::size_t __group_linear_id = __nd_item.get_group_linear_id();      // Merge matrix base diagonal's GLOBAL index

                    // Calculate indexes of left and right base diagonals for the current group
                    const auto [__diagonal_idx_left_global, __diagonal_idx_right_global] = __get_group_base_diagonals(__nd_range_params, __group_linear_id);

                    // Split points on left anr right base diagonals
                    //  - in GLOBAL coordinates
                    const auto& __sp_base_left_global  = __base_diagonals_sp_global_ptr[__diagonal_idx_left_global];
                    const auto& __sp_base_right_global = __base_diagonals_sp_global_ptr[__diagonal_idx_right_global];

                    assert(__sp_base_right_global.first >= __sp_base_left_global.first);
                    assert(__sp_base_right_global.second >= __sp_base_left_global.second);

                    const std::size_t __rng1_wg_data_size = __sp_base_right_global.first - __sp_base_left_global.first;
                    const std::size_t __rng2_wg_data_size = __sp_base_right_global.second - __sp_base_left_global.second;
                    const std::size_t __rng_wg_data_size = __rng1_wg_data_size + __rng2_wg_data_size;

                    assert(__rng_wg_data_size <= __slm_cached_data_size);

                    _RangeValueType* __rng1_cache_slm = std::addressof(__loc_acc[0]);
                    _RangeValueType* __rng2_cache_slm = std::addressof(__loc_acc[0]) + __rng1_wg_data_size;

                    const std::size_t __chunk_of_data_reading = oneapi::dpl::__internal::__dpl_ceiling_div(__rng_wg_data_size, __nd_range_params.wi_in_one_wg);

                    // Read data from __rng1 into __rng1_cache_slm
                    const std::size_t __how_many_wi_reads_rng1 = oneapi::dpl::__internal::__dpl_ceiling_div(__rng1_wg_data_size, __chunk_of_data_reading);
                    if (__local_id < __how_many_wi_reads_rng1)
                        __read_data_into_slm(__rng1, __rng1_cache_slm, __local_id, __chunk_of_data_reading, __rng1_wg_data_size, __sp_base_left_global.first);

                    // Read data from __rng2 into __rng2_cache_slm
                    const std::size_t __how_many_wi_reads_rng2 = oneapi::dpl::__internal::__dpl_ceiling_div(__rng2_wg_data_size, __chunk_of_data_reading);
                    const std::size_t __first_wi_local_id_for_read_rng2 = __nd_range_params.wi_in_one_wg - __how_many_wi_reads_rng2;
                    if (__local_id >= __first_wi_local_id_for_read_rng2)
                        __read_data_into_slm(__rng2, __rng2_cache_slm, __local_id - __first_wi_local_id_for_read_rng2, __chunk_of_data_reading, __rng2_wg_data_size, __sp_base_left_global.second);

                    // Wait until all the data is loaded
                    __dpl_sycl::__group_barrier(__nd_item);

                    const std::size_t __step_of_base_diagonals_in_one_wg_count = __nd_range_params.get_step_of_base_diagonals_in_one_wg_count();

                    // Process subset of diagonals in the current work-item
                    for (std::size_t __diagonal_iteration_idx = 0; __diagonal_iteration_idx < __nd_range_params.diags_per_wi; ++__diagonal_iteration_idx)
                    {
                        //  Calculate __start3 in GLOBAL coordinates because __rng3 is not cached at all
                        const std::size_t __start3 = (__global_linear_id * __nd_range_params.diags_per_wi + __diagonal_iteration_idx) * __nd_range_params.chunk;

                        // Current diagonal inside of the global merge matrix?
                        if (__start3 < __n)
                        {
                            // Calculate LOCAL index of current diagonal
                            const std::size_t __local_diagonal_idx = __local_id * __nd_range_params.diags_per_wi + __diagonal_iteration_idx;

                            // Calculate __i_elem in LOCAL coordinates because __rng1_cache_slm and __rng1_cache_slm is work-group SLM cached copy of source data
                            const std::size_t __i_elem = __local_diagonal_idx * __nd_range_params.chunk;

                            // Current diagonal inside of the local merge matrix?
                            if (__i_elem < __rng_wg_data_size)
                            {
                                // Limitation for the fist and the second ranges in LOCAL coordinates
                                _split_point_t<std::size_t> __sp_base_left_local = {0, 0};
                                _split_point_t<std::size_t> __sp_base_right_local = {__rng1_wg_data_size, __rng2_wg_data_size};

                                bool __sp_local_found = false;
                                _split_point_t<std::size_t> __sp_local_precalculated{0, 0};

#if USE_ADDITIONAL_DIAGONALS
                                // Avoid split-point search on base or additional diagonal due it's already calculated
                                if (__local_diagonal_idx % __step_of_base_diagonals_in_one_wg_count == 0)
                                {
                                    const std::size_t __nearest_diagonal_idx_local = __local_diagonal_idx / __step_of_base_diagonals_in_one_wg_count;

                                    __sp_local_precalculated = __base_diagonals_sp_global_ptr[__diagonal_idx_left_global + __nearest_diagonal_idx_local];
                                    __sp_local_precalculated = __convert_to_local(__sp_base_left_global, __sp_local_precalculated, __global_linear_id, __local_id, __group_linear_id);
                                    __sp_local_found = true;
                                }

                                // Calculate indexes of nearest left and right base diagonals
                                else
                                {
                                    const std::size_t __nearest_diagonal_idx_left_local  = __local_diagonal_idx / __step_of_base_diagonals_in_one_wg_count;
                                    const std::size_t __nearest_diagonal_idx_right_local = __nearest_diagonal_idx_left_local + 1;

                                    // Get split-points from nearest base diagonals in GLOBAL coordinates
                                    const auto& __sp_base_left_local_tmp  = __base_diagonals_sp_global_ptr[__diagonal_idx_left_global + __nearest_diagonal_idx_left_local];
                                    const auto& __sp_base_right_local_tmp = __base_diagonals_sp_global_ptr[__diagonal_idx_left_global + __nearest_diagonal_idx_right_local];

                                    assert(__sp_base_left_local_tmp.first  <= __sp_base_right_local_tmp.first);
                                    assert(__sp_base_left_local_tmp.second <= __sp_base_right_local_tmp.second);
                                        
                                    assert(__sp_base_left_local_tmp.first  >= __sp_base_left_global.first);
                                    assert(__sp_base_left_local_tmp.second >= __sp_base_left_global.second);
                                        
                                    // Convert split-points from nearest base diagonals into local coordinates
                                    __sp_base_left_local  = __convert_to_local(__sp_base_left_global, __sp_base_left_local_tmp, __global_linear_id, __local_id, __group_linear_id);
                                    __sp_base_right_local = __convert_to_local(__sp_base_left_global, __sp_base_right_local_tmp, __global_linear_id, __local_id, __group_linear_id);
                                        
                                    assert(__sp_base_left_local.first + __sp_base_left_local.second <= __i_elem);
                                    assert(__i_elem < __sp_base_right_local.first + __sp_base_right_local.second);
                                }
#endif // USE_ADDITIONAL_DIAGONALS

                                // Find split point in LOCAL coordinates
                                //  - bottom-right split point describes the size of current area between two base diagonals.
                                const _split_point_t<_IdType> __sp_local =
                                    __sp_local_found
                                    ? __sp_local_precalculated
                                    : __find_start_point_in(__rng1_cache_slm, __rng2_cache_slm,                     // SLM cached copy of merging data
                                                            __i_elem,                                               // __i_elem in LOCAL coordinates because __rng1_cache_slm and __rng1_cache_slm is work-group SLM cached copy of source data
                                                            __sp_base_left_local, __sp_base_right_local,            // limitations for __rng1_cache_slm, __rng2_cache_slm in LOCAL coordinates
                                                            __comp);

                                assert(__sp_base_left_local.first  <= __sp_local.first  && __sp_local.first  <= __sp_base_right_local.first);
                                assert(__sp_base_left_local.second <= __sp_local.second && __sp_local.second <= __sp_base_right_local.second);

                                // Merge data for the current diagonal
                                //  - we should have here __sp_global in GLOBAL coordinates
                                __serial_merge(
                                    __rng1_cache_slm, __rng2_cache_slm,                         // SLM cached copy of merging data
                                    __rng3,                                                     // Destination range
                                    (std::size_t)__sp_local.first,                              // __start1 in LOCAL coordinates because __rng1_cache_slm is work-group SLM cached copy of source data
                                    (std::size_t)__sp_local.second,                             // __start2 in LOCAL coordinates because __rng1_cache_slm is work-group SLM cached copy of source data
                                    __start3,                                                   // __start3 in GLOBAL coordinates because __rng3 is not cached at all
                                    __nd_range_params.chunk,
                                    __rng1_wg_data_size, __rng2_wg_data_size,                   // size of rng1 and rng2
                                    __comp);
                            }
                        }
                    }
                });
        });
    }

#if LOG_ND_RANGE_PARAMS
    // TODO remove debug code
    static void
    __log_nd_range_params(const nd_range_params& __nd_range_params)
    {
        auto tmp = __nd_range_params;

        // wg_count = {tmp.wg_count}, wi_in_one_wg = {tmp.wi_in_one_wg}, chunk = {tmp.chunk}, diags_per_wi = {tmp.diags_per_wi}, additional_diags_inside_wg = {tmp.additional_diags_inside_wg}
        tmp = tmp;
    }
#endif

public:

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    __future<sycl::event>
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        using _Range1ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        using _Range2ValueType = oneapi::dpl::__internal::__value_t<_Range2>;
        static_assert(std::is_same_v<_Range1ValueType, _Range2ValueType>, "In this implementation we can merge only data of the same type");

        using _RangeValueType = _Range1ValueType;

        assert(__rng1.size() + __rng2.size() > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // Calculate nd-range params
        const nd_range_params __nd_range_params = eval_nd_range_params(__exec, __rng1, __rng2);

#if LOG_ND_RANGE_PARAMS
        // TODO remove debug code
        __log_nd_range_params(__nd_range_params);
#endif

        // Create storage for save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
        //  - in GLOBAL coordinates
        using __base_diagonals_sp_storage_t = __result_and_scratch_storage<_ExecutionPolicy, _split_point_t<_IdType>>;
        const std::size_t __base_diagonals_count = __nd_range_params.get_base_diagonals_count_in_all_groups();
        __base_diagonals_sp_storage_t __base_diagonals_sp_global_storage{__exec, 0, __base_diagonals_count};

        // 1. Calculate split points on each base diagonal
        //    - one work-item processing one base diagonal
        sycl::event __event = eval_split_points_for_groups(__exec, __rng1, __rng2, __comp,
                                                           __nd_range_params,
                                                           __base_diagonals_sp_global_storage); 
#if WAIT_IN_IMPL
        // TODO remove debug code
        __event.wait();
#endif

        // 2. Merge data using split points on each base diagonal
        //    - one work-item processing one diagonal
        //    - work-items grouped to process diagonals between two base diagonals (include left base diagonal and exclude right base diagonal)
        __event = run_parallel_merge(__event, __exec, __rng1, __rng2, __rng3, __comp,
                                     __nd_range_params,
                                     __base_diagonals_sp_global_storage);
#if WAIT_IN_IMPL
        // TODO remove debug code
        __event.wait();
#endif

        return __event;
    }
};

template <typename... _Name>
class __merge_kernel_name;

template <typename... _Name>
class __diagonals_kernel_name;

template <typename... _Name>
class __diagonals_in_group_kernel_name;

template <typename... _Name>
class __merge_kernel_name_large;

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    //constexpr std::size_t __starting_size_limit_for_large_submitter = 1 * 1'048'576; // 1 Mb
    constexpr std::size_t __starting_size_limit_for_large_submitter = 10 * 1'024;

    using _Range1ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
    using _Range2ValueType = oneapi::dpl::__internal::__value_t<_Range2>;

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
