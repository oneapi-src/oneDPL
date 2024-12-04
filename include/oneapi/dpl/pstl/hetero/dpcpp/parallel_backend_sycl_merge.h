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
auto
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __i_elem, const _Index __n1,
                   const _Index __n2, _Compare __comp)
{
    //searching for the first '1', a lower bound for a diagonal [0, 0,..., 0, 1, 1,.... 1, 1]
    oneapi::dpl::counting_iterator<_Index> __diag_it(0);

    if (__i_elem < __n2) //a condition to specify upper or lower part of the merge matrix to be processed
    {
        const _Index __q = __i_elem;                         //diagonal index
        const _Index __n_diag = std::min<_Index>(__q, __n1); //diagonal size
        auto __res =
            std::lower_bound(__diag_it, __diag_it + __n_diag, 1 /*value to find*/,
                             [&__rng2, &__rng1, __q, __comp](const auto& __i_diag, const auto& __value) mutable {
                                 const auto __zero_or_one = __comp(__rng2[__q - __i_diag - 1], __rng1[__i_diag]);
                                 return __zero_or_one < __value;
                             });
        return std::make_pair(*__res, __q - *__res);
    }
    else
    {
        const _Index __q = __i_elem - __n2;                         //diagonal index
        const _Index __n_diag = std::min<_Index>(__n1 - __q, __n2); //diagonal size
        auto __res =
            std::lower_bound(__diag_it, __diag_it + __n_diag, 1 /*value to find*/,
                             [&__rng2, &__rng1, __n2, __q, __comp](const auto& __i_diag, const auto& __value) mutable {
                                 const auto __zero_or_one = __comp(__rng2[__n2 - __i_diag - 1], __rng1[__q + __i_diag]);
                                 return __zero_or_one < __value;
                             });
        return std::make_pair(__q + *__res, __n2 - *__res);
    }
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
__find_start_point_in(const _Rng1& __rng1, const _Index __rng1_from, _Index __rng1_to, const _Rng2& __rng2,
                      const _Index __rng2_from, _Index __rng2_to, const _Index __i_elem, _Compare __comp)
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

        const _split_point_t<_Index> __result{ *__res, __index_sum - *__res + 1 };
        assert(__result.first + __result.second == __i_elem);

        assert(__rng1_from <= __result.first && __result.first <= __rng1_to);
        assert(__rng2_from <= __result.second && __result.second <= __rng2_to);

        return __result;
    }
    else
    {
        assert(__rng1_from == 0);
        assert(__rng2_from == 0);
        return { __rng1_from, __rng2_from };
    }
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

template <typename _IdType, typename _CustomName, typename _DiagonalsKernelName, typename _MergeKernelName1, typename _MergeKernelName2>
struct __parallel_merge_submitter;

template <typename _IdType, typename _CustomName, typename... _DiagonalsKernelName, typename... _MergeKernelName1, typename... _MergeKernelName2>
struct __parallel_merge_submitter<_IdType, _CustomName,
                                        __internal::__optional_kernel_name<_DiagonalsKernelName...>,
                                        __internal::__optional_kernel_name<_MergeKernelName1...>,
                                        __internal::__optional_kernel_name<_MergeKernelName2...>>
{
protected:

    struct nd_range_params
    {
        std::size_t  base_diag_count = 0;
        std::size_t  steps_between_two_base_diags = 0;
        std::uint8_t chunk = 0;
        _IdType      steps = 0;
    };

    // Calculate nd-range params
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
    nd_range_params
    eval_nd_range_params(_ExecutionPolicy&& __exec, const _Range1& __rng1, const _Range2& __rng2) const
    {
        using _Range1ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        using _Range2ValueType = oneapi::dpl::__internal::__value_t<_Range2>;
        using _RangeValueType = std::conditional_t<(sizeof(_Range1ValueType) > sizeof(_Range2ValueType)), _Range1ValueType, _Range2ValueType>;

        const std::size_t __n = __rng1.size() + __rng2.size();

        constexpr std::size_t __slm_bank_size = 16;     // TODO is it correct value? How to get it from hardware?

        // Calculate how many data items we can read into one SLM bank
        constexpr std::size_t __data_items_in_slm_bank = oneapi::dpl::__internal::__dpl_ceiling_div(__slm_bank_size, sizeof(_RangeValueType));

        // Empirical number of values to process per work-item
        const std::uint8_t __chunk = __exec.queue().get_device().is_cpu() ? 128 : __data_items_in_slm_bank;

        const _IdType __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);
        const _IdType __base_diag_count = __use_base_diags ? 32 * 1'024 : 0;
        const _IdType __steps_between_two_base_diags = __use_base_diags ? oneapi::dpl::__internal::__dpl_ceiling_div(__steps, __base_diag_count) : 0;

        return { __base_diag_count, __steps_between_two_base_diags, __chunk, __steps };
    }

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
            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<_DiagonalsKernelName...>(
                sycl::range</*dim=*/1>(__nd_range_params.base_diag_count + 1), [=](sycl::item</*dim=*/1> __item_id) {
                    auto __global_idx = __item_id.get_linear_id();
                    auto __base_diagonals_sp_global_ptr = _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);

                    _split_point_t<_IdType> __sp = __global_idx == 0 ? _split_point_t<_IdType>{ 0, 0 } : _split_point_t<_IdType>{ __n1, __n2 };

                    if (0 < __global_idx && __global_idx < __nd_range_params.base_diag_count)
                    {
                        const _IdType __i_elem = __global_idx * __nd_range_params.steps_between_two_base_diags * __nd_range_params.chunk;
                        if (__i_elem < __n)
                            __sp = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                    }

                    __base_diagonals_sp_global_ptr[__global_idx] = __sp;
                });
        });

        return __event;
    }

    // Process parallel merge
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    sycl::event
    run_parallel_merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp,
                       const nd_range_params& __nd_range_params) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();

        sycl::event __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);

            __cgh.parallel_for<_MergeKernelName1...>(
                sycl::range</*dim=*/1>(__nd_range_params.steps), [=](sycl::item</*dim=*/1> __item_id) {
                    auto __global_idx = __item_id.get_linear_id();
                    const _IdType __i_elem = __global_idx * __nd_range_params.chunk;

                    _split_point_t<_IdType> __start = __find_start_point(__rng1, __rng2, __i_elem, __n1, __n2, __comp);
                    __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __nd_range_params.chunk, __n1, __n2, __comp);
                });
        });

        return __event;
    }

    // Process parallel merge
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
              typename _Storage>
    sycl::event
    run_parallel_merge(sycl::event __event,
                       _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp,
                       const nd_range_params& __nd_range_params,
                       const _Storage& __base_diagonals_sp_global_storage) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();

        __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            auto __base_diagonals_sp_global_acc = __base_diagonals_sp_global_storage.template __get_scratch_acc<sycl::access_mode::read>(__cgh);

            __cgh.depends_on(__event);

            __cgh.parallel_for<_MergeKernelName2...>(
                sycl::range</*dim=*/1>(__nd_range_params.steps), [=](sycl::item</*dim=*/1> __item_id) {
                    auto __global_idx = __item_id.get_linear_id();
                    const _IdType __i_elem = __global_idx * __nd_range_params.chunk;

                    auto __base_diagonals_sp_global_ptr = _Storage::__get_usm_or_buffer_accessor_ptr(__base_diagonals_sp_global_acc);
                    auto __diagonal_idx = __global_idx / __nd_range_params.steps_between_two_base_diags;

                    _split_point_t<_IdType> __start;
                    if (__global_idx % __nd_range_params.steps_between_two_base_diags != 0)
                    {
                        // Check that we fit into size of scratch
                        assert(__diagonal_idx + 1 < __nd_range_params.base_diag_count + 1);

                        const _split_point_t<_IdType> __sp_left = __base_diagonals_sp_global_ptr[__diagonal_idx];
                        const _split_point_t<_IdType> __sp_right = __base_diagonals_sp_global_ptr[__diagonal_idx + 1];

                        __start = __find_start_point_in(__rng1, __sp_left.first, __sp_right.first, __rng2,
                                                        __sp_left.second, __sp_right.second, __i_elem, __comp);
                    }
                    else
                    {
                        __start = __base_diagonals_sp_global_ptr[__diagonal_idx];
                    }

                    __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __i_elem, __nd_range_params.chunk, __n1, __n2, __comp);
                });
        });

        return __event;
    }

public:

    __parallel_merge_submitter(bool __use_base_diags)
        : __use_base_diags(__use_base_diags)
    {
    }

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        assert(__rng1.size() > 0 || __rng2.size() > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // Calculate nd-range params
        const nd_range_params __nd_range_params = eval_nd_range_params(__exec, __rng1, __rng2);

        // Create storage for save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
        using __base_diagonals_sp_storage_t = __result_and_scratch_storage<_ExecutionPolicy, _split_point_t<_IdType>>;

        __result_and_scratch_storage_base_ptr __p_result_and_scratch_storage_base;

        // Calculation of split points on each base diagonal
        sycl::event __event;
        if (__use_base_diags)
        {
            auto __p_base_diagonals_sp_global_storage = new __base_diagonals_sp_storage_t(__exec, 0, __nd_range_params.base_diag_count + 1);
            __p_result_and_scratch_storage_base.reset(static_cast<__result_and_scratch_storage_base*>(__p_base_diagonals_sp_global_storage));

            __event = eval_split_points_for_groups(__exec, __rng1, __rng2, __comp, __nd_range_params, *__p_base_diagonals_sp_global_storage);

            // Merge data using split points on each base diagonal
            __event = run_parallel_merge(__event, __exec, __rng1, __rng2, __rng3, __comp, __nd_range_params, *__p_base_diagonals_sp_global_storage);
        }
        else
        {
            // Merge data using split points on each base diagonal
            __event = run_parallel_merge(__exec, __rng1, __rng2, __rng3, __comp, __nd_range_params);
        }

        return __future(__event, std::move(__p_result_and_scratch_storage_base));
    }

private:

    const bool __use_base_diags = false;
};

template <typename... _Name>
class __merge_kernel_name1;

template <typename... _Name>
class __merge_kernel_name2;

template <typename... _Name>
class __diagonals_kernel_name;

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const std::size_t __n = __rng1.size() + __rng2.size();

    constexpr std::size_t __starting_size_limit_for_large_submitter = 4 * 1'048'576; // 4 MB
    const bool __use_base_diags = __n < __starting_size_limit_for_large_submitter;

    if (__n <= std::numeric_limits<std::uint32_t>::max())
    {
        using _WiIndex = std::uint32_t;
        using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __diagonals_kernel_name<_CustomName, _WiIndex>>;
        using _MergeKernelName1 = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name1<_CustomName, _WiIndex>>;
        using _MergeKernelName2 = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name2<_CustomName, _WiIndex>>;
        return __parallel_merge_submitter<_WiIndex, _CustomName, _DiagonalsKernelName, _MergeKernelName1, _MergeKernelName2>(__use_base_diags)(
            std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            std::forward<_Range3>(__rng3), __comp);
    }
    else
    {
        using _WiIndex = std::uint64_t;
        using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __diagonals_kernel_name<_CustomName, _WiIndex>>;
        using _MergeKernelName1 = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name1<_CustomName, _WiIndex>>;
        using _MergeKernelName2 = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name2<_CustomName, _WiIndex>>;
        return __parallel_merge_submitter<_WiIndex, _CustomName, _DiagonalsKernelName, _MergeKernelName1, _MergeKernelName2>(__use_base_diags)(
            std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            std::forward<_Range3>(__rng3), __comp);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
