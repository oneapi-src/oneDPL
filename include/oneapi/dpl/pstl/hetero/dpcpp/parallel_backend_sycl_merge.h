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

// TODO required to remove later
#define DEBUG_MERGE_GET_SET_VALUE 0

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

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
std::pair<_Index, _Index>
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2,
                   _Index __diagonal_offset, _Index __n1, _Index __n2,
                   _Compare __comp,
                   const std::pair<_Index, _Index>& __sub_window_offset = {0, 0})
{
    assert(__n1 >= __sub_window_offset.first);
    assert(__n2 >= __sub_window_offset.second);
    assert(__diagonal_offset >= __sub_window_offset.first + __sub_window_offset.second);

    __n1 -= __sub_window_offset.first;
    __n2 -= __sub_window_offset.second;
    __diagonal_offset -= __sub_window_offset.first + __sub_window_offset.second;

    std::pair<_Index, _Index> result{0, 0};

    //searching for the first '1', a lower bound for a diagonal [0, 0,..., 0, 1, 1,.... 1, 1]
    oneapi::dpl::counting_iterator<_Index> __diag_it(0);

    if (__diagonal_offset < __n2) //a condition to specify upper or lower part of the merge matrix to be processed
    {
        const _Index __q = __diagonal_offset;                //diagonal index
        const _Index __n_diag = std::min<_Index>(__q, __n1); //diagonal size
        auto __res =
            std::lower_bound(__diag_it, __diag_it + __n_diag, 1 /*value to find*/,
                             [&__rng2, &__rng1, __q, __comp](const auto& __i_diag, const auto& __value) mutable {
                                 const auto __zero_or_one = __comp(__rng2[__q - __i_diag - 1], __rng1[__i_diag]);
                                 return __zero_or_one < __value;
                             });
        result = std::make_pair(*__res, __q - *__res);
    }
    else
    {
        const _Index __q = __diagonal_offset - __n2;                //diagonal index
        const _Index __n_diag = std::min<_Index>(__n1 - __q, __n2); //diagonal size
        auto __res =
            std::lower_bound(__diag_it, __diag_it + __n_diag, 1 /*value to find*/,
                             [&__rng2, &__rng1, __n2, __q, __comp](const auto& __i_diag, const auto& __value) mutable {
                                 const auto __zero_or_one = __comp(__rng2[__n2 - __i_diag - 1], __rng1[__q + __i_diag]);
                                 return __zero_or_one < __value;
                             });
        result = std::make_pair(__q + *__res, __n2 - *__res);
    }

    result.first += __sub_window_offset.first;
    result.second += __sub_window_offset.second;

    return result;
}

#if DEBUG_MERGE_GET_SET_VALUE
template <typename _Rng, typename _Index>
const auto&
__get_rng_item_val(const _Rng& __rng, const _Index __idx, _Index& __idx_saved)
{
    if (__idx >= __rng.size())
    {
        auto __actual_size = __rng.size();
        __actual_size = __actual_size; // Invalid access:: required to investigate!
    }

    __idx_saved = __idx;
    return __rng[__idx];
}

template <typename _Rng, typename _Index, typename _Value>
void
__set_rng_item_val(_Rng& __rng, _Index __idx, const _Value& __val)
{
    if (__idx >= __rng.size())
    {
        auto __actual_size = __rng.size();
        __actual_size = __actual_size; // Invalid access:: required to investigate!
    }
    __rng[__idx] = __val;
}
#endif

// Do serial merge of the data from rng1 (starting from start1) and rng2 (starting from start2) and writing
// to rng3 (starting from start3) in 'chunk' steps, but do not exceed the total size of the sequences (n1 and n2)
template <typename _Rng1, typename _Rng2, typename _Rng3, typename _Index, typename _Compare>
[[maybe_unused]] std::pair<_Index, _Index>
__serial_merge(const _Rng1& __rng1, const _Rng2& __rng2, _Rng3& __rng3,
               _Index __start1, _Index __start2, _Index __start3,
               const std::uint8_t __diagonals_interval,
               _Index __n1, _Index __n2, _Compare __comp)
{
    std::pair<_Index, _Index> result{0, 0};

    if (__start1 >= __n1)
    {
        result.first = __n1;

        //copying a residual of the second seq
        const _Index __n = std::min<_Index>(__n2 - __start2, __diagonals_interval);
        for (std::uint8_t __i = 0; __i < __n; ++__i)
        {
#if !DEBUG_MERGE_GET_SET_VALUE
            __rng3[__start3 + __i] = __rng2[__start2 + __i];
            result.second = __start2 + __i;
#else
            __set_rng_item_val(__rng3, __start3 + __i, __get_rng_item_val(__rng2, __start2 + __i, result.second));
#endif
        }
    }
    else if (__start2 >= __n2)
    {
        result.second = __n2;

        //copying a residual of the first seq
        const _Index __n = std::min<_Index>(__n1 - __start1, __diagonals_interval);
        for (std::uint8_t __i = 0; __i < __n; ++__i)
        {
#if !DEBUG_MERGE_GET_SET_VALUE
            __rng3[__start3 + __i] = __rng1[__start1 + __i];
            result.first = __start1 + __i;
#else
            __set_rng_item_val(__rng3, __start3 + __i, __get_rng_item_val(__rng1, __start1 + __i, result.first));
#endif
        }
    }
    else
    {
        for (std::uint8_t __i = 0; __i < __diagonals_interval && __start1 < __n1 && __start2 < __n2; ++__i)
        {
            const auto& __val1 = __rng1[__start1];
            const auto& __val2 = __rng2[__start2];

            if (__comp(__val2, __val1))
            {
                result.second = __start2;
#if !DEBUG_MERGE_GET_SET_VALUE
                __rng3[__start3 + __i] = __val2;
#else
                __set_rng_item_val(__rng3, __start3 + __i, __val2);
#endif

                // Pair operation fot the next ++__start2 inside if condition
                ++result.second;
                if (++__start2 == __n2)
                {
                    //copying a residual of the first seq
                    for (++__i; __i < __diagonals_interval && __start1 < __n1; ++__i, ++__start1)
                    {
#if !DEBUG_MERGE_GET_SET_VALUE
                        __rng3[__start3 + __i] = __rng1[__start1];
                        result.first = __start1;
#else
                        __set_rng_item_val(__rng3, __start3 + __i, __get_rng_item_val(__rng1, __start1, result.first));
#endif
                    }
                }
            }
            else
            {
                result.first = __start1;
#if !DEBUG_MERGE_GET_SET_VALUE
                __rng3[__start3 + __i] = __val1;
#else
                __set_rng_item_val(__rng3, __start3 + __i, __val1);
#endif

                // Pair operation fot the next ++__start1 inside if condition
                ++result.first;
                if (++__start1 == __n1)
                {
                    //copying a residual of the second seq
                    for (++__i; __i < __diagonals_interval && __start2 < __n2; ++__i, ++__start2)
                    {
#if !DEBUG_MERGE_GET_SET_VALUE
                        __rng3[__start3 + __i] = __rng2[__start2];
                        result.second = __start2;
#else
                        __set_rng_item_val(__rng3, __start3 + __i, __get_rng_item_val(__rng2, __start2, result.second));
#endif
                    }
                }
            }
        }
    }

    return result;
}

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _IdType, typename _Name>
struct __parallel_merge_submitter;

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter<_IdType, __internal::__optional_kernel_name<_Name...>>
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
        //const std::uint8_t __diagonals_interval = __exec.queue().get_device().is_cpu() ? 128 : 4;
        const std::uint8_t __diagonals_interval = 10;

        // Number of diagonals to process
        const _IdType __diagonals_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __diagonals_interval);

        auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            __cgh.parallel_for<_Name...>(
                sycl::range</*dim=*/1>(__diagonals_count), [=](sycl::item</*dim=*/1> __item_id) {
                    const _IdType __diagonal_offset = __item_id.get_linear_id() * __diagonals_interval;
                    const auto __start = __find_start_point(__rng1, __rng2, __diagonal_offset, __n1, __n2, __comp);
                    __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __diagonal_offset,
                                   __diagonals_interval, __n1, __n2, __comp);
                });
        });

        return __future(__event);
    }
};

template <typename _IdType, typename _Name>
struct __parallel_merge_submitter_large;

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter_large<_IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        const _IdType __n1 = __rng1.size();
        const _IdType __n2 = __rng2.size();
        const _IdType __n = __n1 + __n2;

        assert(__n > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // Empirical number of values to process per work-item
        const std::uint8_t __diagonals_interval = __exec.queue().get_device().is_cpu() ? 128 : 4;

        // Returns the number of parallel compute units available to the device. The minimum value is 1.
        const std::uint32_t __max_cu = oneapi::dpl::__internal::__max_compute_units(__exec);

        // Eval the amount of diagonals in all data to process
        const _IdType __diagonals_count_per_all_data = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __diagonals_interval);

        // Eval the amount of diagonals in each work item
        const _IdType __diagonals_count_per_work_item = oneapi::dpl::__internal::__dpl_ceiling_div(__diagonals_count_per_all_data, __max_cu);

        // Eval the required amount of work items
        const auto __work_items_count = oneapi::dpl::__internal::__dpl_ceiling_div(__diagonals_count_per_all_data, __diagonals_count_per_work_item);

        auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
            __cgh.parallel_for<_Name...>(
                sycl::range</*dim=*/1>(__work_items_count), [=](sycl::item</*dim=*/1> __item_id) {

                    const auto __item_idx = __item_id.get_linear_id();

                    // Eval the global diagonal index for the first diagonal in the current work item
                    const auto __global_diagonal_idx = __item_idx * __diagonals_count_per_work_item;

                    // top left offset for data finding
                    std::pair<_IdType, _IdType> __sub_window_offset{0, 0};

                    // Iterate all diagonals in the current work item
                    for (_IdType __diagonal_idx = 0; __diagonal_idx < __diagonals_count_per_work_item; ++__diagonal_idx)
                    {
                        // Eval the offset for the current processing diagonal
                        const _IdType __diagonal_offset = (__global_diagonal_idx + __diagonal_idx) * __diagonals_interval;
                        if (__diagonal_offset < __n)
                        {
                            const auto __start = __find_start_point(__rng1, __rng2, __diagonal_offset, __n1, __n2, __comp, __sub_window_offset);

                            __sub_window_offset = __serial_merge(__rng1, __rng2, __rng3,
                                                                 __start.first, __start.second, __diagonal_offset,
                                                                 __diagonals_interval,
                                                                 __n1, __n2, __comp);
                        }
                    }
                });
        });

        return __future(__event);
    }
};

template <typename... _Name>
class __merge_kernel_name;

template <typename... _Name>
class __merge_kernel_name_large;

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __rng1,
                 _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const auto __n = __rng1.size() + __rng2.size();

    // TODO required to find the better value instead of 16 * 1'048'576 on GPU
    if (__n < 16 * 1'048'576)
    {
        if (__n <= std::numeric_limits<std::uint32_t>::max())
        {
            using _WiIndex = std::uint32_t;
            using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter<_WiIndex, _MergeKernel>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter<_WiIndex, _MergeKernel>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
    }
    else
    {
        if (__n <= std::numeric_limits<std::uint32_t>::max())
        {
            using _WiIndex = std::uint32_t;
            using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_WiIndex, _MergeKernel>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large<_WiIndex, _MergeKernel>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
