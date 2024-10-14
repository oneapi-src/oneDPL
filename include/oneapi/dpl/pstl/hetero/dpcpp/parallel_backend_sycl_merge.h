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

#define USING_PARALLEL_MERGE_SUBMITTER_LARGE_ALWAYS 1

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter_large<_IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        const std::size_t __n1 = __rng1.size();         // Horizontal data
        const std::size_t __n2 = __rng2.size();         // Vertical data
        const std::size_t __n = __n1 + __n2;

        using _ValueType1 = typename std::iterator_traits<decltype(__rng1.begin())>::value_type;
        using _ValueType2 = typename std::iterator_traits<decltype(__rng2.begin())>::value_type;

        assert(__n > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        ////////////////////////////////
        // Detect SLM size

        // Pessimistically only use half of the memory to take into account memory used by compiled kernel                                          // 10'000 + 5'000;  100'000 + 50'000; 16'000'000 + 8'000'000
        const std::size_t __max_slm_size = __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>() / 2;                 // ==65535          ==65'535          ==65'535

        // The size of the data type in bytes
        const std::size_t __wg_src_pairs = oneapi::dpl::__internal::__dpl_ceiling_div(__max_slm_size, sizeof(_ValueType1) + sizeof(_ValueType2));   // ==8'192          == 8'192          == 8'192

        ////////////////////////////////
        // Eval sub-window size processing by work-group
        const std::size_t __window_items_count_h = std::min(__wg_src_pairs, (std::size_t)__n1); // Horizontal size                                  //                  == 8'192          == 8'192
        const std::size_t __window_items_count_v = std::min(__wg_src_pairs, (std::size_t)__n2); // Vertical size                                    //                  == 8'192          == 8'192

        ////////////////////////////////
        // Eval diagonal's distance: each work-group processing one sub-window

        // Returns the maximum number of work-items that this device is capable of executing in a work-group.
        // The minimum value specified in the __wg_size_limit
        const std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec, (std::size_t)4096);                                // ==1'024          ==1'024           == 1'024
                   
        // Empirical number of values to process per work-item
        const std::size_t __diagonals_interval_min = __exec.queue().get_device().is_cpu() ? 128 : 4;
        const std::size_t __diagonals_interval = std::max(__diagonals_interval_min,
                                                  oneapi::dpl::__internal::__dpl_ceiling_div(__window_items_count_h, __wgroup_size));               // ==8              ==8

        ////////////////////////////////
        // Calculate the sub-window amount

        const std::size_t __sub_window_count_h = oneapi::dpl::__internal::__dpl_ceiling_div(__n1, __window_items_count_h);                          // ==2              ==13              ==26 ?
        const std::size_t __sub_window_count_v = oneapi::dpl::__internal::__dpl_ceiling_div(__n2, __window_items_count_v);                          // ==1              ==7               ==9  ?

        ////////////////////////////////
        // Calculate full diagonal count for all data size
        const std::size_t __diagonals_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __diagonals_interval);

        // Create storage for split points (in the best case - in device memory to avoid copy data to host)
        using __split_points_scratch_storage_t = __result_and_scratch_storage<_ExecutionPolicy, _IdType>;
        __split_points_scratch_storage_t __split_points_scratch_storage(__exec, /* 1 item - result, but it's doesn-t used*/ 0, /* split points in diagonals*/ __diagonals_count);

        ////////////////////////////////
        // Two kernels:
        // 1. Find split points on the diagonals
        // 2. Merge by split points

        // Run Kernel 1: find split points at the diagonals
        auto __event_find_split_points = __exec.queue().submit([&](sycl::handler& __cgh) {


            });

        // Run Kernel 2: merge by split points
        auto __event_merge = __exec.queue().submit([&](sycl::handler& __cgh) {

                // We should wait for the first kernel to finish
                __cgh.depends_on(__event_find_split_points);

                [[maybe_unused]] auto __split_points_acc = __split_points_scratch_storage.__get_scratch_acc(__cgh);

            });

        //auto __event = __exec.queue().submit([&](sycl::handler& __cgh) {
        //    oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2, __rng3);
        //    __cgh.parallel_for<_Name...>(
        //        sycl::range</*dim=*/1>(__work_items_count), [=](sycl::item</*dim=*/1> __item_id) {

        //            const auto __item_idx = __item_id.get_linear_id();

        //            // Eval the global diagonal index for the first diagonal in the current work item
        //            const auto __global_diagonal_idx = __item_idx * __diagonals_count_per_work_item;

        //            // Iterate all diagonals in the current work item
        //            for (_IdType __diagonal_idx = 0; __diagonal_idx < __diagonals_count_per_work_item; ++__diagonal_idx)
        //            {
        //                // Eval the offset for the current processing diagonal
        //                const _IdType __diagonal_offset = (__global_diagonal_idx + __diagonal_idx) * __diagonals_interval;
        //                if (__diagonal_offset < __n)
        //                {
        //                    const auto __start = __find_start_point(__rng1, __rng2, __diagonal_offset, __n1, __n2, __comp);

        //                    __serial_merge(__rng1, __rng2, __rng3, __start.first, __start.second, __diagonal_offset,
        //                                   __diagonals_interval, __n1, __n2, __comp);
        //                }
        //            }
        //        });
        //});

        return __future(__event_merge);
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
#if !USING_PARALLEL_MERGE_SUBMITTER_LARGE_ALWAYS
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
#endif // USING_PARALLEL_MERGE_SUBMITTER_LARGE_ALWAYS
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
