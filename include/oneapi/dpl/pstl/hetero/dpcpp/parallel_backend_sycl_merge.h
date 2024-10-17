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
__find_start_point_in(const _Rng1& __rng1, const _Index __rng1_from, const _Index __rng1_to,
                      const _Rng1& __rng2, const _Index __rng2_from, const _Index __rng2_to,
                      const _Index __i_elem,
                      const _Index __n1, const _Index __n2,
                      _Compare __comp)
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

template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
auto
__find_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __i_elem, const _Index __n1,
                   const _Index __n2, _Compare __comp)
{
    return __find_start_point_in(__rng1, 0, __rng1.size(), __rng2, 0, __rng2.size(), __i_elem, __n1, __n2, __comp);
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
struct __parallel_merge_submitter_large_V0;

template <typename _IdType, typename _Name>
struct __parallel_merge_submitter_large_V1;

#define USING_PARALLEL_MERGE_SUBMITTER_LARGE_ALWAYS 1

template <typename... _Name>
class __merge_kernel_name_large;

template <typename... _Name>
class _find_split_points_kernel;

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter_large_V0<_IdType, __internal::__optional_kernel_name<_Name...>>
{
  private:

    // Define the global range
    static constexpr int _Dims2 = 2; // Number of global dimensions in nd-range: 2
    static constexpr int _Dim_H = 0; // Horizontal dimension index in nd-range
    static constexpr int _Dim_V = 1; // Vertical dimension index in nd-range

    // Calculate global range size based on available full amount of work-items on each dimension level for the current device
    template <typename _ExecutionPolicy>
    auto
    __eval_global_range_size(_ExecutionPolicy& __exec, const std::size_t __n1, const std::size_t __n2,
                             const std::size_t __local_size_x) const
    {
        const std::size_t __local_size_y = __local_size_x;

        const std::size_t __max_work_item_size_h = oneapi::dpl::__internal::__max_work_item_sizes<_Dims2>(__exec)[_Dim_H];
        const std::size_t __max_work_item_size_v = oneapi::dpl::__internal::__max_work_item_sizes<_Dims2>(__exec)[_Dim_V];
        assert(__local_size_x <= __max_work_item_size_h);
        assert(__local_size_y <= __max_work_item_size_v);

        const std::size_t __global_size_x_fit_into_n1 = oneapi::dpl::__internal::__dpl_ceiling_div(__n1, __local_size_x);
        const std::size_t __global_size_x_fit_into_n2 = oneapi::dpl::__internal::__dpl_ceiling_div(__n2, __local_size_y);

        const std::size_t __global_size_x = __global_size_x_fit_into_n1 * __local_size_x;
        assert(__global_size_x <= __max_work_item_size_h);

        const std::size_t __global_size_y = __global_size_x_fit_into_n2 * __local_size_y;
        assert(__global_size_y <= __max_work_item_size_v);

        return std::make_tuple(__global_size_x, __global_size_y);
    }

  public:

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp) const
    {
        const std::size_t __n1 = __rng1.size();         // Horizontal data
        const std::size_t __n2 = __rng2.size();         // Vertical data
        const std::size_t __n = __n1 + __n2;

        using _ValueType1 = typename std::iterator_traits<decltype(__rng1.begin())>::value_type;
        using _ValueType2 = typename std::iterator_traits<decltype(__rng2.begin())>::value_type;

        assert(__n1 > 0);                                                                   // 17'000'000
        assert(__n2 > 0);                                                                   //  8'500'000
        assert(__n > 0);        // TODO should we remove this assert?                       // 25'500'000

        // Build Kernel name for split points Kernel
        using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
        using _FindSplitPointsKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            _find_split_points_kernel, _CustomName, _Range1, _Range2, _Range3, _IdType, _Compare>;

        _PRINT_INFO_IN_DEBUG_MODE(__exec);

        // TODO required special implementation for two corner cases when *__rng1.rbegin() <= *__rng2.begin()
        // or *__rng2.rbegin() <= *__rng1.begin() : in these cases we should simple copy source data into __rng3

        // Returns the maximum number of work-items that this device is capable of executing in a work-group.
        // The minimum value specified in the __wg_size_limit
        const std::size_t __max_wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec/*, (std::size_t)4096*/);

        // Pessimistically only use half of the memory to take into account memory used by compiled kernel
        const std::size_t __max_slm_size = __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>() / 2;

        // The size of the data type in bytes
        const std::size_t __wg_src_pairs = oneapi::dpl::__internal::__dpl_ceiling_div(__max_slm_size, sizeof(_ValueType1) + sizeof(_ValueType2));

        // Calculate the local range
        const std::size_t __local_size = std::min(std::min(std::max(__n1, __n2), __max_wgroup_size), __wg_src_pairs);

        // Calculate global range size
        const auto [__global_size_x, __global_size_y] = __eval_global_range_size(__exec, __n1, __n2, __local_size);

        // Define nd-ranges
        const sycl::range<_Dims2>    __global_range         {__global_size_x, __global_size_y};
        const sycl::range<_Dims2>    __local_range          {__local_size,    1              };
        const sycl::nd_range<_Dims2> __merge_matrix_nd_range{__global_range,  __local_range  };

        ////////////////////////////////
        // Eval diagonal's distance: each work-group processing one sub-window

        // Empirical number of values to process per work-item
        //const std::size_t __diagonals_interval = __exec.queue().get_device().is_cpu() ? 128 : 4;
        const std::size_t __diagonals_interval = 4;

        ////////////////////////////////
        // Calculate full diagonal count for all data size
        const std::size_t __diagonals_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __diagonals_interval);

        // Define the type for describing split point
        using __split_point_t = std::pair<_IdType, _IdType>;

        // Create storage for split points (in the best case - in device memory to avoid copy data to host)
        using __split_points_scratch_storage_t = __result_and_scratch_storage<_ExecutionPolicy, __split_point_t>;
        __split_points_scratch_storage_t __split_points(__exec,
                                                        0,                      // The amount of result items
                                                        __diagonals_count);     // The amount of split points

        ////////////////////////////////
        // Two kernels:
        // 1. Find split points on the diagonals
        // 2. Merge by split points

        // Run Kernel 1: find split points at the diagonals
        auto __event_find_split_points = __exec.queue().submit([&](sycl::handler& __cgh) {

            const auto __cashed_items_count = __local_size + 1;     // The number of items of __rng1 and __rng2 to cash in SLM

            // Cash the portion of source processing data for the current work-group in SLM
            __dpl_sycl::__local_accessor<_ValueType1> __loc_acc_rng1_h(__cashed_items_count, __cgh);      // Cashed data from __rng1
            __dpl_sycl::__local_accessor<_ValueType2> __loc_acc_rng2_v(__cashed_items_count, __cgh);      // Cashed data from __rng2

            // Get access to the split points
            auto __split_points_acc = __split_points.__get_scratch_acc(__cgh);
            auto __split_points_ptr = __split_points_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__split_points_acc);

            __cgh.parallel_for<_FindSplitPointsKernel>(
                __merge_matrix_nd_range,
                [=](sycl::nd_item<_Dims2> __nd_item) {

                    // Return the number of work-groups for Dimension in the iteration space.
                    const auto __work_groups_amount_h = __nd_item.get_group_range(_Dim_H);
                    const auto __work_groups_amount_v = __nd_item.get_group_range(_Dim_V);

                    // Return the constituent element of the global id representing the work-items position in the nd-range in the given Dimension.
                    const auto __global_id = __nd_item.get_global_id(_Dim_H);

                    // Return the constituent element of the local id representing the work-items position within the current work-group in the given Dimension.
                    const auto __local_id = __nd_item.get_local_id(_Dim_H);
                    assert(__nd_item.get_local_id(_Dim_V) == 1);

                    // 1. Load data into SLM, if we are on first row or first column
                    {
                        //          Work-items in the current work-group: 'L' - load data; '.' - doing nothing.
                        //          0 1 2 3 4 5 6 7 8 9 .....  <-- __local_id
                        //        0 L . . . . . . . . .                             load: __rng1[0], __rng2[0]
                        //        1 . L . . . . . . . .                             load: __rng1[1], __rng2[1]
                        //        2 . . L . . . . . . .                             load: __rng1[2], __rng2[2]
                        //        3 . . . L . . . . . .                             load: __rng1[3], __rng2[3]
                        //        4 . . . . L . . . . .                             load: __rng1[4], __rng2[4]
                        //                    L                                     load: __rng1[5]
                        //                      L                                   load: __rng1[6]
                        //                        L                                 load: __rng1[7]
                        //                          L                               load: __rng1[8]
                        //                            L                             load: __rng1[9]
                        //      ...
                        //        ^
                        // __local_id

                        // Load the extra elements on the right and on the bottom side of the current sub-window
                        if (__local_id + 1 == __local_size && __global_id + 1 < __n1)
                            __loc_acc_rng1_h[__local_id + 1] = __rng1[__global_id + 1];
                        if (__local_id + 1 == __local_size && __global_id + 1 < __n2)
                            __loc_acc_rng2_v[__local_id + 1] = __rng2[__global_id + 1];
                    }

                    // 2. Barrier to wait for all work-items in the current work-group to load data into SLM
                    __dpl_sycl::__group_barrier(__nd_item);

                    // 3. Analyze data placed on diagonal: find split points and save results into the scratch storage
                    //          Work-items in the current work-group: L - load data; X - doing nothing.
                    //          0 1 2 3 4 5 6 7 8 9 .....  <-- __local_id_h
                    //        0        /
                    //        1       <upper-right point> (3, 1)
                    //        2      <current point> (2, 2)    -> check comparison for current and upper-right points: __comp(__rng1[2], __rng2[2]) != __comp(__rng1[3], __rng2[1])
                    //        3     /                                - if the result is true -> (3,1) is a split point
                    //        4    /
                    //      ...
                    //        ^
                    // __local_id_v
#if 0
                    const auto __current_diagonal_global_offset = __global_id_h + __global_id_v;
                    if (__current_diagonal_global_offset % __diagonals_interval == 0)
                    {
                        // 3.1 Add init point
                        if (__global_id_h == 0 && __global_id_v == 0)
                        {
                            __split_points_ptr[0] = __split_point_t{0, 0};
                        }

                        // TODO required analyze the case when __n1 == 1 and / or __n2 == 1

                        // 3.2 Analyze current point and upper-right point
                        else if (__local_id_h + 1 < __local_size_y + 1 && __global_id_h + 1 < __n1 && __local_id_v > 0)
                        {
                            if (__comp(__loc_acc_rng1_h[__local_id_h], __loc_acc_rng2_v[__local_id_v]) !=
                                __comp(__loc_acc_rng1_h[__local_id_h + 1], __loc_acc_rng2_v[__local_id_v - 1]))
                            {
                                // Add the global coordinates of upper-right point of current as a found split point
                                __split_points_ptr[__current_diagonal_global_offset] = __split_point_t{__global_id_h + 1, __global_id_v - 1};
                            }
                        }
                    }
#endif
                });
        });

        // Run Kernel 2: merge by split points
        auto __event_merge = __exec.queue().submit([&](sycl::handler& __cgh) {

            // We should wait for the first kernel to finish
            __cgh.depends_on(__event_find_split_points);

            // Get access to the split points
            auto __split_points_acc = __split_points.__get_scratch_acc(__cgh);
            auto __split_points_ptr = __split_points_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__split_points_acc);

            __cgh.parallel_for<_Name...>(
                sycl::range</*dim=*/1>(__diagonals_count), [=](sycl::item</*dim=*/1> __item_id) {
                    const auto __idx = __item_id.get_linear_id();

                    // Get current splitting point
                    const auto& __start = __split_points_ptr[__idx];
                    assert(__start.first > 0 || __start.second > 0 || __idx == 0);

                    const _IdType __diagonal_offset = __idx * __diagonals_interval;

                    // Run merge from current splitting point
                    __serial_merge(__rng1, __rng2, __rng3, (std::size_t)__start.first, (std::size_t)__start.second,
                                   (std::size_t)__diagonal_offset, __diagonals_interval, __n1, __n2, __comp);
                });
        });

        return __future(__event_merge);
    }
};

template <typename _IdType, typename... _Name>
struct __parallel_merge_submitter_V1<_IdType, __internal::__optional_kernel_name<_Name...>>
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

template <typename... _Name>
class __merge_kernel_name;

template <typename _IdType, typename... _Name>
using __parallel_merge_submitter_large = __parallel_merge_submitter_large_V0<_IdType, ..._Name>;
//using __parallel_merge_submitter_large = __parallel_merge_submitter_large_V0<_IdType, ..._Name>;

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
            return __parallel_merge_submitter_large_V0<_WiIndex, _MergeKernel>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
        else
        {
            using _WiIndex = std::uint64_t;
            using _MergeKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __merge_kernel_name_large<_CustomName, _WiIndex>>;
            return __parallel_merge_submitter_large_V0<_WiIndex, _MergeKernel>()(
                std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
                std::forward<_Range3>(__rng3), __comp);
        }
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_H
