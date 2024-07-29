// -*- C++ -*-
//===-- parallel_backend_sycl_merge_sort.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H

#include <cmath>     // std::log2
#include <limits>    // std::numeric_limits
#include <cassert>   // assert
#include <utility>   // std::swap
#include <cstdint>   // std::uint32_t, ...
#include <algorithm> // std::min

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "parallel_backend_sycl_merge.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <std::uint16_t __data_per_workitem>
struct __subgroup_bubble_sorter
{
    template <typename _Storage, typename _Compare, typename _Size>
    void
    sort(_Storage& __storage, _Compare __comp, _Size __start, _Size __end) const
    {
        for (std::int64_t i = __start; i < __end; ++i)
        {
            for (std::int64_t j = __start + 1; j < __start + __end - i; ++j)
            {
                // forwarding references allow binding of internal tuple of references with rvalue
                // TODO: avoid bank conflicts, or reconsider the algorithm
                auto&& __first_item = __storage[j - 1];
                auto&& __second_item = __storage[j];
                if (__comp(__second_item, __first_item))
                {
                    using std::swap;
                    swap(__first_item, __second_item);
                }
            }
        }
    }
};

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size>
struct __group_merge_path_sorter
{
    template <typename _Storage, typename _Compare, typename _Size1, typename _Size2>
    bool
    sort(const sycl::nd_item<1>& __item, _Storage& __storage, _Compare __comp, _Size1 __start, _Size1 __end,
         _Size2 __sorted) const
    {
        constexpr std::uint32_t __sorted_final = __data_per_workitem * __workgroup_size;
        static_assert((__sorted_final & (__sorted_final - 1)) == 0);

        const std::uint32_t __id = __item.get_local_linear_id() * __data_per_workitem;

        bool __data_in_temp = false;
        std::uint32_t __next_sorted = __sorted * 2;
        std::int16_t __iters = std::log2(__sorted_final) - std::log2(__sorted);
        for (std::int16_t __i = 0; __i < __iters; ++__i)
        {
            const std::uint32_t __id_local = __id % __next_sorted;
            // Borders of the ranges to be merged
            const std::uint32_t __start1 = std::min<std::uint32_t>((__id / __next_sorted) * __next_sorted, __end);
            const std::uint32_t __end1 = std::min<std::uint32_t>(__start1 + __sorted, __end);
            const std::uint32_t __start2 = __end1;
            const std::uint32_t __end2 = std::min<std::uint32_t>(__start2 + __sorted, __end);
            const std::uint32_t __n1 = __end1 - __start1;
            const std::uint32_t __n2 = __end2 - __start2;

            const auto& __in = __storage.begin() + __data_in_temp * __sorted_final;
            const auto& __out = __storage.begin() + (!__data_in_temp) * __sorted_final;
            const auto& __in1 = __in + __start1;
            const auto& __in2 = __in + __start2;

            const auto __start = __find_start_point(__in1, __in2, __id_local, __n1, __n2, __comp);
            __serial_merge(__in1, __in2, __out, __start.first, __start.second, __id, __data_per_workitem, __n1, __n2,
                           __comp);
            __dpl_sycl::__group_barrier(__item);

            __sorted = __next_sorted;
            __next_sorted *= 2;
            __data_in_temp = !__data_in_temp;
        }
        return __data_in_temp;
    }
};

template <std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size, typename _Range, typename _Compare>
struct __leaf_sorter
{
    using _T = oneapi::dpl::__internal::__value_t<_Range>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Range>;
    using _Storage = __dpl_sycl::__local_accessor<_T>;
    // TODO: select a better sub-group sorter depending on sort stability,
    //       a type (e.g. it can be trivially copied for shuffling within a sub-group)
    using _SubGroupSorter = __subgroup_bubble_sorter<__data_per_workitem>;
    using _GroupSorter = __group_merge_path_sorter<__data_per_workitem, __workgroup_size>;

    static constexpr std::uint32_t __wg_process_size = __data_per_workitem * __workgroup_size;

    static auto
    storage(sycl::handler& __cgh)
    {
        return _Storage(2 * __wg_process_size, __cgh);
    }

    __leaf_sorter(_Range& __rng, _Compare __comp, _Storage __storage, std::int64_t __n)
        : __rng(__rng), __comp(__comp), __storage(__storage), __n(__n), __sub_group_sorter(), __group_sorter()
    {
    }

    void
    sort(const sycl::nd_item<1>& __item) const
    {
        sycl::sub_group __sg = __item.get_sub_group();
        sycl::group __wg = __item.get_group();
        std::uint32_t __wg_id = __wg.get_group_linear_id();
        std::uint32_t __sg_id = __sg.get_group_linear_id();
        std::uint32_t __sg_size = __sg.get_local_linear_range();
        std::uint32_t __sg_inner_id = __sg.get_local_linear_id();
        std::uint32_t __sg_process_size = __sg_size * __data_per_workitem;
        std::size_t __wg_start = __wg_id * __wg_process_size;
        std::size_t __sg_start = __sg_id * __sg_process_size;
        std::size_t __wg_end = __wg_start + std::min<std::size_t>(__wg_process_size, __n - __wg_start);
        std::uint32_t __adjusted_wg_size = __wg_end - __wg_start;
        // 1. Load
        // TODO: add a specialization for a case __global_value_id < __n condition is true for the whole work-group
        _ONEDPL_PRAGMA_UNROLL
        for (std::int32_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            std::size_t __local_value_id = __sg_offset + __sg_inner_id;
            std::size_t __global_value_id = __wg_start + __local_value_id;
            if (__global_value_id < __n)
            {
                __storage[__local_value_id] = __rng[__global_value_id];
            }
        }
        sycl::group_barrier(__sg);
        // 2. Sort on sub-group level
        // TODO: move border selection inside the sub-group algorithm since it depends on a particular implementation
        std::uint32_t __item_start = __sg_start + __sg_inner_id * __data_per_workitem;
        std::uint32_t __item_end = __item_start + __data_per_workitem;
        __item_start = std::min<std::uint32_t>(__item_start, __adjusted_wg_size);
        __item_end = std::min<std::uint32_t>(__item_end, __adjusted_wg_size);
        __sub_group_sorter.sort(__storage, __comp, __item_start, __item_end);
        __dpl_sycl::__group_barrier(__item);

        // 3. Sort on work-group level
        bool __data_in_temp = __group_sorter.sort(__item, __storage, __comp, static_cast<std::uint32_t>(0),
                                                  __adjusted_wg_size, __data_per_workitem);
        // barrier is not needed here because of the barrier inside the sort method

        // 4. Store
        _ONEDPL_PRAGMA_UNROLL
        for (std::int32_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            std::size_t __local_value_id = __sg_offset + __sg_inner_id;
            std::size_t __global_value_id = __wg_start + __local_value_id;
            if (__global_value_id < __n)
            {
                __rng[__global_value_id] = __storage[__local_value_id + __data_in_temp * __wg_process_size];
            }
        }
    }

    _Range __rng;
    _Compare __comp;
    _Storage __storage;
    _Size __n;
    _SubGroupSorter __sub_group_sorter;
    _GroupSorter __group_sorter;
};

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _IdType, typename _LeafSortName, typename _GlobalSortName, typename _CopyBackName>
struct __parallel_sort_submitter;

template <typename _IdType, typename... _LeafSortName, typename... _GlobalSortName, typename... _CopyBackName>
struct __parallel_sort_submitter<_IdType, __internal::__optional_kernel_name<_LeafSortName...>,
                                 __internal::__optional_kernel_name<_GlobalSortName...>,
                                 __internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
    auto
    operator()(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp) const
    {
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        const std::size_t __n = __rng.size();
        assert(__n > 1);

        // TODO: select __workgroup_size and __data_per_workitem according to the available SLM
        constexpr ::std::uint32_t __workgroup_size = 256;
        constexpr ::std::uint32_t __data_per_workitem = 4;
        constexpr ::std::uint32_t __leaf_size = __workgroup_size * __data_per_workitem;
        std::uint32_t __wg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __leaf_size);

        // 1. Perform sorting of the leaves of the merge sort tree
        sycl::event __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            using _LeafSorter =
                __leaf_sorter<__data_per_workitem, __workgroup_size, std::decay_t<_Range>, std::decay_t<_Compare>>;
            auto __storage = _LeafSorter::storage(__cgh);
            _LeafSorter __sorter(__rng, __comp, __storage, __n);
            const sycl::nd_range<1> __nd_range(sycl::range<1>(__wg_count * __workgroup_size),
                                               sycl::range<1>(__workgroup_size));
            __cgh.parallel_for<_LeafSortName...>(__nd_range, [=](sycl::nd_item<1> __item) { __sorter.sort(__item); });
        });

        // 2. Merge sorting
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _Tp> __temp_buf(__exec, __n);
        auto __temp = __temp_buf.get_buffer();
        bool __data_in_temp = false;
        _IdType __n_sorted = __leaf_size;
        const bool __is_cpu = __exec.queue().get_device().is_cpu();
        const std::uint32_t __chunk = __is_cpu ? 32 : 4;
        const std::uint32_t __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        const std::size_t __n_power2 = oneapi::dpl::__internal::__dpl_bit_ceil(__n);
        const std::int64_t __n_iter = std::log2(__n_power2) - std::log2(__leaf_size);
        for (std::int64_t __i = 0; __i < __n_iter; ++__i)
        {
            __event1 = __exec.queue().submit([&, __n_sorted, __data_in_temp](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);

                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                sycl::accessor __dst(__temp, __cgh, sycl::read_write, sycl::no_init);

                __cgh.parallel_for<_GlobalSortName...>(
                    sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id) {
                        const _IdType __i_elem = __item_id.get_linear_id() * __chunk;
                        const auto __i_elem_local = __i_elem % (__n_sorted * 2);

                        const auto __offset = std::min<_IdType>((__i_elem / (__n_sorted * 2)) * (__n_sorted * 2), __n);
                        const auto __n1 = std::min<_IdType>(__offset + __n_sorted, __n) - __offset;
                        const auto __n2 = std::min<_IdType>(__offset + __n1 + __n_sorted, __n) - (__offset + __n1);

                        if (__data_in_temp)
                        {
                            const auto& __rng1 = oneapi::dpl::__ranges::drop_view_simple(__dst, __offset);
                            const auto& __rng2 = oneapi::dpl::__ranges::drop_view_simple(__dst, __offset + __n1);

                            const auto start = __find_start_point(__rng1, __rng2, __i_elem_local, __n1, __n2, __comp);
                            __serial_merge(__rng1, __rng2, __rng /*__rng3*/, start.first, start.second, __i_elem,
                                           __chunk, __n1, __n2, __comp);
                        }
                        else
                        {
                            const auto& __rng1 = oneapi::dpl::__ranges::drop_view_simple(__rng, __offset);
                            const auto& __rng2 = oneapi::dpl::__ranges::drop_view_simple(__rng, __offset + __n1);

                            const auto start = __find_start_point(__rng1, __rng2, __i_elem_local, __n1, __n2, __comp);
                            __serial_merge(__rng1, __rng2, __dst /*__rng3*/, start.first, start.second, __i_elem,
                                           __chunk, __n1, __n2, __comp);
                        }
                    });
            });
            __n_sorted *= 2;
            __data_in_temp = !__data_in_temp;
        }

        // 3. If the data remained in the temporary buffer then copy it back
        if (__data_in_temp)
        {
            __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
                // We cannot use __cgh.copy here because of zip_iterator usage
                __cgh.parallel_for<_CopyBackName...>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                    const _IdType __idx = __item_id.get_linear_id();
                    __rng[__idx] = __temp_acc[__idx];
                });
            });
        }

        return __future(__event1);
    }
};

template <typename... _Name>
class __sort_leaf_kernel;

template <typename... _Name>
class __sort_global_kernel;

template <typename... _Name>
class __sort_copy_back_kernel;

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
auto
__parallel_sort_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __rng,
                     _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const auto __n = __rng.size();
    if (__n <= std::numeric_limits<std::uint32_t>::max())
    {
        using _wi_index_type = std::uint32_t;
        using _LeafSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sort_leaf_kernel<_CustomName, _wi_index_type>>;
        using _GlobalSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sort_global_kernel<_CustomName, _wi_index_type>>;
        using _CopyBackKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sort_copy_back_kernel<_CustomName, _wi_index_type>>;
        return __parallel_sort_submitter<_wi_index_type, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
            oneapi::dpl::__internal::__device_backend_tag{}, std::forward<_ExecutionPolicy>(__exec),
            std::forward<_Range>(__rng), __comp);
    }
    else
    {
        using _wi_index_type = std::uint64_t;
        using _LeafSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sort_leaf_kernel<_CustomName, _wi_index_type>>;
        using _GlobalSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sort_global_kernel<_CustomName, _wi_index_type>>;
        using _CopyBackKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sort_copy_back_kernel<_CustomName, _wi_index_type>>;
        return __parallel_sort_submitter<_wi_index_type, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
            oneapi::dpl::__internal::__device_backend_tag{}, std::forward<_ExecutionPolicy>(__exec),
            std::forward<_Range>(__rng), __comp);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H
