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
#include <variant>   // std::variant, std::visit
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

struct __subgroup_bubble_sorter
{
    template <typename _Storage, typename _Compare, typename _Size>
    void
    sort(_Storage& __storage, _Compare __comp, _Size __start, _Size __end) const
    {
        for (_Size i = __start; i < __end; ++i)
        {
            for (_Size j = __start + 1; j < __start + __end - i; ++j)
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
            // TODO: copy the data into registers before the merge to halve the required amount of SLM
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

template <std::uint16_t _DataPerWorkitem, std::uint16_t _WorkGroupSize, typename _Range, typename _Compare>
struct __leaf_sorter
{
    static constexpr std::uint16_t __data_per_workitem = _DataPerWorkitem;
    static constexpr std::uint16_t __workgroup_size = _WorkGroupSize;
    static constexpr std::uint32_t __process_size = __data_per_workitem * __workgroup_size;

    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Range>;
    using _Storage = __dpl_sycl::__local_accessor<_Tp>;
    // TODO: select a better sub-group sorter depending on sort stability,
    //       a type (e.g. it can be trivially copied for shuffling within a sub-group)
    using _SubGroupSorter = __subgroup_bubble_sorter;
    using _GroupSorter = __group_merge_path_sorter<__data_per_workitem, __workgroup_size>;

    constexpr static std::uint32_t
    storage_size()
    {
        return 2 * __data_per_workitem * __workgroup_size;
    }

    void
    initialize_storage(sycl::handler& __cgh)
    {
        __storage = _Storage(storage_size(), __cgh);
    }

    __leaf_sorter(_Range& __rng, _Compare __comp)
        : __rng(__rng), __comp(__comp), __n(__rng.size()), __sub_group_sorter(), __group_sorter()
    {
    }

    void
    sort(const sycl::nd_item<1>& __item) const
    {
        sycl::sub_group __sg = __item.get_sub_group();
        sycl::group __wg = __item.get_group();
        const std::uint32_t __wg_id = __wg.get_group_linear_id();
        const std::uint32_t __sg_id = __sg.get_group_linear_id();
        const std::uint32_t __sg_size = __sg.get_local_linear_range();
        const std::uint32_t __sg_inner_id = __sg.get_local_linear_id();
        const std::uint32_t __sg_process_size = __sg_size * __data_per_workitem;
        const std::size_t __wg_start = __wg_id * __process_size;
        const std::uint32_t __sg_start = __sg_id * __sg_process_size;
        const std::size_t __wg_end = __wg_start + std::min<std::size_t>(__process_size, __n - __wg_start);
        const std::uint32_t __adjusted_wg_size = __wg_end - __wg_start;
        // 1. Load
        // TODO: add a specialization for a case __global_value_id < __n condition is true for the whole work-group
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            const std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            const std::uint32_t __local_value_id = __sg_offset + __sg_inner_id;
            const std::size_t __global_value_id = __wg_start + __local_value_id;
            if (__global_value_id < __n)
            {
                __storage[__local_value_id] = std::move(__rng[__global_value_id]);
            }
        }
        sycl::group_barrier(__sg);

        // 2. Sort on sub-group level
        // TODO: move border selection inside the sub-group algorithm since it depends on a particular implementation
        // TODO: set a threshold for bubble sorter (likely 4 items)
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
        for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            const std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            const std::uint32_t __local_value_id = __sg_offset + __sg_inner_id;
            const std::size_t __global_value_id = __wg_start + __local_value_id;
            if (__global_value_id < __n)
            {
                __rng[__global_value_id] = std::move(__storage[__local_value_id + __data_in_temp * __process_size]);
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

template <typename _LeafDPWI, typename _LeafWGS, typename _LeafSortName>
struct __sort_leaf_submitter;

template <typename _LeafDPWI, typename _LeafWGS, typename... _Name>
struct __sort_leaf_submitter<_LeafDPWI, _LeafWGS, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _Compare, typename _LeafSorter>
    sycl::event
    operator()(sycl::queue& __q, _Range& __rng, _Compare __comp, _LeafSorter& __leaf_sorter) const
    {
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        const std::size_t __n = __rng.size();
        assert(__n > 1);

        // 1. Perform sorting of the leaves of the merge sort tree
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __leaf_sorter.initialize_storage(__cgh);
            const std::uint32_t __wg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __leaf_sorter.__process_size);
            const sycl::nd_range<1> __nd_range(sycl::range<1>(__wg_count * __leaf_sorter.__workgroup_size),
                                               sycl::range<1>(__leaf_sorter.__workgroup_size));
            __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __item) {
                __leaf_sorter.sort(__item);
            });
        });
    }
};

template <typename _IdType, typename _GlobalSortName>
struct __sort_global_submitter;

template <typename _IdType, typename... _Name>
struct __sort_global_submitter<_IdType, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _Compare, typename _TempRange>
    sycl::event
    operator()(sycl::queue& __q, _Range& __rng, _Compare __comp, std::uint32_t __leaf,
               _TempRange& __temp, bool& __data_in_temp,  sycl::event& __e) const
    {
        _IdType __n_sorted = __leaf;
        _IdType __n = __rng.size();
        const bool __is_cpu = __q.get_device().is_cpu();
        const std::uint32_t __chunk = __is_cpu ? 32 : 4;
        const std::uint32_t __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        const std::size_t __n_power2 = oneapi::dpl::__internal::__dpl_bit_ceil(__n);
        const std::int64_t __n_iter = std::log2(__n_power2) - std::log2(__leaf);
        for (std::int64_t __i = 0; __i < __n_iter; ++__i)
        {
            __e = __q.submit([&, __n_sorted, __data_in_temp](sycl::handler& __cgh) {
                __cgh.depends_on(__e);

                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                sycl::accessor __dst(__temp, __cgh, sycl::read_write, sycl::no_init);

                __cgh.parallel_for<_Name...>(
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
        return __e;
    }
};

template <typename _CopyBackName>
struct __sort_copy_back_submitter;

template <typename... _Name>
struct __sort_copy_back_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _TempRange>
    sycl::event
    operator()(sycl::queue& __q, _Range& __rng, _TempRange& __temp, bool __data_in_temp, sycl::event& __e) const
    {
        if (__data_in_temp)
        {
            return __q.submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__e);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
                // We cannot use __cgh.copy here because of zip_iterator usage
                __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__rng.size()), [=](sycl::item</*dim=*/1> __item_id) {
                    const auto __idx = __item_id.get_linear_id();
                    __rng[__idx] = __temp_acc[__idx];
                });
            });
        }
        else
        {
            return __e;
        }
    }
};

template <typename _Range, typename _Compare>
struct __leaf_sorter_selector
{
    // 1024 is greater than or equal the maximum work-group size for the majority of devices
    // 8 is the maximum reasonable value for bubble sub-group sorter
    using _LeafXL = __leaf_sorter<8, 1024, _Range, _Compare>;
    using _LeafL = __leaf_sorter<4, 512, _Range, _Compare>;
    using _LeafM = __leaf_sorter<2, 256, _Range, _Compare>;
    // 2 is the smallest reasonable value for merge-path group sorter
    // SYCL specification requires that local memory size should be at least 32 KB,
    // what is enough to sort 256 byte elements (32KB / (2 * 64)).
    // It is unlikely that the element to sort is larger than 256 bytes.
    using _LeafS = __leaf_sorter<2, 64, _Range, _Compare>;

    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;

    std::variant<_LeafXL, _LeafL, _LeafM, _LeafS>
    select(const sycl::queue& __q, _Range& __rng, _Compare __comp) const
    {
        const auto __max_wg_size = __q.get_device().template get_info<sycl::info::device::max_work_group_size>();
        auto __max_slm_items = __q.get_device().template get_info<sycl::info::device::local_mem_size>() / sizeof(_Tp);
        // Get the work group size adjusted to the local memory limit.
        // Pessimistically reduce it by 0.8 to take into account memory used by compiled kernel.
        // TODO: find a way to generalize getting of reliable work-group size.
        __max_slm_items /= 0.8;

        if (__max_slm_items > _LeafXL::storage_size() && __max_wg_size >= 1024 && __rng.size() >= (1 << 20))
        {
            return _LeafXL{__rng, __comp};
        }
        else if (__max_slm_items > _LeafL::storage_size() && __max_wg_size >= 512)
        {
            return _LeafL{__rng, __comp};
        }
        else if (__max_slm_items > _LeafM::storage_size() && __max_wg_size >= 256)
        {
            return _LeafM{__rng, __comp};
        }
        else
        {
            return _LeafS{__rng, __comp};
        }
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
    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    const auto __n = __rng.size();

    auto __index_selector = [__n]() -> std::variant<std::uint32_t, std::uint64_t> {
        if (__n <= std::numeric_limits<std::uint32_t>::max())
            return std::uint32_t{};
        else
            return std::uint64_t{};
    };
    auto __index_alternatives = __index_selector();
    auto __leaf_sorter_alternatives = __leaf_sorter_selector<std::decay_t<_Range>, _Compare>().select(
        __exec.queue(), __rng, __comp);

    return std::visit(
        [&](auto& __leaf_sorter, auto __index) {
            using _LeafSorterT = std::decay_t<decltype(__leaf_sorter)>;
            using _IndexT = decltype(__index);
            using _LeafDPWI = std::integral_constant<std::uint16_t, _LeafSorterT::__data_per_workitem>;
            using _LeafWGS = std::integral_constant<std::uint16_t, _LeafSorterT::__workgroup_size>;

            using _LeafSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __sort_leaf_kernel<_CustomName, _LeafDPWI, _LeafWGS>>;
            using _GlobalSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __sort_global_kernel<_CustomName, _IndexT>>;
            using _CopyBackKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
                __sort_copy_back_kernel<_CustomName>>;

            auto&& __q = __exec.queue();
            // 1. Sort the leaves of the merge sort tree within a work-group
            sycl::event __e = __sort_leaf_submitter<_LeafDPWI, _LeafWGS, _LeafSortKernel>()(
                __q, __rng, __comp, __leaf_sorter);

            // 2. Sort the whole range
            bool __data_in_temp = false;
            oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _Tp> __temp_buf(__exec, __n);
            auto __temp = __temp_buf.get_buffer();
            constexpr std::uint32_t __leaf = _LeafSorterT::__process_size;
            __e = __sort_global_submitter<_IndexT, _GlobalSortKernel>()(
                __q, __rng, __comp, __leaf, __temp, __data_in_temp, __e);

            // 3. Copy back the result, if it is in the temporary buffer
            __e = __sort_copy_back_submitter<_CopyBackKernel>()(__q, __rng, __temp, __data_in_temp, __e);
            return __future(__e);
        },
        __leaf_sorter_alternatives, __index_alternatives);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H
