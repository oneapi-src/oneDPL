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

#include <bit>         // std::bit_floor
#include <cmath>       // std::log2
#include <limits>      // std::numeric_limits
#include <cassert>     // assert
#include <utility>     // std::swap
#include <cstdint>     // std::uint32_t, ...
#include <algorithm>   // std::min, std::max_element
#include <type_traits> // std::decay_t, std::integral_constant

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

template <std::uint16_t __data_per_workitem>
struct __group_merge_path_sorter
{
    template <typename _Storage, typename _Compare, typename _Size1, typename _Size2>
    bool
    sort(const sycl::nd_item<1>& __item, _Storage& __storage, _Compare __comp, _Size1 __start, _Size1 __end,
         _Size2 __sorted, std::uint32_t __workgroup_size) const
    {
        const std::uint32_t __sorted_final = __data_per_workitem * __workgroup_size;

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

            const auto& __it_in = __storage.begin() + __data_in_temp * __sorted_final;
            const auto& __it_out = __storage.begin() + (!__data_in_temp) * __sorted_final;
            const auto& __in1 = __it_in + __start1;
            const auto& __in2 = __it_in + __start2;

            const auto __start = __find_start_point(__in1, __in2, __id_local, __n1, __n2, __comp);
            // TODO: copy the data into registers before the merge to halve the required amount of SLM
            __serial_merge(__in1, __in2, __it_out, __start.first, __start.second, __id, __data_per_workitem, __n1, __n2,
                           __comp);
            __dpl_sycl::__group_barrier(__item);

            __sorted = __next_sorted;
            __next_sorted *= 2;
            __data_in_temp = !__data_in_temp;
        }
        return __data_in_temp;
    }
};

template <std::uint16_t _DataPerWorkitem, typename _Range, typename _Compare>
struct __leaf_sorter
{
    static constexpr std::uint16_t __data_per_workitem = _DataPerWorkitem;

    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Range>;
    using _Storage = __dpl_sycl::__local_accessor<_Tp>;
    // TODO: select a better sub-group sorter depending on sort stability,
    //       a type (e.g. it can be trivially copied for shuffling within a sub-group)
    using _SubGroupSorter = __subgroup_bubble_sorter;
    using _GroupSorter = __group_merge_path_sorter<__data_per_workitem>;

    static std::uint32_t
    storage_size(std::uint32_t __future_workgroup_size)
    {
        return 2 * __data_per_workitem * __future_workgroup_size;
    }

    void
    initialize_storage(sycl::handler& __cgh)
    {
        __storage = _Storage(storage_size(__workgroup_size), __cgh);
    }

    __leaf_sorter(_Range& __rng, _Compare __comp, std::uint32_t __workgroup_size)
        : __rng(__rng), __comp(__comp), __n(__rng.size()), __workgroup_size(__workgroup_size),
          __process_size(__data_per_workitem * __workgroup_size), __sub_group_sorter(), __group_sorter()
    {
        assert((__process_size & (__process_size - 1)) == 0 && "Process size must be a power of 2");
    }

    void
    sort(const sycl::nd_item<1>& __item) const
    {
        sycl::sub_group __sg = __item.get_sub_group();
        sycl::group __wg = __item.get_group();
        const std::uint32_t __wg_id = __wg.get_group_linear_id();
        const std::uint32_t __sg_id = __sg.get_group_linear_id();
        const std::uint32_t __sg_size = __sg.get_local_linear_range();
        const std::uint32_t __sg_local_id = __sg.get_local_linear_id();
        const std::uint32_t __sg_process_size = __sg_size * __data_per_workitem;
        const std::size_t __wg_start = __wg_id * __process_size;
        const std::uint32_t __sg_start = __sg_id * __sg_process_size;
        const std::size_t __wg_end = __wg_start + std::min<std::size_t>(__process_size, __n - __wg_start);
        const std::uint32_t __adjusted_process_size = __wg_end - __wg_start;

        // 1. Load
        // TODO: add a specialization for a case __global_value_id < __n condition is true for the whole work-group
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            const std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            const std::uint32_t __local_value_id = __sg_offset + __sg_local_id;
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
        std::uint32_t __item_start = __sg_start + __sg_local_id * __data_per_workitem;
        std::uint32_t __item_end = __item_start + __data_per_workitem;
        __item_start = std::min<std::uint32_t>(__item_start, __adjusted_process_size);
        __item_end = std::min<std::uint32_t>(__item_end, __adjusted_process_size);
        __sub_group_sorter.sort(__storage, __comp, __item_start, __item_end);
        __dpl_sycl::__group_barrier(__item);

        // 3. Sort on work-group level
        bool __data_in_temp = __group_sorter.sort(__item, __storage, __comp, static_cast<std::uint32_t>(0),
                                                  __adjusted_process_size, __data_per_workitem, __workgroup_size);
        // barrier is not needed here because of the barrier inside the sort method

        // 4. Store
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint16_t __i = 0; __i < __data_per_workitem; ++__i)
        {
            const std::uint32_t __sg_offset = __sg_start + __i * __sg_size;
            const std::uint32_t __local_value_id = __sg_offset + __sg_local_id;
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
    std::uint32_t __workgroup_size;
    std::uint32_t __process_size;
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
    template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare, typename _LeafSorter>
    auto
    operator()(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp,
               _LeafSorter& __leaf_sorter) const
    {
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        const std::size_t __n = __rng.size();
        assert(__n > 1);

        const std::uint32_t __leaf = __leaf_sorter.__process_size;
        // 1. Perform sorting of the leaves of the merge sort tree
        sycl::event __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __leaf_sorter.initialize_storage(__cgh);
            const std::uint32_t __wg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __leaf);
            const sycl::nd_range<1> __nd_range(sycl::range<1>(__wg_count * __leaf_sorter.__workgroup_size),
                                               sycl::range<1>(__leaf_sorter.__workgroup_size));
            __cgh.parallel_for<_LeafSortName...>(__nd_range,
                                                 [=](sycl::nd_item<1> __item) { __leaf_sorter.sort(__item); });
        });

        // 2. Merge sorting
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _Tp> __temp_buf(__exec, __n);
        auto __temp = __temp_buf.get_buffer();
        bool __data_in_temp = false;
        _IdType __n_sorted = __leaf;
        const bool __is_cpu = __exec.queue().get_device().is_cpu();
        const std::uint32_t __chunk = __is_cpu ? 32 : 4;
        const std::size_t __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        const std::size_t __n_power2 = oneapi::dpl::__internal::__dpl_bit_ceil(__n);
        const std::int64_t __n_iter = std::log2(__n_power2) - std::log2(__leaf);
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

template <typename _Range, typename _Compare>
struct __leaf_sorter_selector
{
    // 8 is the maximum reasonable value for bubble sub-group sorter due to algorithm complexity
    // TODO: reconsider the value if another algorithm is used,
    //       or an internal cap is set (e.g. sorting 2 sequences of 4/8 items each)
    using _Leaf8 = __leaf_sorter<8, _Range, _Compare>;
    using _Leaf4 = __leaf_sorter<4, _Range, _Compare>;
    // 2 is the smallest reasonable value for merge-path group sorter since it loads 2 values at least
    using _Leaf2 = __leaf_sorter<2, _Range, _Compare>;

    using _Tp = oneapi::dpl::__internal::__value_t<_Range>;

    template <typename _ExecutionPolicy, typename _SubmitterT>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range& __rng, _Compare __comp, const _SubmitterT& __submitter) const
    {
        const std::size_t __n = __rng.size();
        auto __device = __exec.queue().get_device();

        std::size_t __max_wg_size = __device.template get_info<sycl::info::device::max_work_group_size>();
        __max_wg_size = oneapi::dpl::__internal::__dpl_bit_floor(__max_wg_size);

        const auto __sg_sizes = __device.template get_info<sycl::info::device::sub_group_sizes>();
        const auto __max_sg_size = __sg_sizes.empty() ? 1 : *std::max_element(__sg_sizes.begin(), __sg_sizes.end());
        // TODO: reconsider the constant if the corresponding query appears in the SYCL specification
        // 8 (or 6, which is slightly less) appears to be common for modern Intel/AMD/Nvidia GPUs see:
        // Intel: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/intel-xe-gpu-architecture.html:
        //        see: "Threads / XVE" (8)
        // Nvidia: https://xmartlabs.github.io/cuda-calculator (they deprecated their official calculator with no alternative):
        //         see: "Active Thread Blocks per Multiprocessor" (6)
        // AMD: https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html#mi300x-occupancy-vgpr-table
        //         see: "Occupancy per EU" (8)
        const std::uint8_t __oversubscription = __device.is_gpu() ? 8 : 1;
        const auto __max_cu = __device.template get_info<sycl::info::device::max_compute_units>();
        const auto __saturation_point = __max_cu * __max_sg_size * __oversubscription;
        const auto __desired_data_per_workitem = __n / __saturation_point;

        // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
        // TODO: investigate if the adjustment can be less conservative
        const std::size_t __max_slm_items =
            __device.template get_info<sycl::info::device::local_mem_size>() / (sizeof(_Tp) * 2);
        if (__max_slm_items >= _Leaf8::storage_size(__max_wg_size) && __desired_data_per_workitem >= 8)
        {
            _Leaf8 __leaf(__rng, __comp, __max_wg_size);
            return __submitter(__leaf);
        }

        if (__max_slm_items >= _Leaf4::storage_size(__max_wg_size) && __desired_data_per_workitem >= 4)
        {
            _Leaf4 __leaf(__rng, __comp, __max_wg_size);
            return __submitter(__leaf);
        }

        if (__max_slm_items >= _Leaf2::storage_size(__max_wg_size) && __desired_data_per_workitem >= 2)
        {
            _Leaf2 __leaf(__rng, __comp, __max_wg_size);
            return __submitter(__leaf);
        }

        const std::size_t __slm_max_wg_size = __max_slm_items / _Leaf2::storage_size(1);
        // __n is taken as is because of the bit floor and processing 2 items per work-item
        // hence the processed size always fits a single work-group if __n is chosen
        __max_wg_size = std::min<std::size_t>({__max_wg_size, __slm_max_wg_size, __n});
        __max_wg_size = oneapi::dpl::__internal::__dpl_bit_floor(__max_wg_size);

        _Leaf2 __leaf(__rng, __comp, __max_wg_size);
        return __submitter(__leaf);
    }
};

template <typename... _Name>
class __sort_leaf_kernel;

template <typename... _Name>
class __sort_global_kernel;

template <typename... _Name>
class __sort_copy_back_kernel;

template <typename _ExecutionPolicy, typename _Range, typename _Compare, typename _IndexT>
struct __parallel_sort_impl_submitter
{
    _ExecutionPolicy __exec;
    _Range __rng;
    _Compare __comp;

    template <typename _LeafSorter>
    auto
    operator()(_LeafSorter& __leaf_sorter) const
    {
        using _LeafSorterT = std::decay_t<decltype(__leaf_sorter)>;
        using _LeafDPWI    = std::integral_constant<std::uint16_t, _LeafSorterT::__data_per_workitem>;

        using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

        // TODO: split the submitter into multiple ones to avoid extra compilation of kernels
        // - _LeafSortKernel does not need _IndexT
        // - _GlobalSortKernel does not need _LeafDPWI
        // - _CopyBackKernel does not need either of them
        using _LeafSortKernel   = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel     <_CustomName, _IndexT, _LeafDPWI>>;
        using _GlobalSortKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel   <_CustomName, _IndexT, _LeafDPWI>>;
        using _CopyBackKernel   = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName, _IndexT, _LeafDPWI>>;

        return __parallel_sort_submitter<_IndexT, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>{}(
            oneapi::dpl::__internal::__device_backend_tag{}, __exec, __rng, __comp, __leaf_sorter);
    }
};

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
auto
__parallel_sort_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __rng,
                     _Compare __comp)
{
    const auto __n = __rng.size();

    if (__n <= std::numeric_limits<std::uint32_t>::max())
    {
        __parallel_sort_impl_submitter<_ExecutionPolicy, _Range, _Compare, std::uint32_t> __submitter{__exec, __rng, __comp};
        return __leaf_sorter_selector<_Range, _Compare>{}(__exec, __rng, __comp, __submitter);
    }

    __parallel_sort_impl_submitter<_ExecutionPolicy, _Range, _Compare, std::uint64_t> __submitter{__exec, __rng, __comp};
    return __leaf_sorter_selector<_Range, _Compare>{}(__exec, __rng, __comp, __submitter);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_MERGE_SORT_H
