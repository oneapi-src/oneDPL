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

#include <cmath> // std::log2
#include <limits> // std::numeric_limits
#include <cassert> // assert
#include <utility> // std::swap
#include <cstdint> // std::uint32_t, ...
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

struct __leaf_sort_kernel
{
    template <typename _Acc, typename _Size1, typename _Compare>
    void
    operator()(const _Acc& __acc, const _Size1 __start, const _Size1 __end, _Compare __comp) const
    {
        for (_Size1 i = __start; i < __end; ++i)
        {
            for (_Size1 j = __start + 1; j < __start + __end - i; ++j)
            {
                // forwarding references allow binding of internal tuple of references with rvalue
                auto&& __first_item = __acc[j - 1];
                auto&& __second_item = __acc[j];
                if (__comp(__second_item, __first_item))
                {
                    using std::swap;
                    swap(__first_item, __second_item);
                }
            }
        }
    }
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

        const bool __is_cpu = __exec.queue().get_device().is_cpu();
        const std::uint32_t __leaf = __is_cpu ? 16 : 4;
        _Size __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __leaf);

        // 1. Perform sorting of the leaves of the merge sort tree
        sycl::event __event1 = __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __cgh.parallel_for<_LeafSortName...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id)
            {
                const _IdType __i_elem = __item_id.get_linear_id() * __leaf;
                __leaf_sort_kernel()(__rng, __i_elem, std::min<_IdType>(__i_elem + __leaf, __n), __comp);
            });
        });

        // 2. Merge sorting
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _Tp> __temp_buf(__exec, __n);
        auto __temp = __temp_buf.get_buffer();
        bool __data_in_temp = false;
        _IdType __n_sorted = __leaf;
        const std::uint32_t __chunk = __is_cpu ? 32 : 4;
        __steps = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk);

        const std::size_t __n_power2 = oneapi::dpl::__internal::__dpl_bit_ceil(__n);
        const std::int64_t __n_iter = std::log2(__n_power2) - std::log2(__leaf);
        for (std::int64_t __i = 0; __i < __n_iter; ++__i)
        {
            __event1 = __exec.queue().submit([&, __n_sorted, __data_in_temp](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);

                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                sycl::accessor __dst(__temp, __cgh, sycl::read_write, sycl::no_init);

                __cgh.parallel_for<_GlobalSortName...>(sycl::range</*dim=*/1>(__steps), [=](sycl::item</*dim=*/1> __item_id)
                    {
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
                            __serial_merge(__rng1, __rng2, __rng/*__rng3*/, start.first, start.second, __i_elem, __chunk, __n1, __n2, __comp);
                        }
                        else
                        {
                            const auto& __rng1 = oneapi::dpl::__ranges::drop_view_simple(__rng, __offset);
                            const auto& __rng2 = oneapi::dpl::__ranges::drop_view_simple(__rng, __offset + __n1);

                            const auto start = __find_start_point(__rng1, __rng2, __i_elem_local, __n1, __n2, __comp);
                            __serial_merge(__rng1, __rng2, __dst/*__rng3*/, start.first, start.second, __i_elem, __chunk, __n1, __n2, __comp);
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
        using _LeafSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel<_CustomName, _wi_index_type>>;
        using _GlobalSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName, _wi_index_type>>;
        using _CopyBackKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName, _wi_index_type>>;
        return __parallel_sort_submitter<_wi_index_type, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
            oneapi::dpl::__internal::__device_backend_tag{}, std::forward<_ExecutionPolicy>(__exec),
            std::forward<_Range>(__rng), __comp);
    }
    else
    {
        using _wi_index_type = std::uint64_t;
        using _LeafSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_leaf_kernel<_CustomName, _wi_index_type>>;
        using _GlobalSortKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName, _wi_index_type>>;
        using _CopyBackKernel =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName, _wi_index_type>>;
        return __parallel_sort_submitter<_wi_index_type, _LeafSortKernel, _GlobalSortKernel, _CopyBackKernel>()(
            oneapi::dpl::__internal::__device_backend_tag{}, std::forward<_ExecutionPolicy>(__exec),
            std::forward<_Range>(__rng), __comp);
    }
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_H
