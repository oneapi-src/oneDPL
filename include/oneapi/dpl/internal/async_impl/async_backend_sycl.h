// -*- C++ -*-
//===-- async_backend_sycl.h ----------------------------------------------===//
//
// Copyright (C) 2019-2021 Intel Corporation
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

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL
#ifndef _ONEDPL_async_backend_sycl_H
#define _ONEDPL_async_backend_sycl_H

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// TODO: Merge experimental async pattern into dpcpp backend
//------------------------------------------------------------------------
// parallel_transform_reduce - async pattern
//------------------------------------------------------------------------

template <typename _Tp, ::std::size_t __grainsize = 4, typename _ExecutionPolicy, typename _Up, typename _Cp,
          typename _Rp, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
__parallel_transform_reduce_async(_ExecutionPolicy&& __exec, _Up __u, _Cp __combine, _Rp __brick_reduce,
                                  _Ranges&&... __rngs)
{
    auto __n = __get_first_range(__rngs...).size();
    assert(__n > 0);

    using _Size = decltype(__n);
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_reduce_kernel<_Up, _Cp, _Rp, __kernel_name, _Ranges...>;
#else
    using __kernel_name_t = __parallel_reduce_kernel<__kernel_name>;
#endif

    sycl::cl_uint __max_compute_units = oneapi::dpl::__internal::__max_compute_units(__exec);
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    // change __work_group_size according to local memory limit
    __work_group_size = oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, _Tp>(
        ::std::forward<_ExecutionPolicy>(__exec), __work_group_size);
#if _ONEDPL_COMPILE_KERNEL
    sycl::kernel __kernel = __kernel_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    __work_group_size = ::std::min(__work_group_size, oneapi::dpl::__internal::__kernel_work_group_size(
                                                          ::std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif
    ::std::size_t __iters_per_work_item = __grainsize;
    // distribution is ~1 work groups per compute init
    if (__exec.queue().get_device().is_cpu())
        __iters_per_work_item = (__n - 1) / (__max_compute_units * __work_group_size) + 1;
    ::std::size_t __size_per_work_group =
        __iters_per_work_item * __work_group_size;            // number of buffer elements processed within workgroup
    _Size __n_groups = (__n - 1) / __size_per_work_group + 1; // number of work groups
    _Size __n_items = (__n - 1) / __iters_per_work_item + 1;  // number of work items

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __work_group_size, __max_compute_units);

    // Create temporary global buffers to store temporary values
    sycl::buffer<_Tp> __temp(sycl::range<1>(2 * __n_groups));
    // __is_first == true. Reduce over each work_group
    // __is_first == false. Reduce between work groups
    bool __is_first = true;

    // For memory utilization it's better to use one big buffer instead of two small because size of the buffer is close to a few MB
    _Size __offset_1 = 0;
    _Size __offset_2 = __n_groups;

    sycl::event __reduce_event;
    do
    {
        __reduce_event = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); //get an access to data under SYCL buffer
            auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
            sycl::accessor<_Tp, 1, access_mode::read_write, sycl::access::target::local> __temp_local(
                sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<__kernel_name_t>(
#if _ONEDPL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    ::std::size_t __global_idx = __item_id.get_global_id(0);
                    ::std::size_t __local_idx = __item_id.get_local_id(0);
                    // 1. Initialization (transform part). Fill local memory
                    if (__is_first)
                    {
                        __u(__item_id, __n, __iters_per_work_item, __global_idx, __temp_local, __rngs...);
                    }
                    else
                    {
                        // TODO: check the approach when we use grainsize here too
                        if (__global_idx < __n_items)
                            __temp_local[__local_idx] = __temp_acc[__offset_2 + __global_idx];
                        __item_id.barrier(sycl::access::fence_space::local_space);
                    }
                    // 2. Reduce within work group using local memory
                    _Tp __result = __brick_reduce(__item_id, __global_idx, __n_items, __temp_local);
                    if (__local_idx == 0)
                    {
                        __temp_acc[__offset_1 + __item_id.get_group(0)] = __result;
                    }
                });
        });
        if (__is_first)
        {
            __is_first = false;
        }
        ::std::swap(__offset_1, __offset_2);
        __n_items = __n_groups;
        __n_groups = (__n_items - 1) / __work_group_size + 1;
    } while (__n_items > 1);
    //return future to postpone implicit synchronization point accessing return value
    return ::oneapi::dpl::__internal::__future<_Tp>(__reduce_event, __temp, __combine, __offset_2);
}

} // namespace __par_backend_hetero

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_async_backend_sycl_H */
