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

//------------------------------------------------------------------------
// parallel_transform_reduce - async pattern
//------------------------------------------------------------------------

template <typename _Tp, typename _ExecutionPolicy, typename _Up, typename _Cp, typename _Rp, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__future<_Tp>>
__parallel_transform_reduce_async(_ExecutionPolicy&& __exec, _Up __u, _Cp __combine, _Rp __brick_reduce,
                                  _Ranges&&... __rngs)
{
    auto __n = __get_first_range(__rngs...).size();
    assert(__n > 0);

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_reduce_kernel<_Up, _Cp, _Rp, __kernel_name, _Ranges...>;
#else
    using __kernel_name_t = __parallel_reduce_kernel<__kernel_name>;
#endif

    auto __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
    // change __wgroup_size according to local memory limit
    __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size<_ExecutionPolicy, _Tp>(
        ::std::forward<_ExecutionPolicy>(__exec), __wgroup_size);
#if _ONEDPL_COMPILE_KERNEL
    auto __kernel = __kernel_name_t::__compile_kernel(::std::forward<_ExecutionPolicy>(__exec));
    __wgroup_size = ::std::min(__wgroup_size, oneapi::dpl::__internal::__kernel_work_group_size(
                                                  ::std::forward<_ExecutionPolicy>(__exec), __kernel));
#endif
    auto __mcu = oneapi::dpl::__internal::__max_compute_units(::std::forward<_ExecutionPolicy>(__exec));
    auto __n_groups = (__n - 1) / __wgroup_size + 1;
    __n_groups = ::std::min(decltype(__n_groups)(__mcu), __n_groups);
    // TODO: try to change __n_groups with another formula for more perfect load balancing

    _PRINT_INFO_IN_DEBUG_MODE(__exec, __wgroup_size, __mcu);

    // Create temporary global buffers to store temporary values
    auto __temp_1 = sycl::buffer<_Tp>(sycl::range<1>(__n_groups));
    auto __temp_2 = sycl::buffer<_Tp>(sycl::range<1>(__n_groups));
    // __is_first == true. Reduce over each work_group
    // __is_first == false. Reduce between work groups
    bool __is_first = true;
    auto __buf_1_ptr = &__temp_1; // __buf_1_ptr is not accessed on the device when __is_first == true
    auto __buf_2_ptr = &__temp_1;
    sycl::event __reduce_event;
    do
    {
        __reduce_event = __exec.queue().submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);

            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...); // get an access to data under SYCL buffer
            auto __temp_1_acc = __buf_1_ptr->template get_access<access_mode::read_write>(__cgh);
            auto __temp_2_acc = __buf_2_ptr->template get_access<access_mode::write>(__cgh);
            sycl::accessor<_Tp, 1, access_mode::read_write, sycl::access::target::local> __temp_local(
                sycl::range<1>(__wgroup_size), __cgh);
            __cgh.parallel_for<__kernel_name_t>(
#if _ONEDPL_COMPILE_KERNEL
                __kernel,
#endif
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __wgroup_size), sycl::range<1>(__wgroup_size)),
                [=](sycl::nd_item<1> __item_id) {
                    auto __global_idx = __item_id.get_global_id(0);
                    auto __local_idx = __item_id.get_local_id(0);
                    // 1. Initialization (transform part). Fill local memory
                    if (__is_first)
                    {
                        __u(__item_id, __global_idx, __n, __temp_local, __rngs...);
                    }
                    else
                    {
                        if (__global_idx < __n)
                            __temp_local[__local_idx] = __temp_1_acc[__global_idx];
                        __item_id.barrier(sycl::access::fence_space::local_space);
                    }
                    // 2. Reduce within work group using local memory
                    auto __res = __brick_reduce(__item_id, __global_idx, __n, __temp_local);
                    if (__local_idx == 0)
                    {
                        __temp_2_acc[__item_id.get_group(0)] = __res;
                    }
                });
        });
        if (__is_first)
        {
            __buf_2_ptr = &__temp_2;
            __is_first = false;
        }
        else
        {
            ::std::swap(__buf_1_ptr, __buf_2_ptr);
        }
        __n = __n_groups;
        __n_groups = (__n - 1) / __wgroup_size + 1;
    } while (__n > 1);

    return ::oneapi::dpl::__internal::__future<_Tp>(__reduce_event, *__buf_1_ptr, *__buf_2_ptr);
}

} // namespace __par_backend_hetero

} // namespace dpl

} // namespace oneapi

#endif /* _ONEDPL_async_backend_sycl_H */
