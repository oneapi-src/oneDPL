// -*- C++ -*-
//===-- sycl_defs.h ---------------------------------------------===//
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

// This file contains SYCL specific macros and abstractions
// to support different versions of SYCL and to simplify its interfaces
//
// Include this header instead of sycl.hpp throughout the project

#ifndef _ONEDPL_sycl_defs_H
#define _ONEDPL_sycl_defs_H

#include <CL/sycl.hpp>
#if _ONEDPL_FPGA_DEVICE
#    include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define __LIBSYCL_VERSION                                                                                          \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define __LIBSYCL_VERSION 0
#endif

// Macros to check the new SYCL features
#define _ONEDPL_NO_INIT_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_KERNEL_BUNDLE_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT (__LIBSYCL_VERSION >= 50300)

namespace __dpl_sycl
{

using __no_init =
#if _ONEDPL_NO_INIT_PRESENT
    sycl::property::no_init;
#else
    sycl::property::noinit;
#endif

template <typename _T>
using __plus =
#if _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT
    sycl::plus<_T>;
#else
    sycl::ONEAPI::plus<_T>;
#endif

template <typename _Buffer>
constexpr auto
__get_buffer_size(const _Buffer& __buffer)
{
#if __LIBSYCL_VERSION >= 50300
    return __buffer.size();
#else
    return __buffer.get_count();
#endif
}

template <typename _Accessor>
constexpr auto
__get_accessor_size(const _Accessor& __accessor)
{
#if __LIBSYCL_VERSION >= 50300
    return __accessor.size();
#else
    return __accessor.get_count();
#endif
}

template <typename _Item>
constexpr void
__group_barrier(_Item __item)
{
#if __LIBSYCL_VERSION >= 50300
    sycl::group_barrier(__item.get_group(), sycl::memory_scope::work_group);
#else
    __item.barrier(sycl::access::fence_space::local_space);
#endif
}

template <typename... _Args>
constexpr auto
__group_broadcast(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::group_broadcast(__args...);
#else
    return sycl::ONEAPI::broadcast(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__exclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::exclusive_scan_over_group(__args...);
#else
    return sycl::ONEAPI::exclusive_scan(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__inclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::inclusive_scan_over_group(__args...);
#else
    return sycl::ONEAPI::inclusive_scan(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__reduce_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::reduce_over_group(__args...);
#else
    return sycl::ONEAPI::reduce(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_exclusive_scan(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_exclusive_scan(__args...);
#else
    return sycl::ONEAPI::exclusive_scan(__args...);
#endif
}

#if _ONEDPL_FPGA_DEVICE
#    if __LIBSYCL_VERSION >= 50400
using fpga_emulator_selector = sycl::ext::intel::fpga_emulator_selector;
using fpga_selector = sycl::ext::intel::fpga_selector;
#    else
using fpga_emulator_selector = sycl::INTEL::fpga_emulator_selector;
using fpga_selector = sycl::INTEL::fpga_selector;
#    endif
#endif // _ONEDPL_FPGA_DEVICE

} // namespace __dpl_sycl

#endif /* _ONEDPL_sycl_defs_H */
