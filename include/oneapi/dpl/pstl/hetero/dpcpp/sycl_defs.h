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

#if !_ONEDPL_BACKEND_SYCL
#    error SYCL backend is not specified.
#endif

#include <CL/sycl.hpp>

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define __LIBSYCL_VERSION                                                                                          \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define __LIBSYCL_VERSION 0
#endif

// Macros to check the new SYCL features
#define _ONEDPL_KERNEL_BUNDLE_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT (__LIBSYCL_VERSION >= 50300)

inline namespace __internal
{
namespace __sycl
{

template <typename _Buffer>
auto
__get_buffer_size(const _Buffer& __buffer)
{
#if __LIBSYCL_VERSION >= 50300
    return __buffer.size();
#else
    return __buffer.get_count();
#endif
}

template <typename _Accessor>
auto
__get_accessor_size(const _Accessor& __accessor)
{
#if __LIBSYCL_VERSION >= 50300
    return __accessor.size();
#else
    return __accessor.get_count();
#endif
}

template <typename _Item>
void
__group_barrier(_Item __item)
{
#if __LIBSYCL_VERSION >= 50300
    sycl::group_barrier(__item.get_group(), sycl::memory_scope::work_group);
#else
    __item.barrier(sycl::access::fence_space::local_space);
#endif
}

template <typename... _Args>
auto
__group_broadcast(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::group_broadcast(__args...);
#else
    return sycl::ONEAPI::broadcast(__args...);
#endif
}

template <typename... _Args>
auto
__exclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::exclusive_scan_over_group(__args...);
#else
    return sycl::ONEAPI::exclusive_scan(__args...);
#endif
}

template <typename... _Args>
auto
__inclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::inclusive_scan_over_group(__args...);
#else
    return sycl::ONEAPI::inclusive_scan(__args...);
#endif
}

template <typename... _Args>
auto
__reduce_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::reduce_over_group(__args...);
#else
    return sycl::ONEAPI::reduce(__args...);
#endif
}

template <typename... _Args>
auto
__joint_exclusive_scan(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_exclusive_scan(__args...);
#else
    return sycl::ONEAPI::exclusive_scan(__args...);
#endif
}

} // namespace __sycl
} // namespace __internal

#endif /* _ONEDPL_sycl_defs_H */
