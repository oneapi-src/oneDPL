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

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define __LIBSYCL_VERSION                                                                                          \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define __LIBSYCL_VERSION 0
#endif

#if _ONEDPL_FPGA_DEVICE
#    if __LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif

// Macros to check the new SYCL features
#define _ONEDPL_NO_INIT_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_KERNEL_BUNDLE_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT (__LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2023_ATOMIC_REF_PRESENT (__LIBSYCL_VERSION >= 50500)

namespace __dpl_sycl
{

using __no_init =
#if _ONEDPL_NO_INIT_PRESENT
    sycl::property::no_init;
#else
    sycl::property::noinit;
#endif

#if _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::has_known_identity<_BinaryOp, _T>;

#elif __LIBSYCL_VERSION == 50200
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::ONEAPI::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::ONEAPI::has_known_identity<_BinaryOp, _T>;
#endif // _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT

#if _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT
template <typename _T>
using __plus = sycl::plus<_T>;

template <typename _T>
using __maximum = sycl::maximum<_T>;

template <typename _T>
using __minimum = sycl::minimum<_T>;

#else  // _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT
template <typename _T>
using __plus = sycl::ONEAPI::plus<_T>;

template <typename _T>
using __maximum = sycl::ONEAPI::maximum<_T>;

template <typename _T>
using __minimum = sycl::ONEAPI::minimum<_T>;
#endif // _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT

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
#if 0 //__LIBSYCL_VERSION >= 50300
    //TODO: usage of sycl::group_barrier: probably, we have to revise SYCL parallel patterns which use a group_barrier.
    // 1) sycl::group_barrier() implementation is not ready
    // 2) sycl::group_barrier and sycl::item::group_barrier are not quite equivalent
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
#    if __LIBSYCL_VERSION >= 50300
using __fpga_emulator_selector = sycl::ext::intel::fpga_emulator_selector;
using __fpga_selector = sycl::ext::intel::fpga_selector;
#    else
using __fpga_emulator_selector = sycl::INTEL::fpga_emulator_selector;
using __fpga_selector = sycl::INTEL::fpga_selector;
#    endif
#endif // _ONEDPL_FPGA_DEVICE

using __target =
#if __LIBSYCL_VERSION >= 50400
    sycl::target;
#else
    sycl::access::target;
#endif

constexpr __target __target_device =
#if __LIBSYCL_VERSION >= 50400
    __target::device;
#else
    __target::global_buffer;
#endif

template <typename _DataT>
using __buffer_allocator =
#if __LIBSYCL_VERSION >= 50707
    sycl::buffer_allocator<_DataT>;
#else
    sycl::buffer_allocator;
#endif

template <typename _AtomicType, sycl::access::address_space _Space>
struct __Atomic :
#if _ONEDPL_SYCL2023_ATOMIC_REF_PRESENT
    sycl::atomic<_AtomicType, _Space>
{
    template <typename __Accessor>
    __Atomic(__Accessor _acc, ::std::size_t _offset) : sycl::atomic<_AtomicType, _Space>(_acc.get_pointer() + _offset)
    {
    }
};
#else
    sycl::atomic_ref<_AtomicType, sycl::memory_order::relaxed, sycl::memory_scope::work_group, _Space>
{
    template <typename __Accessor>
    __Atomic(__Accessor _acc, ::std::size_t _offset)
        : sycl::atomic_ref<_AtomicType, sycl::memory_order::relaxed, sycl::memory_scope::work_group, _Space>(
              _acc[_offset])
    {
    }
};
#endif // _ONEDPL_SYCL2023_ATOMIC_REF_PRESENT

} // namespace __dpl_sycl

#endif /* _ONEDPL_sycl_defs_H */
