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

#ifndef _ONEDPL_SYCL_DEFS_H
#define _ONEDPL_SYCL_DEFS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

// If SYCL_LANGUAGE_VERSION is still not set after including the SYCL header, issue an error
#if !(SYCL_LANGUAGE_VERSION || CL_SYCL_LANGUAGE_VERSION)
#    error "Device execution policies are enabled, \
        but SYCL_LANGUAGE_VERSION/CL_SYCL_LANGUAGE_VERSION macros are not defined"
#endif

#include <memory>

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define _ONEDPL_LIBSYCL_VERSION                                                                                    \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#endif
#define _ONEDPL_LIBSYCL_VERSION_LESS_THAN(_VERSION) (_ONEDPL_LIBSYCL_VERSION && _ONEDPL_LIBSYCL_VERSION < _VERSION)

#if _ONEDPL_FPGA_DEVICE
#    if _ONEDPL_LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif

// SYCL 2020 feature macros. They are enabled by default:
// A SYCL implementation is assumed to support the features unless specified otherwise.
// This is controlled by extendable logic: !(A && A < SUPPORTED_VER_A) && !(B && B < SUPPORTED_VER_B) && ...,
// where A, B, etc., are macros representing the version of a specific SYCL implementation.
#define _ONEDPL_SYCL2020_BITCAST_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_NO_INIT_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_BUFFER_SIZE_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_ACCESSOR_SIZE_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_REQD_SUB_GROUP_SIZE_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_TARGET_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400))
#define _ONEDPL_SYCL2020_TARGET_DEVICE_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400))
#define _ONEDPL_SYCL2020_ATOMIC_REF_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50500))
#define _ONEDPL_SYCL2020_SUB_GROUP_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700))
#define _ONEDPL_SYCL2020_SUBGROUP_BARRIER_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700))
// 20201214 value corresponds to DPC++ 2021.1.2
#define _ONEDPL_SYCL2020_KERNEL_DEVICE_API_PRESENT                                                                     \
    (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700) || __SYCL_COMPILER_VERSION > 20201214)
#define _ONEDPL_SYCL2020_BUFFER_ALLOCATOR_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000))
#define _ONEDPL_SYCL2020_LOCAL_ACCESSOR_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000))
// The unified future supporting USM memory and buffers is only supported after DPC++ 2023.1 but not by 2023.2.
#define _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT                                                          \
    (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60100) || _ONEDPL_LIBSYCL_VERSION != 60200)
#define _ONEDPL_SYCL2020_HOST_TARGET_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200))
#define _ONEDPL_SYCL2020_HOST_ACCESSOR_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200))
#define _ONEDPL_SYCL2020_GET_HOST_ACCESS_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200))
#define _ONEDPL_SYCL2020_LOCAL_ACC_GET_MULTI_PTR_PRESENT (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(70000))

// Feature macros for DPC++ SYCL runtime library alternatives to non-supported SYCL 2020 features
#define _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT (_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_LIBSYCL_KNOWN_IDENTITY_PRESENT (_ONEDPL_LIBSYCL_VERSION == 50200)
#define _ONEDPL_LIBSYCL_SUB_GROUP_MASK_PRESENT                                                                         \
    (SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 1 && _ONEDPL_LIBSYCL_VERSION >= 50700)

#define _ONEDPL_SYCL_DEVICE_COPYABLE_SPECIALIZATION_BROKEN (_ONEDPL_LIBSYCL_VERSION_LESS_THAN(70100))

// TODO: determine which compiler configurations provide subgroup load/store
#define _ONEDPL_SYCL_SUB_GROUP_LOAD_STORE_PRESENT false

// Macro to check if we are compiling for SPIR-V devices. This macro must only be used within
// SYCL kernels for determining SPIR-V compilation. Using this macro on the host may lead to incorrect behavior.
#ifndef _ONEDPL_DETECT_SPIRV_COMPILATION // Check if overridden for testing
#    if (defined(__SPIR__) || defined(__SPIRV__)) && defined(__SYCL_DEVICE_ONLY__)
#        define _ONEDPL_DETECT_SPIRV_COMPILATION 1
#    else
#        define _ONEDPL_DETECT_SPIRV_COMPILATION 0
#    endif
#endif // _ONEDPL_DETECT_SPIRV_COMPILATION

#if _ONEDPL_SYCL2020_REQD_SUB_GROUP_SIZE_PRESENT
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE) sycl::reqd_sub_group_size(SIZE)
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE) intel::reqd_sub_group_size(SIZE)
#endif

// This macro is intended to be used for specifying a subgroup size as a SYCL kernel attribute for SPIR-V targets
// only. For non-SPIR-V targets, it will be empty. This macro should only be used in device code and may lead
// to incorrect behavior if used on the host.
#if _ONEDPL_DETECT_SPIRV_COMPILATION
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(SIZE) _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE)
#else
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(SIZE)
#endif

namespace __dpl_sycl
{

using __no_init =
#if _ONEDPL_SYCL2020_NO_INIT_PRESENT
    sycl::property::no_init;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
    sycl::property::noinit;
#endif

#if _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::has_known_identity<_BinaryOp, _T>;

#elif _ONEDPL_LIBSYCL_KNOWN_IDENTITY_PRESENT
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::ONEAPI::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::ONEAPI::has_known_identity<_BinaryOp, _T>;
#endif // _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT

template <typename _BinaryOp, typename _T>
inline constexpr auto __known_identity_v = __known_identity<_BinaryOp, _T>::value;

#if _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT
template <typename _T = void>
using __plus = sycl::plus<_T>;

template <typename _T = void>
using __maximum = sycl::maximum<_T>;

template <typename _T = void>
using __minimum = sycl::minimum<_T>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
template <typename _T>
using __plus = sycl::ONEAPI::plus<_T>;

template <typename _T>
using __maximum = sycl::ONEAPI::maximum<_T>;

template <typename _T>
using __minimum = sycl::ONEAPI::minimum<_T>;
#endif // _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT

#if _ONEDPL_SYCL2020_SUB_GROUP_PRESENT
using __sub_group = sycl::sub_group;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700)
using __sub_group = sycl::ONEAPI::sub_group;
#endif

template <typename _Buffer>
constexpr auto
__get_buffer_size(const _Buffer& __buffer)
{
#if _ONEDPL_SYCL2020_BUFFER_SIZE_PRESENT
    return __buffer.size();
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
    return __buffer.get_count();
#endif
}

template <typename _Accessor>
constexpr auto
__get_accessor_size(const _Accessor& __accessor)
{
#if _ONEDPL_SYCL2020_ACCESSOR_SIZE_PRESENT
    return __accessor.size();
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
    return __accessor.get_count();
#endif
}

template <typename _Item>
constexpr void
__group_barrier(_Item __item)
{
#if 0 // !defined(_ONEDPL_LIBSYCL_VERSION) || _ONEDPL_LIBSYCL_VERSION >= 50300
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
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::broadcast(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__exclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::exclusive_scan_over_group(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::exclusive_scan(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__inclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::inclusive_scan_over_group(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::inclusive_scan(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__reduce_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::reduce_over_group(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::reduce(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__any_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::any_of_group(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::any_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__all_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::all_of_group(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::all_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__none_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::none_of_group(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::none_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_exclusive_scan(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_exclusive_scan(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::exclusive_scan(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_inclusive_scan(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_inclusive_scan(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::inclusive_scan(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_reduce(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_reduce(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::reduce(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_any_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_any_of(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::any_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_all_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_all_of(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::all_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_none_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_none_of(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::none_of(::std::forward<_Args>(__args)...);
#endif
}

#if _ONEDPL_FPGA_DEVICE
#    if _ONEDPL_LIBSYCL_VERSION >= 60100
inline auto __fpga_emulator_selector()
{
    return sycl::ext::intel::fpga_emulator_selector_v;
}
inline auto __fpga_selector()
{
    return sycl::ext::intel::fpga_selector_v;
}

#    elif _ONEDPL_LIBSYCL_VERSION >= 50300
inline auto __fpga_emulator_selector()
{
    return sycl::ext::intel::fpga_emulator_selector{};
}
inline auto __fpga_selector()
{
    return sycl::ext::intel::fpga_selector{};
}
#    else
inline auto __fpga_emulator_selector()
{
    return sycl::INTEL::fpga_emulator_selector{};
}
inline auto __fpga_selector()
{
    return sycl::INTEL::fpga_selector{};
}
#    endif
#endif // _ONEDPL_FPGA_DEVICE

using __target =
#if _ONEDPL_SYCL2020_TARGET_PRESENT
    sycl::target;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400)
    sycl::access::target;
#endif

constexpr __target __target_device =
#if _ONEDPL_SYCL2020_TARGET_DEVICE_PRESENT
    __target::device;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400)
    __target::global_buffer;
#endif

constexpr __target __host_target =
#if _ONEDPL_SYCL2020_HOST_TARGET_PRESENT
    __target::host_task;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200)
    __target::host_buffer;
#endif

template <typename _DataT>
using __buffer_allocator =
#if _ONEDPL_SYCL2020_BUFFER_ALLOCATOR_PRESENT
    sycl::buffer_allocator<_DataT>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000)
    sycl::buffer_allocator;
#endif

template <typename _AtomicType, sycl::access::address_space _Space>
#if _ONEDPL_SYCL2020_ATOMIC_REF_PRESENT
using __atomic_ref = sycl::atomic_ref<_AtomicType, sycl::memory_order::relaxed, sycl::memory_scope::work_group, _Space>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50500)
struct __atomic_ref : sycl::atomic<_AtomicType, _Space>
{
    explicit __atomic_ref(_AtomicType& ref)
        : sycl::atomic<_AtomicType, _Space>(sycl::multi_ptr<_AtomicType, _Space>(&ref)){};
};
#endif // _ONEDPL_SYCL2020_ATOMIC_REF_PRESENT

template <typename _DataT, int _Dimensions = 1>
using __local_accessor =
#if _ONEDPL_SYCL2020_LOCAL_ACCESSOR_PRESENT
    sycl::local_accessor<_DataT, _Dimensions>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000)
    sycl::accessor<_DataT, _Dimensions, sycl::access::mode::read_write, __dpl_sycl::__target::local>;
#endif

template <typename _Buf>
auto
__get_host_access(_Buf&& __buf)
{
#if _ONEDPL_SYCL2020_GET_HOST_ACCESS_PRESENT
    return ::std::forward<_Buf>(__buf).get_host_access(sycl::read_only);
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200)
    return ::std::forward<_Buf>(__buf).template get_access<sycl::access::mode::read>();
#endif
}

template <typename _Acc>
auto
__get_accessor_ptr(const _Acc& __acc)
{
#if _ONEDPL_SYCL2020_LOCAL_ACC_GET_MULTI_PTR_PRESENT
    return __acc.template get_multi_ptr<sycl::access::decorated::no>().get();
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(70000)
    return __acc.get_pointer();
#endif
}

#if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO) || defined(SYCL_EXT_ACPP_BACKEND_LEVEL_ZERO)
#    define _ONEDPL_SYCL_L0_EXT_PRESENT 1
#endif
#if _ONEDPL_SYCL_L0_EXT_PRESENT
#    if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
inline constexpr auto __level_zero_backend = sycl::backend::ext_oneapi_level_zero;
#    elif defined(SYCL_EXT_ACPP_BACKEND_LEVEL_ZERO)
inline constexpr auto __level_zero_backend = sycl::backend::level_zero;
#    endif
#endif

} // namespace __dpl_sycl

#endif // _ONEDPL_SYCL_DEFS_H
