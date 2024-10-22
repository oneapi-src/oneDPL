// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

// This header checks availability of heterogeneous backends, and configures them if available

#ifndef _ONEDPL_HETERO_BACKEND_CONFIG
#define _ONEDPL_HETERO_BACKEND_CONFIG

// Detect both Intel(R) oneAPI DPC++/C++ Compiler and oneAPI DPC++ compiler
// Rely on an extension attribute, which is present in both compilers
// A predefined macro cannot be used since oneAPI DPC++/C++ Compiler provides the same set of macros as Clang
#if __has_cpp_attribute(intel::kernel_args_restrict)
#    define _ONEDPL_DPCPP_COMPILER 1
#else
#    define _ONEDPL_DPCPP_COMPILER 0
#endif

// --------------------------------------------------------------------------------------------------------------------
// Enablement of heterogeneous backends
// --------------------------------------------------------------------------------------------------------------------

// Preliminary check SYCL availability
#define _ONEDPL_SYCL_HEADER_PRESENT (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))
#define _ONEDPL_SYCL_LANGUAGE_VERSION_PRESENT (SYCL_LANGUAGE_VERSION || CL_SYCL_LANGUAGE_VERSION)
#if _ONEDPL_SYCL_HEADER_PRESENT
#    if _ONEDPL_SYCL_LANGUAGE_VERSION_PRESENT
#        define _ONEDPL_SYCL_AVAILABLE 1
// DPC++/C++ Compilers pre-define SYCL_LANGUAGE_VERSION with -fsycl option
#    elif !_ONEDPL_DPCPP_COMPILER
// Other implementations might define the macro in the SYCL header
#        define _ONEDPL_SYCL_POSSIBLY_AVAILABLE 1
#    endif
#endif

// If DPCPP backend is not explicitly turned off and SYCL is definitely available, enable it
#if (ONEDPL_USE_DPCPP_BACKEND || !defined(ONEDPL_USE_DPCPP_BACKEND)) && _ONEDPL_SYCL_AVAILABLE
#    define _ONEDPL_BACKEND_SYCL 1
#endif

// Try checking SYCL_LANGUAGE_VERSION after sycl.hpp inclusion if SYCL availability has not been proven yet
// Include SYCL headers for reliable configurations only, the set can be extended in the future
#if defined(__ADAPTIVECPP__)
#    define _ONEDPL_SAFE_TO_INCLUDE_SYCL 1
#else
#    define _ONEDPL_SAFE_TO_INCLUDE_SYCL 0
#endif
#if defined(_ONEDPL_SYCL_POSSIBLY_AVAILABLE) && _ONEDPL_SAFE_TO_INCLUDE_SYCL
#    if __has_include(<sycl/sycl.hpp>)
#        include <sycl/sycl.hpp>
#    else
#        include <CL/sycl.hpp>
#    endif
#    if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#        define _ONEDPL_BACKEND_SYCL 1
#    endif
#endif // _ONEDPL_SYCL_POSSIBLY_AVAILABLE

// If DPCPP backend is explicitly requested and SYCL is definitely not available, throw an error
#if ONEDPL_USE_DPCPP_BACKEND && !_ONEDPL_SYCL_AVAILABLE
#    error "Device execution policies are enabled, \
        but SYCL* headers are not found or the compiler does not support SYCL"
#endif

// If at least one heterogeneous backend is available, enable them
#if _ONEDPL_BACKEND_SYCL
#    if _ONEDPL_HETERO_BACKEND
#        undef _ONEDPL_HETERO_BACKEND
#    endif
#    define _ONEDPL_HETERO_BACKEND 1
#endif

// --------------------------------------------------------------------------------------------------------------------
// Configuration of heterogeneous backends
// --------------------------------------------------------------------------------------------------------------------

#if defined(ONEDPL_FPGA_DEVICE)
#    undef _ONEDPL_FPGA_DEVICE
#    define _ONEDPL_FPGA_DEVICE ONEDPL_FPGA_DEVICE
#endif

#if !defined(ONEDPL_ALLOW_DEFERRED_WAITING)
#    define ONEDPL_ALLOW_DEFERRED_WAITING 0
#endif

#if defined(ONEDPL_FPGA_EMULATOR)
#    undef _ONEDPL_FPGA_EMU
#    define _ONEDPL_FPGA_EMU ONEDPL_FPGA_EMULATOR
#endif

#if defined(ONEDPL_USE_PREDEFINED_POLICIES)
#    undef _ONEDPL_PREDEFINED_POLICIES
#    define _ONEDPL_PREDEFINED_POLICIES ONEDPL_USE_PREDEFINED_POLICIES
#elif !defined(_ONEDPL_PREDEFINED_POLICIES)
#    define _ONEDPL_PREDEFINED_POLICIES 1
#endif

// --------------------------------------------------------------------------------------------------------------------
// Configuration of SYCL heterogeneous backend
// --------------------------------------------------------------------------------------------------------------------

#if _ONEDPL_BACKEND_SYCL
// Include sycl specific options
// FPGA doesn't support sub-groups
#    if !(_ONEDPL_FPGA_DEVICE)
#        define _USE_SUB_GROUPS 1
#        define _USE_GROUP_ALGOS 1
#    endif

#    define _USE_RADIX_SORT (_USE_SUB_GROUPS && _USE_GROUP_ALGOS)

// Compilation of a kernel is requiried to obtain valid work_group_size
// when target devices are CPU or FPGA emulator. Since CPU and GPU devices
// cannot be distinguished during compilation, the macro is enabled by default.
#    if !defined(_ONEDPL_COMPILE_KERNEL)
#        define _ONEDPL_COMPILE_KERNEL 1
#    endif

#    define _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT __has_builtin(__builtin_sycl_unique_stable_name)
#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_HETERO_BACKEND_CONFIG
