// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#ifndef _DPSTD_CONFIG_H
#define _DPSTD_CONFIG_H

// Check deprecated macros
#if defined(_PSTL_BACKEND_SYCL)
#    pragma message(                                                                                                   \
        "WARNING: _PSTL_BACKEND_SYCL is deprecated and will be removed by Gold. To not have a dependency on device policies, please set ONEDPL_USE_DPCPP_BACKEND to 0.")
#elif defined(ONEDPL_STANDARD_POLICIES_ONLY)
#    pragma message(                                                                                                   \
        "WARNING: ONEDPL_STANDARD_POLICIES_ONLY is deprecated and will be removed by Gold. To not have a dependency on device policies, please set ONEDPL_USE_DPCPP_BACKEND to 0.")
#    define _PSTL_BACKEND_SYCL !ONEDPL_STANDARD_POLICIES_ONLY
#elif defined(ONEDPL_USE_DPCPP_BACKEND)
#    define _PSTL_BACKEND_SYCL ONEDPL_USE_DPCPP_BACKEND
#endif

#if defined(_PSTL_FPGA_DEVICE)
#    pragma message("WARNING: _PSTL_FPGA_DEVICE is deprecated. Please define ONEDPL_FPGA_DEVICE instead.")
#elif defined(ONEDPL_FPGA_DEVICE)
#    undef _PSTL_FPGA_DEVICE
#    define _PSTL_FPGA_DEVICE ONEDPL_FPGA_DEVICE
#endif

#if defined(_PSTL_FPGA_EMU)
#    pragma message("WARNING: _PSTL_FPGA_EMU is deprecated. Please define ONEDPL_FPGA_EMULATOR instead.")
#elif defined(ONEDPL_FPGA_EMULATOR)
#    undef _PSTL_FPGA_EMU
#    define _PSTL_FPGA_EMU ONEDPL_FPGA_EMULATOR
#endif

// macros for deprecation
#if (__cplusplus >= 201402L)
#    define _DPSTD_DEPRECATED [[deprecated]]
#    define _DPSTD_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#elif _MSC_VER
#    define _DPSTD_DEPRECATED __declspec(deprecated)
#    define _DPSTD_DEPRECATED_MSG(msg) __declspec(deprecated(msg))
#elif (__GNUC__ || __clang__)
#    define _DPSTD_DEPRECATED __attribute__((deprecated))
#    define _DPSTD_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#else
#    define _DPSTD_DEPRECATED
#    define _DPSTD_DEPRECATED_MSG(msg)
#endif

#define _ITERATORS_DEPRECATED _DPSTD_DEPRECATED
#define _POLICY_DEPRECATED _DPSTD_DEPRECATED

#endif /* _DPSTD_config_H */
