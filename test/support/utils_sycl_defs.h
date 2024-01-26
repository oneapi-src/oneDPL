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

// This file contains SYCL-specific macros and abstractions to support different versions of SYCL library

#ifndef _UTILS_SYCL_DEFS_H
#define _UTILS_SYCL_DEFS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define TEST_LIBSYCL_VERSION                                                                                    \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define TEST_LIBSYCL_VERSION 0
#endif

#if ONEDPL_FPGA_DEVICE
#    if TEST_LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif // ONEDPL_FPGA_DEVICE

namespace TestUtils
{
using no_init =
#if TEST_LIBSYCL_VERSION >= 50300
    sycl::property::no_init;
#else
    sycl::property::noinit;
#endif

#if ONEDPL_FPGA_DEVICE
#    if TEST_LIBSYCL_VERSION >= 50300
using fpga_emulator_selector = sycl::ext::intel::fpga_emulator_selector;
using fpga_selector = sycl::ext::intel::fpga_selector;
#    else
using fpga_emulator_selector = sycl::INTEL::fpga_emulator_selector;
using fpga_selector = sycl::INTEL::fpga_selector;
#    endif
#endif // ONEDPL_FPGA_DEVICE

template <typename Buf>
auto
get_host_access(Buf&& buf)
{
#if TEST_LIBSYCL_VERSION >= 60200
    return ::std::forward<Buf>(buf).get_host_access(sycl::read_only);
#else
    return ::std::forward<_Buf>(buf).template get_access<sycl::access::mode::read>();
#endif
}

} // TestUtils

#endif //  _UTILS_SYCL_DEFS_H
