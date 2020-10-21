// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

// Define SYCL specific options

#ifndef _ONEDPL_SYCL_CONFIG_H
#define _ONEDPL_SYCL_CONFIG_H

// This is supposed to be a simple config file which only sets flags, but
// unfortunately we also need to include CL/sycl.hpp here.
// The issue is that __SYCL_COMPILER_VERSION is defined inside of that header
// and not by the compiler itself, so we need to pull the option first to
// correctly set flags that depends on it.
#if _PSTL_BACKEND_SYCL
#    include <CL/sycl.hpp>
#endif

// FPGA doesn't support sub-groups
#if !(_PSTL_FPGA_DEVICE)
#    define _USE_SUB_GROUPS 1
#    define _USE_GROUP_ALGOS 1
#endif

#define _USE_RADIX_SORT (_USE_SUB_GROUPS && _USE_GROUP_ALGOS)

// Compilation of a kernel is requiried to obtain valid work_group_size
// when target devices are CPU or FPGA emulator. Since CPU and GPU devices
// cannot be distinguished during compilation, the macro is enabled by default.
#if !defined(_PSTL_COMPILE_KERNEL)
#    define _PSTL_COMPILE_KERNEL 1
#endif

#endif /* _ONEDPL_SYCL_CONFIG_H */
