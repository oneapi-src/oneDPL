// -*- C++ -*-
//===-- scan.pass.cpp -----------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

int
main()
{
    int n = 1 << 16;
    sycl::queue q;
    int* in_ptr = sycl::malloc_device<int>(n, q);
    int* out_ptr = sycl::malloc_device<int>(n, q);
    oneapi::dpl::experimental::igpu::single_pass_inclusive_scan(oneapi::dpl::execution::dpcpp_default, in_ptr, in_ptr+n, out_ptr, ::std::plus<int>());
    return 0;
}
