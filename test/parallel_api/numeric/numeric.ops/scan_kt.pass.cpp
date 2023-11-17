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
    bool all_passed = true;

    //for (size_t n = 0; n <= 100000; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    for (size_t n = 1024; n <= 100000; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    {
        std::cout << "Testing " << n << '\n';
        std::vector<int> v(n, 1);
        sycl::queue q;
        int* in_ptr = sycl::malloc_device<int>(n, q);
        int* out_ptr = sycl::malloc_device<int>(n, q);


        q.copy(v.data(), in_ptr, n).wait();
        std::inclusive_scan(v.begin(), v.end(), v.begin());

        using KernelParams = oneapi::dpl::experimental::kt::kernel_param<8, 256, class ScanKernel>;
        oneapi::dpl::experimental::kt::single_pass_inclusive_scan<KernelParams>(q, in_ptr, in_ptr+n, out_ptr, ::std::plus<int>());

        std::vector<int> tmp(n, 0);

        q.copy(out_ptr, tmp.data(), n).wait();

        bool passed = true;
        for (size_t i  = 0; i < n; ++i)
        {
            if (tmp[i] != v[i])
            {
                passed = false;
                std::cout << "expected " << i << ' ' << v[i] << ' ' << tmp[i] << '\n';
            }
        }

        if (passed)
            std::cout << "passed" << std::endl;
        else
            std::cout << "failed" << std::endl;

        if (!passed)
            return 1;

        all_passed &= passed;
    }

    return !all_passed;
}
