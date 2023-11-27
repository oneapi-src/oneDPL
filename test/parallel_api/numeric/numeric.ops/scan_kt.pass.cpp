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

    for (size_t n = 0; n <= 1000000000; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    {
        using Type = float;
        std::cout << "Testing " << n << '\n';
        std::vector<Type> v(n, 1);
        sycl::queue q;
        Type* in_ptr = sycl::malloc_device<Type>(n, q);
        Type* out_ptr = sycl::malloc_device<Type>(n, q);


        q.copy(v.data(), in_ptr, n).wait();
        std::inclusive_scan(v.begin(), v.end(), v.begin());

        using KernelParams = oneapi::dpl::experimental::kt::kernel_param<8, 256, class ScanKernel>;
        oneapi::dpl::experimental::kt::single_pass_inclusive_scan<KernelParams>(q, in_ptr, in_ptr+n, out_ptr, ::std::plus<Type>());

        std::vector<Type> tmp(n, 0);

        q.copy(out_ptr, tmp.data(), n).wait();

        bool passed = true;
        for (size_t i  = 0; i < n; ++i)
        {
            if constexpr (std::is_floating_point<Type>::value)
            {
                if (std::fabs(tmp[i] - v[i]) > 0.001)
                {
                    passed = false;
                    std::cout << "expected " << i << ' ' << v[i] << ' ' << tmp[i] << '\n';
                }
            }
            else
            {
                if (tmp[i] != v[i])
                {
                    passed = false;
                    std::cout << "expected " << i << ' ' << v[i] << ' ' << tmp[i] << '\n';
                }
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
