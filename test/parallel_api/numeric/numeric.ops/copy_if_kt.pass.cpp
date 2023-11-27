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
    sycl::queue q;

    for (int logn : {4, 8, 10, 12, 14})
    {
        std::cout << "Testing 2^" << logn << std::endl;
        int n = 1 << logn;
        std::cout << "n:" << n << std::endl;
        std::vector<int> v(n, 0);
        //for (size_t i = 0; i < v.size(); ++i)
        //  std::cout << v[i] << ",";
        //std::cout << std::endl;

        int* in_ptr = sycl::malloc_device<int>(n, q);
        int* out_ptr = sycl::malloc_device<int>(n, q);

        constexpr int n_elements_per_workitem = 8;

        q.copy(v.data(), in_ptr, n).wait();
        using KernelParams = oneapi::dpl::experimental::kt::kernel_param<n_elements_per_workitem, 128, class ScanKernel>;
        oneapi::dpl::experimental::kt::single_pass_copy_if<KernelParams>(q, in_ptr, in_ptr+n, out_ptr, [](int x) { return x == 0; });

        std::vector<int> tmp(n, 0);
        q.copy(out_ptr, tmp.data(), n);
        q.wait();

        std::copy_if(v.begin(), v.end(), v.begin(), [](int x) { return x == 0; });

        bool passed = true;
        // for (size_t i  = 0; i < n; ++i)
        // {
        //     if (tmp[i] != v[i])
        //     {
        //         passed = false;
        //         std::cout << "expected " << i << ' ' << v[i] << ' ' << tmp[i] << '\n';
        //     }
        // }

        // if (passed)
        //     std::cout << " passed" << std::endl;
        // else
        //     std::cout << " failed" << std::endl;

        for (size_t i = 0; i < n/(n_elements_per_workitem*128) + 1; ++i) {
          std::cout << "i:" << i << " count:" << tmp[i] << std::endl;
        }

        all_passed &= passed;
        sycl::free(in_ptr, q);
        sycl::free(out_ptr, q);
    }

    return !all_passed;
}
