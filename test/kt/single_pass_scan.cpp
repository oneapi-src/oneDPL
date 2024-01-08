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
#include <fstream>
#include <iomanip>

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)
int
main()
{
    bool all_passed = true;

    size_t max_size = (1ul << 33);
    for (size_t n = 0; n <= max_size; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    {
        std::optional<std::ofstream> error_file;
        using Type = int;
        std::cout << "Testing " << n << " (" << sizeof(Type) * n * 2 << " bytes)" << std::endl;

        std::vector<Type> v(n, 1);
        std::vector<Type> ground(n, 1);
        sycl::queue q;
        Type* in_ptr = sycl::malloc_device<Type>(n, q);
        Type* out_ptr = sycl::malloc_device<Type>(n, q);


        q.copy(v.data(), in_ptr, n).wait();
        //std::inclusive_scan(v.begin(), v.end(), ground.begin());
        if (n > 0)
        {
            ground[0] = v[0];
            for (size_t i = 1; i < n; ++i)
                ground[i] = v[i] + ground[i-1];
        }

        using KernelParams = oneapi::dpl::experimental::kt::kernel_param<8, 256, class ScanKernel>;
        oneapi::dpl::experimental::kt::single_pass_inclusive_scan<KernelParams>(q, in_ptr, in_ptr+n, out_ptr, ::std::plus<Type>());

        std::vector<Type> tmp(n, 0);

        q.copy(out_ptr, tmp.data(), n).wait();

        bool passed = true;
        for (size_t i  = 0; i < n; ++i)
        {
            if constexpr (std::is_floating_point<Type>::value)
            {
        //        if (std::fabs(tmp[i] - ground[i]) > 0.001)
                {
                    if (!error_file)
                    {
                        std::stringstream ss;
                        ss << "scan_kt_errors_" << n << ".dat";
                        error_file.emplace(ss.str());
                    }
                    *error_file << i <<  ' ' << std::setprecision(15)  << ground[i] << ' ' << tmp[i] << '\n';
                    passed = false;
                    //std::cout << "expected " << i << ' ' << v[i] << ' ' << tmp[i] << "# " <<   (std::fabs(tmp[i] - v[i])) << '\n';
                }
            }
            else
            {
                if (tmp[i] != ground[i])
                {
                    if (!error_file)
                    {
                        std::stringstream ss;
                        ss << "scan_kt_errors_" << n << ".dat";
                        error_file.emplace(ss.str());
                    }
                    *error_file << i <<  ' ' << std::setprecision(15)  << ground[i] << ' ' << tmp[i] << '\n';
                    passed = false;
                    //std::cout << "expected " << i << ' ' << v[i] << ' ' << tmp[i] << '\n';
                }
            }
        }

        if (passed)
            std::cout << "passed" << std::endl;
        else
            std::cout << "failed" << std::endl;

        //if (!passed)
        //    return 1;

        all_passed &= passed;

        sycl::free(in_ptr, q);
        sycl::free(out_ptr, q);
    }

    return !all_passed;
}
