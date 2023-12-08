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
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

using namespace TestUtils;

template <typename T, typename Predicate>
class CopyIfKernel;

template <typename T, typename Predicate, typename Generator>
bool
test(Predicate pred, Generator gen)
{
    bool all_passed = true;
    sycl::queue q;

    for (int logn : {4, 8, 10, 12, 14, 15, 18})
    {
        int n = 1 << logn;

        Sequence<T> in(n, [&](size_t k) -> T { return gen(n ^ k); });

        Sequence<T> std_out(n);

        T* in_ptr = sycl::malloc_device<T>(n, q);
        T* out_ptr = sycl::malloc_device<T>(n, q);
        size_t* out_num = sycl::malloc_device<size_t>(1, q);

        constexpr int n_elements_per_workitem = 8;

        q.copy(in.data(), in_ptr, n).wait();
        using KernelParams =
            oneapi::dpl::experimental::kt::kernel_param<n_elements_per_workitem, 128, CopyIfKernel<T, Predicate>>;
        oneapi::dpl::experimental::kt::single_pass_copy_if<KernelParams>(q, in_ptr, in_ptr + n, out_ptr, out_num, pred);

        Sequence<T> kt_out(n);
        size_t num_selected = 0;
        q.copy(out_ptr, kt_out.data(), n);
        q.copy(out_num, &num_selected, 1);
        q.wait();

        auto std_out_end = std::copy_if(in.begin(), in.end(), std_out.begin(), pred);

        bool passed = true;
        if (num_selected != (std_out_end - std_out.begin()))
        {
            passed = false;
            std::cout << "Num selected wrong: expected " << (std_out_end - std_out.begin()) << " " << num_selected
                      << "\n";
        }

        for (size_t i = 0; i < (std_out_end - std_out.begin()); ++i)
        {
            if (kt_out[i] != std_out[i])
            {
                passed = false;
                std::cout << "expected " << i << ' ' << std_out[i] << ' ' << kt_out[i] << '\n';
            }
        }

        if (passed)
            std::cout << " passed" << std::endl;
        else
            std::cout << " failed" << std::endl;

        all_passed &= passed;
        sycl::free(in_ptr, q);
        sycl::free(out_ptr, q);
        sycl::free(out_num, q);
    }

    return all_passed;
}

int
main()
{
    bool all_passed = true;
    all_passed &=
        test<float64_t>([](const float64_t& x) { return x * x <= 1024; },
                        [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? float64_t(j % 32) : float64_t(j % 33 + 34); });
    all_passed &= test<int>([](const int&) { return true; }, [](size_t j) { return j; });
    all_passed &= test<std::int32_t>([](const std::int32_t& x) { return x != 42; },
                                     [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? std::int32_t(j + 1) : 42; });

    return all_passed;
}
