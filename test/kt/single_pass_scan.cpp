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


#include <oneapi/dpl/experimental/kernel_templates>

#if LOG_TEST_INFO
#include <iostream>
#endif

#include "../support/test_config.h"
#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"

//#include _PSTL_TEST_HEADER(execution)
//#include _PSTL_TEST_HEADER(numeric)

template <typename T, sycl::usm::alloc _alloc_type, typename KernelParam>
void
test_usm(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>()
              ">(" << size << ");" << std::endl;
#endif
    std::vector<T> expected(size);
    //generate_data(expected.data(), size, 42);
    std::fill(expected.begin(), expected.end(), 42);

    TestUtils::usm_data_transfer<_alloc_type, T> dt_input(q, expected.begin(), expected.end());
    TestUtils::usm_data_transfer<_alloc_type, T> dt_output(q, size);

    std::inclusive_scan(expected.begin(), expected.end(), expected.begin());

    oneapi::dpl::experimental::kt::single_pass_inclusive_scan(q, dt_input.get_data(), dt_input.get_data() + size, dt_output.get_data(), ::std::plus<T>(), param)
        .wait();

    std::vector<T> actual(size);
    dt_output.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, typename KernelParam>
bool
can_run_test(sycl::queue q, KernelParam param)
{
    const auto max_slm_size = q.get_device().template get_info<sycl::info::device::local_mem_size>();
    // skip tests with error: LLVM ERROR: SLM size exceeds target limits
    return sizeof(T) * param.data_per_workitem * param.workgroup_size < max_slm_size;
}

template <typename T, typename KernelParam>
void
test_general_cases(sycl::queue q, std::size_t size, KernelParam param)
{
    test_usm<T, sycl::usm::alloc::shared>(q, size, param);
    test_usm<T, sycl::usm::alloc::device>(q, size, param);
//    test_sycl_iterators<T, IsAscending, RadixBits>(q, size, param);
//    test_sycl_buffer<T, IsAscending, RadixBits>(q, size, param);
}


int
main()
{
    using T = int;
    constexpr int data_per_work_item = 8;
    constexpr int work_group_size = 256;
    constexpr oneapi::dpl::experimental::kt::kernel_param<data_per_work_item, work_group_size> params;
    auto q = TestUtils::get_test_queue();
    bool run_test = can_run_test<T>(q, params);
    if (run_test)
    {
        try
        {
            /*for (auto size : sort_sizes)
            {
                test_general_cases<TEST_KEY_TYPE, Ascending, TestRadixBits>(q, size, params);
                test_general_cases<TEST_KEY_TYPE, Descending, TestRadixBits>(q, size, params);
            }*/
            test_general_cases<T>(q, 65536, params);
        }
        catch (const ::std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    return TestUtils::done(run_test);
    /*bool all_passed = true;

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

    return !all_passed;*/
}
