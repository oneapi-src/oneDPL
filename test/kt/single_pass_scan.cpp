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

#include "esimd_radix_sort_utils.h"

inline const std::vector<std::size_t> scan_sizes = {
    1,       6,         16,      43,        256,           316,           2048,
    5072,    8192,      14001,   1 << 14,   (1 << 14) + 1, 50000,         67543,
    100'000, 1 << 17,   179'581, 250'000,   1 << 18,       (1 << 18) + 1, 500'000,
    888'235, 1'000'000, 1 << 20, 10'000'000};

template <typename T, sycl::usm::alloc _alloc_type, typename KernelParam>
void
test_usm(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>()
              ">(" << size << ");" << std::endl;
#endif
    std::vector<T> expected(size);
    generate_data(expected.data(), size, 42);

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
void
test_sycl_iterators(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif
    std::vector<T> input(size);
    generate_data(input.data(), size, 42);
    std::vector<T> ref(input);
    std::inclusive_scan(std::begin(ref), std::end(ref), std::begin(ref));
    {
        sycl::buffer<T> buf(input.data(), input.size());
        sycl::buffer<T> buf_out(input.size());
        oneapi::dpl::experimental::kt::single_pass_inclusive_scan(q, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), oneapi::dpl::begin(buf_out), ::std::plus<T>(),
                                                                      param)
            .wait();
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}


template <typename T, typename KernelParam>
void
test_general_cases(sycl::queue q, std::size_t size, KernelParam param)
{
    test_usm<T, sycl::usm::alloc::shared>(q, size, param);
    test_usm<T, sycl::usm::alloc::device>(q, size, param);
    test_sycl_iterators<T>(q, size, param);
}


int
main()
{
    using T = int;
    constexpr int data_per_work_item = 8;
    constexpr int work_group_size = 256;
    constexpr oneapi::dpl::experimental::kt::kernel_param<data_per_work_item, work_group_size> params;
    auto q = TestUtils::get_test_queue();
    try
    {
        for (auto size : scan_sizes)
          test_general_cases<T>(q, size, params);
    }
    catch (const ::std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }

    return TestUtils::done();
}
