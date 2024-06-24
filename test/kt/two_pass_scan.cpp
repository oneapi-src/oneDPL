// -*- C++ -*-
//===-- single_pass_scan.cpp ----------------------------------------------===//
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

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>

#if LOG_TEST_INFO
#    include <iostream>
#endif

#include <oneapi/dpl/ranges>

#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"
#include "../support/scan_serial_impl.h"

#include "esimd_radix_sort_utils.h"

#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>
#include <type_traits>

inline const std::vector<std::size_t> scan_sizes = {
    1,       6,         16,      43,        256,           316,           2048,
    5072,    8192,      14001,   1 << 14,   (1 << 14) + 1, 50000,         67543,
    100'000, 1 << 17,   179'581, 250'000,   1 << 18,       (1 << 18) + 1, 500'000,
    888'235, 1'000'000, 1 << 20, 10'000'000};


template <typename BinOp, typename T>
auto
generate_scan_data(T* input, std::size_t size, std::uint32_t seed)
{
    // Integer numbers are generated even for floating point types in order to avoid rounding errors,
    // and simplify the final check
    using substitute_t = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;

    const substitute_t start = std::is_signed_v<T> ? -10 : 0;
    const substitute_t end = 10;

    std::default_random_engine gen{seed};
    std::uniform_int_distribution<substitute_t> dist(start, end);
    std::generate(input, input + size, [&] { return dist(gen); });

    if constexpr (std::is_same_v<std::multiplies<T>, BinOp>)
    {
        std::size_t custom_item_count = size < 5 ? size : 5;
        std::fill(input + custom_item_count, input + size, 1);
        std::replace(input, input + custom_item_count, 0, 2);
        std::shuffle(input, input + size, gen);
    }
}

template <typename T, sycl::usm::alloc _alloc_type, typename BinOp, typename KernelParam>
void
test_usm(sycl::queue q, std::size_t size, BinOp bin_op, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>() << ">("
              << size << ");" << std::endl;
#endif
    std::vector<T> expected(size);
    generate_scan_data<BinOp>(expected.data(), size, 42);

    TestUtils::usm_data_transfer<_alloc_type, T> dt_input(q, expected.begin(), expected.end());
    TestUtils::usm_data_transfer<_alloc_type, T> dt_output(q, size);

    inclusive_scan_serial(expected.begin(), expected.end(), expected.begin(), bin_op);

    oneapi::dpl::experimental::kt::gpu::two_pass_inclusive_scan<typename KernelParam::kernel_name>(
        q, dt_input.get_data(), dt_input.get_data() + size, dt_output.get_data(), bin_op, oneapi::dpl::unseq_backend::__known_identity<BinOp, T>);

    std::vector<T> actual(size);
    dt_output.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, typename BinOp, typename KernelParam>
void
test_sycl_iterators(sycl::queue q, std::size_t size, BinOp bin_op, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif
    std::vector<T> input(size);
    std::vector<T> output(size);
    generate_scan_data<BinOp>(input.data(), size, 42);
    std::vector<T> ref(input);
    inclusive_scan_serial(std::begin(ref), std::end(ref), std::begin(ref), bin_op);
    {
        sycl::buffer<T> buf(input.data(), input.size());
        sycl::buffer<T> buf_out(output.data(), output.size());
        oneapi::dpl::experimental::kt::gpu::two_pass_inclusive_scan<typename KernelParam::kernel_name>(
          q, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), oneapi::dpl::begin(buf_out), bin_op, oneapi::dpl::unseq_backend::__known_identity<BinOp, T>);
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, output, msg.c_str());
}

template <typename T, typename BinOp, typename KernelParam>
void
test_all_view(sycl::queue q, std::size_t size, BinOp bin_op, KernelParam param)
{
#    if LOG_TEST_INFO
    std::cout << "\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#    endif
    std::vector<T> input(size);
    generate_scan_data<BinOp>(input.data(), size, 42);
    std::vector<T> ref(input);
    sycl::buffer<T> buf_out(input.size());

    inclusive_scan_serial(std::begin(ref), std::end(ref), std::begin(ref), bin_op);
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read> view(buf);
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view_out(buf_out);
        oneapi::dpl::experimental::kt::gpu::ranges::two_pass_transform_inclusive_scan<typename KernelParam::kernel_name>(
          q, view, view_out, bin_op, oneapi::dpl::__internal::__no_op(), oneapi::dpl::unseq_backend::__known_identity<BinOp, T>);
    }

    auto acc = buf_out.get_host_access();

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, acc, msg.c_str());
}

template <typename T, typename BinOp, typename KernelParam>
void
test_buffer(sycl::queue q, std::size_t size, BinOp bin_op, KernelParam param)
{
#    if LOG_TEST_INFO
    std::cout << "\ttest_buffer(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#    endif
    std::vector<T> input(size);
    generate_scan_data<BinOp>(input.data(), size, 42);
    std::vector<T> ref(input);
    sycl::buffer<T> buf_out(input.size());

    inclusive_scan_serial(std::begin(ref), std::end(ref), std::begin(ref), bin_op);
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::kt::gpu::ranges::two_pass_transform_inclusive_scan<typename KernelParam::kernel_name>(
          q, buf, buf_out, bin_op, oneapi::dpl::__internal::__no_op(), oneapi::dpl::unseq_backend::__known_identity<BinOp, T>);
    }

    auto acc = buf_out.get_host_access();

    std::string msg = "wrong results with buffer, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, acc, msg.c_str());
}

template <typename T, typename BinOp, typename KernelParam>
void
test_general_cases(sycl::queue q, std::size_t size, BinOp bin_op, KernelParam param)
{
    test_usm<T, sycl::usm::alloc::shared>(q, size, bin_op, TestUtils::get_new_kernel_params<0>(param));
    test_usm<T, sycl::usm::alloc::device>(q, size, bin_op, TestUtils::get_new_kernel_params<1>(param));
    test_sycl_iterators<T>(q, size, bin_op, TestUtils::get_new_kernel_params<2>(param));
    test_all_view<T>(q, size, bin_op, TestUtils::get_new_kernel_params<3>(param));
    test_buffer<T>(q, size, bin_op, TestUtils::get_new_kernel_params<4>(param));
}

template <typename T, typename KernelParam>
void
test_all_cases(sycl::queue q, std::size_t size, KernelParam param)
{
    test_general_cases<T>(q, size, std::plus<T>{}, TestUtils::get_new_kernel_params<0>(param));
#if _PSTL_GROUP_REDUCTION_MULT_INT64_BROKEN
    static constexpr bool int64_mult_broken = std::is_integral_v<T> && (sizeof(T) == 8);
#else
    static constexpr bool int64_mult_broken = 0;
#endif
    if constexpr (!int64_mult_broken)
    {
        test_general_cases<T>(q, size, std::multiplies<T>{}, TestUtils::get_new_kernel_params<1>(param));
    }
}


template <typename KernelParam>
void
test_all_types(sycl::queue q, std::size_t size, KernelParam params)
{
    test_all_cases<uint8_t>(q, size, params);
    test_all_cases<int8_t>(q, size, params);

    test_all_cases<uint16_t>(q, size, params);
    test_all_cases<int16_t>(q, size, params);

    test_all_cases<uint32_t>(q, size, params);
    test_all_cases<int>(q, size, params);

    test_all_cases<size_t>(q, size, params);
    test_all_cases<uint64_t>(q, size, params);

    test_all_cases<float>(q, size, params);
    test_all_cases<double>(q, size, params);
}

int
main()
{
    constexpr oneapi::dpl::experimental::kt::kernel_param<0, 0> params;
    auto q = TestUtils::get_test_queue();

    try
    {
        for (auto size : scan_sizes)
            test_all_types(q, size, params);
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }

    return TestUtils::done(true);
}
