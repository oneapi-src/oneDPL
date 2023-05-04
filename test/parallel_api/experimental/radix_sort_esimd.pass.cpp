// -*- C++ -*-
//===-- rasix_sort_esimd.pass.cpp -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <iostream>

constexpr std::uint16_t kWorkGroupSize = 256;
constexpr std::uint16_t kDataPerWorkItem = 16;

template <typename T>
typename ::std::enable_if_t<std::is_arithmetic_v<T>, void>
generate_data(T* input, std::size_t size)
{
    std::default_random_engine gen{std::random_device{}()};
    std::size_t unique_threshold = 75 * size / 100;
    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(0);
        std::generate(input, input + unique_threshold, [&]{ return dist(gen); });
    }
    else
    {
        std::uniform_real_distribution<T> dist(0.0, log2(1e12));
        std::generate(input, input + unique_threshold, [&]{ return exp2(dist(gen)); });
    }
    for(uint32_t i = 0, j = unique_threshold; j < size; ++i, ++j)
    {
        input[j] = input[i];
    }
}

#if _ENABLE_RANGES_TESTING
template<typename T>
void test_all_view(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> expected(input);
    std::sort(std::begin(expected), std::end(expected));
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize, kDataPerWorkItem>(policy, view);
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(expected, input, msg.c_str());
}

template<typename T>
void test_subrange_view(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    T* input = sycl::malloc_shared<T>(size, q);
    T* expected = sycl::malloc_host<T>(size, q);
    generate_data(expected, size);
    q.copy(expected, input, size).wait();
    std::sort(expected, expected + size);

    oneapi::dpl::experimental::ranges::views::subrange view(input, input + size);
    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize, kDataPerWorkItem>(policy, view);

    T* host_input = sycl::malloc_host<T>(size, q);
    q.copy(input, host_input, size).wait();

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(expected, input, size, msg.c_str());

    sycl::free(input, q);
    sycl::free(expected, q);
    sycl::free(host_input, q);
}
#endif // _ENABLE_RANGES_TESTING

template<typename T>
void test_usm(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    T* input = sycl::malloc_shared<T>(size, q);
    T* expected = sycl::malloc_host<T>(size, q);
    generate_data(expected, size);
    q.copy(expected, input, size).wait();
    std::sort(expected, expected + size);
    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize, kDataPerWorkItem>(policy, input, input + size);

    T* host_input = sycl::malloc_host<T>(size, q);
    q.copy(input, host_input, size).wait();

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected, input, size, msg.c_str());

    sycl::free(input, q);
    sycl::free(expected, q);
    sycl::free(host_input, q);
}

template<typename T>
void test_sycl_iterators(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> expected(input);
    std::sort(std::begin(expected), std::end(expected));

    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize, kDataPerWorkItem>(policy, oneapi::dpl::begin(input),
                                                                                   oneapi::dpl::end(input));

    std::string msg = "wrong results with sycl_iterator, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(expected, input, msg.c_str());
}

void test_small_sizes()
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    generate_data(input.data(), input.size());
    std::vector<uint32_t> expected(input);

    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize, kDataPerWorkItem>(policy, oneapi::dpl::begin(input),
                                                                                   oneapi::dpl::begin(input));
    EXPECT_EQ_RANGES(expected, input, "sort modified input data when size == 0");
    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize, kDataPerWorkItem>(policy, oneapi::dpl::begin(input),
                                                                                   oneapi::dpl::begin(input) + 1);
    EXPECT_EQ_RANGES(expected, input, "sort modified input data when size == 1");
}

// TODO: add ascending and descending sorting orders
// TODO: provide exit code to indicate wrong results
template<typename T>
void test_general_cases(std::size_t size)
{
//#if _ENABLE_RANGES_TESTING
//    std::cout << "\ttest_all_view<T>(size);" << std::endl;
//    test_all_view<T>(size);
//    std::cout << "\ttest_subrange_view<T>(size);" << std::endl;
//    test_subrange_view<T>(size);
//#endif // _ENABLE_RANGES_TESTING
    std::cout << "\ttest_usm<T>(size);" << std::endl;
    test_usm<T>(size);
    //std::cout << "\ttest_sycl_iterators<T>(size);" << std::endl;
    //test_sycl_iterators<T>(size);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const std::vector<std::size_t> sizes = {
        79'873
    };

    try
    {
        for(auto size: sizes)
        {
            std::cout << "Run tests for size = " << size << std::endl;

            test_general_cases<uint32_t>(size);
            // test_general_cases<int>(size);
            // test_general_cases<float>(size);
            // test_general_cases<double>(size);
        }
        test_small_sizes();
    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
