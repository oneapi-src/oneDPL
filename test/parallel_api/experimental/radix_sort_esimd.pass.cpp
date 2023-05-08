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

template <typename T>
typename ::std::enable_if_t<std::is_arithmetic_v<T>, void>
generate_data(T* input, std::size_t size)
{
    std::default_random_engine gen{42};
    std::size_t unique_threshold = 75 * size / 100;
    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
        std::generate(input, input + unique_threshold, [&]{ return dist(gen); });
    }
    else
    {
        std::uniform_real_distribution<T> dist_real(std::numeric_limits<T>::min(), log2(1e12));
        std::uniform_int_distribution<int> dist_binary(0, 1);
        auto randomly_signed_real = [&dist_real, &dist_binary, &gen](){
            auto v = exp2(dist_real(gen));
            return dist_binary(gen) == 0 ? v: -v;
        };
        std::generate(input, input + unique_threshold, [&]{ return randomly_signed_real(); });
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
    std::vector<T> ref(input);
    std::sort(std::begin(ref), std::end(ref));
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::esimd::radix_sort<256,16>(policy, view);
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template<typename T>
void test_subrange_view(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    T* input = sycl::malloc_shared<T>(size, q);
    T* ref = sycl::malloc_host<T>(size, q);
    generate_data(ref, size);
    q.copy(ref, input, size).wait();
    std::sort(ref, ref + size);

    oneapi::dpl::experimental::ranges::views::subrange view(input, input + size);
    oneapi::dpl::experimental::esimd::radix_sort<256,16>(policy, view);

    T* host_input = sycl::malloc_host<T>(size, q);
    q.copy(input, host_input, size).wait();

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(ref, input, size, msg.c_str());

    sycl::free(input, q);
    sycl::free(ref, q);
    sycl::free(host_input, q);
}
#endif // _ENABLE_RANGES_TESTING

template<typename T>
void test_usm(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    T* input = sycl::malloc_shared<T>(size, q);
    T* ref = sycl::malloc_host<T>(size, q);
    generate_data(ref, size);
    q.copy(ref, input, size).wait();
    std::sort(ref, ref + size);
    oneapi::dpl::experimental::esimd::radix_sort<256,16>(policy, input, input + size);

    T* host_input = sycl::malloc_host<T>(size, q);
    q.copy(input, host_input, size).wait();

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(ref, input, size, msg.c_str());

    sycl::free(input, q);
    sycl::free(ref, q);
    sycl::free(host_input, q);
}

template<typename T>
void test_sycl_iterators(std::size_t size)
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::sort(std::begin(ref), std::end(ref));

    oneapi::dpl::experimental::esimd::radix_sort<256,16>(policy, oneapi::dpl::begin(input), oneapi::dpl::end(input));

    std::string msg = "wrong results with sycl_iterator, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

void test_small_sizes()
{
    sycl::queue q{};
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    generate_data(input.data(), input.size());
    std::vector<uint32_t> ref(input);

    oneapi::dpl::experimental::esimd::radix_sort<256,16>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input));
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 0");
    oneapi::dpl::experimental::esimd::radix_sort<256,16>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input) + 1);
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 1");
}

// TODO: add ascending and descending sorting orders
// TODO: provide exit code to indicate wrong results
template<typename T>
void test_general_cases(std::size_t size)
{
#if _ENABLE_RANGES_TESTING
    test_all_view<T>(size);
    test_subrange_view<T>(size);
#endif // _ENABLE_RANGES_TESTING
    test_usm<T>(size);
    test_sycl_iterators<T>(size);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const std::vector<std::size_t> sizes = {
        6, 16, 42, 256, 316, 2048, 5072, 8192, 14001,                        // one work-group
        2<<14, 50000, 67543, 100'000, 2<<17, 179'581, 250'000,               // cooperative
        2<<18, 500'000, 888'235, 1'000'000, 2<<20, 10'000'000                // onesweep
    };

    try
    {
        for(auto size: sizes)
        {
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
