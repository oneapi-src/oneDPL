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
        6, 16, 42, 256, 316, 2048, 5072, 8192, 14001,                        // one work-group
        2<<14, 50000, 67543, 70'000, 71'000, 72'000, 73'000, 74'000, 75'000,  76'000, 77'000,
        78'000, 78'100, 78'200, 78'300, 78'400, 78'500, 78'600, 78'700, 78'800, 78'900,
        79'000, 79'100, 79'200, 79'300, 79'400, 79'500, 79'600, 79'700,
        
        79'800, 79'801, 79'802, 79'803, 79'804, 79'805, 79'806, 79'807, 79'808, 79'809,
        79'810, 79'811, 79'812, 79'813, 79'814, 79'815, 79'816, 79'817, 79'818, 79'819,
        79'820, 79'821, 79'822, 79'823, 79'824, 79'825, 79'826, 79'827, 79'828, 79'829,
        79'830, 79'831, 79'832, 79'833, 79'834, 79'835, 79'836, 79'837, 79'838, 79'839,
        79'840, 79'841, 79'842, 79'843, 79'844, 79'845, 79'846, 79'847, 79'848, 79'849,
        79'850, 79'851, 79'852, 79'853, 79'854, 79'855, 79'856, 79'857, 79'858, 79'859,
        79'860, 79'861, 79'862, 79'863, 79'864, 79'865, 79'866, 79'867, 79'868, 79'869,
        79'870, 79'871, 79'872, 79'873, 79'874, 79'875, 79'876, 79'877, 79'878, 79'879,
        79'880, 79'881, 79'882, 79'883, 79'884, 79'885, 79'886, 79'887, 79'888, 79'889,
        79'890, 79'891, 79'892, 79'893, 79'894, 79'895, 79'896, 79'897, 79'898, 79'899,

        79'900,
        79'910, 79'911, 79'911, 79'913, 79'914, 79'915, 79'916, 79'917, 79'918, 79'919,
        79'920, 79'921, 79'921, 79'923, 79'924, 79'925, 79'926, 79'927, 79'928, 79'929,
        79'930, 79'931, 79'931, 79'933, 79'934, 79'935, 79'936, 79'937, 79'938, 79'939,
        79'940, 79'941, 79'941, 79'943, 79'944, 79'945, 79'946, 79'947, 79'948, 79'949,
        79'950, 79'951, 79'951, 79'953, 79'954, 79'955, 79'956, 79'957, 79'958, 79'959,
        79'960, 79'961, 79'961, 79'963, 79'964, 79'965, 79'966, 79'967, 79'968, 79'969,
        79'970, 79'971, 79'971, 79'973, 79'974, 79'975, 79'976, 79'977, 79'978, 79'979,
        79'980, 79'981, 79'981, 79'983, 79'984, 79'985, 79'986, 79'987, 79'988, 79'989,
        79'990, 79'991, 79'991, 79'993, 79'994, 79'995, 79'996, 79'997, 79'998, 79'999,
        80'000, 90'000, 100'000, 2<<17, 179'581, 250'000,               // cooperative
        2<<18, 500'000, 888'235, 1'000'000, 2<<20, 10'000'000                // onesweep
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
