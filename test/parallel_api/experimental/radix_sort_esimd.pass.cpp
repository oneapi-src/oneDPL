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

#include "support/sycl_alloc_utils.h"

#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <iostream>
#include <cmath>
#include <limits>

#ifndef LOG_TEST_INFO
#define LOG_TEST_INFO 0
#endif

#ifndef TEST_ALL_INPUTS
#define TEST_ALL_INPUTS 0
#endif

template <typename T, bool Order>
struct Compare : public std::less<T> {};

template <typename T>
struct Compare<T, false> : public std::greater<T> {};

constexpr bool kAscending = true;
constexpr bool kDescending = false;

constexpr ::std::uint16_t kWorkGroupSize = 256;
constexpr ::std::uint16_t kDataPerWorkItem = 16;

#if LOG_TEST_INFO
struct TypeInfo
{
    template <typename T>
    const std::string& name()
    {
        static const std::string kTypeName = "unknown type name";
        return kTypeName;
    }

    template <>
    const std::string& name<int16_t>()
    {
        static const std::string kTypeName = "int16_t";
        return kTypeName;
    }

    template <>
    const std::string& name<uint16_t>()
    {
        static const std::string kTypeName = "uint16_t";
        return kTypeName;
    }

    template <>
    const std::string& name<uint32_t>()
    {
        static const std::string kTypeName = "uint32_t";
        return kTypeName;
    }

    template <>
    const std::string& name<uint64_t>()
    {
        static const std::string kTypeName = "uint64_t";
        return kTypeName;
    }

    template <>
    const std::string& name<int64_t>()
    {
        static const std::string kTypeName = "int64_t";
        return kTypeName;
    }

    template <>
    const std::string& name<int>()
    {
        static const std::string kTypeName = "int";
        return kTypeName;
    }

    template <>
    const std::string& name<float>()
    {
        static const std::string kTypeName = "float";
        return kTypeName;
    }

    template <>
    const std::string& name<double>()
    {
        static const std::string kTypeName = "double";
        return kTypeName;
    }
};
struct USMAllocPresentation
{
    template <sycl::usm::alloc>
    const std::string& name()
    {
        static const std::string kUSMAllocTypeName = "unknown";
        return kUSMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::host>()
    {
        static const std::string kUSMAllocTypeName = "sycl::usm::alloc::host";
        return kUSMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::device>()
    {
        static const std::string kUSMAllocTypeName = "sycl::usm::alloc::device";
        return kUSMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::shared>()
    {
        static const std::string kUSMAllocTypeName = "sycl::usm::alloc::shared";
        return kUSMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::unknown>()
    {
        static const std::string kUSMAllocTypeName = "sycl::usm::alloc::unknown";
        return kUSMAllocTypeName;
    }
};
#endif // LOG_TEST_INFO

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

template<typename Container1, typename Container2>
void print_data(const Container1& expected, const Container2& actual, std::size_t first, std::size_t n = 0)
{
    if (expected.size() <= first) return;
    if (n==0 || expected.size() < first+n)
        n = expected.size() - first;

    if constexpr (std::is_floating_point_v<typename Container1::value_type>)
        std::cout << std::hexfloat;
    else
        std::cout << std::hex;

    for (std::size_t i=first; i < first+n; ++i)
    {
        std::cout << actual[i] << " --- " << expected[i] << std::endl;
    }

    if constexpr (std::is_floating_point_v<typename Container1::value_type>)
        std::cout << std::defaultfloat << std::endl;
    else
        std::cout << std::dec << std::endl;
}

#if _ENABLE_RANGES_TESTING
template<typename T, bool Order>
void test_all_view(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::sort(std::begin(ref), std::end(ref), Compare<T, Order>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize,kDataPerWorkItem,Order>(policy, view);
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template<typename T, bool Order>
void test_subrange_view(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_subrange_view<T, " << Order << ">(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_input(q, expected.begin(), expected.end());

    std::sort(expected.begin(), expected.end(), Compare<T, Order>{});

    oneapi::dpl::experimental::ranges::views::subrange view(dt_input.get_data(), dt_input.get_data() + size);
    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize,kDataPerWorkItem,Order>(policy, view);

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

#endif // _ENABLE_RANGES_TESTING

template<typename T, sycl::usm::alloc _alloc_type, bool Order>
void test_usm(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>() << ", " << Order << ">("<< size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<_alloc_type, T> dt_input(q, expected.begin(), expected.end());

    std::sort(expected.begin(), expected.end(), Compare<T, Order>{});

    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize,kDataPerWorkItem,Order>(policy, dt_input.get_data(), dt_input.get_data() + size);

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, bool Order>
void
test_usm(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_usm<T, " << Order << ">(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#endif

    test_usm<T, sycl::usm::alloc::shared, Order>(size);
    test_usm<T, sycl::usm::alloc::device, Order>(size);
}

template<typename T, bool Order>
void test_sycl_iterators(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::sort(std::begin(ref), std::end(ref), Compare<T, Order>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize,kDataPerWorkItem,Order>(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

void test_small_sizes()
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_small_sizes();" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    std::vector<uint32_t> ref(input);

    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize,kDataPerWorkItem,kAscending>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input));
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 0");
    oneapi::dpl::experimental::esimd::radix_sort<kWorkGroupSize,kDataPerWorkItem,kAscending>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input) + 1);
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 1");
}

template <typename T>
void test_general_cases(std::size_t size)
{
#if _ENABLE_RANGES_TESTING
    if constexpr (sizeof(T) <= sizeof(::std::uint32_t))
    {
        test_all_view<T, kAscending>(size);
        test_all_view<T, kDescending>(size);
        test_subrange_view<T, kAscending>(size);
        test_subrange_view<T, kDescending>(size);
    }
    else
    {
        // TODO required to implement
    }
#endif // _ENABLE_RANGES_TESTING
    test_usm<T, kAscending>(size);
    test_usm<T, kDescending>(size);

    if constexpr (sizeof(T) <= sizeof(::std::uint32_t))
    {
        test_sycl_iterators<T, kAscending>(size);
        test_sycl_iterators<T, kDescending>(size);
    }
    else
    {
        // TODO required to implement
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const std::vector<std::size_t> sizes = {
        6, 16, 43, 256, 316, 2048, 5072, 8192, 14001, 1<<14,
        (1<<14)+1, 50000, 67543, 100'000, 1<<17, 179'581, 250'000, 1<<18,
        (1<<18)+1, 500'000, 888'235, 1'000'000, 1<<20, 10'000'000
    };

    try
    {
#if TEST_ALL_INPUTS
        const std::vector<std::size_t> onewg_sizes   {std::begin(sizes),      std::begin(sizes) + 10};
        const std::vector<std::size_t> coop_sizes    {std::begin(sizes) + 10, std::begin(sizes) + 18};
        const std::vector<std::size_t> onesweep_sizes{std::begin(sizes) + 18, std::end(sizes)};

        for(auto size: onewg_sizes)
        {
            test_general_cases<int16_t >(size);
            test_general_cases<uint16_t>(size);
            test_general_cases<int     >(size);
            test_general_cases<uint32_t>(size);
            test_general_cases<uint64_t>(size);
            test_general_cases<int64_t >(size);
            test_general_cases<float   >(size);
            //test_general_cases<double  >(size);
        }
        for(auto size: coop_sizes)
        {
            test_general_cases<int16_t >(size);
            test_general_cases<uint16_t>(size);
            test_general_cases<int     >(size);
            test_general_cases<uint32_t>(size);
            test_general_cases<uint64_t>(size);
            test_general_cases<int64_t >(size);
            test_general_cases<float   >(size);
            //test_general_cases<double  >(size);
        }
        for(auto size: onesweep_sizes)
        {
            test_usm<int16_t,  kAscending>(size);
            test_usm<uint16_t, kAscending>(size);
            test_usm<int,      kAscending>(size);
            test_usm<uint32_t, kAscending>(size);
            // Not implemented for onesweep
            //test_usm<uint64_t, kAscending>(size);
            //test_usm<int64_t,  kAscending>(size);
            test_usm<float,    kAscending>(size);

            test_usm<int16_t,  kDescending>(size);
            test_usm<uint16_t, kDescending>(size);
            test_usm<int,      kDescending>(size);
            test_usm<uint32_t, kDescending>(size);
            // Not implemented for onesweep
            //test_usm<uint64_t, kDescending>(size);
            //test_usm<int64_t,  kDescending>(size);
            test_usm<float,    kDescending>(size);
        }
        test_small_sizes();
#else
        for(auto size: sizes)
        {
            test_usm<int16_t,  sycl::usm::alloc::shared, kAscending>(size);
            test_usm<uint16_t, sycl::usm::alloc::shared, kAscending>(size);
            test_usm<int,      sycl::usm::alloc::shared, kAscending>(size);
            test_usm<uint32_t, sycl::usm::alloc::shared, kAscending>(size);
            // Not implemented for onesweep
            if (size <= 262144)
            {
                test_usm<uint64_t, sycl::usm::alloc::shared, kAscending>(size);
                test_usm<int64_t,  sycl::usm::alloc::shared, kAscending>(size);
            }
            test_usm<float,    sycl::usm::alloc::shared, kAscending>(size);

            test_usm<int16_t,  sycl::usm::alloc::shared, kDescending>(size);
            test_usm<uint16_t, sycl::usm::alloc::shared, kDescending>(size);
            test_usm<int,      sycl::usm::alloc::shared, kDescending>(size);
            test_usm<uint32_t, sycl::usm::alloc::shared, kDescending>(size);
            // Not implemented for onesweep
            if (size <= 262144)
            {
                test_usm<uint64_t, sycl::usm::alloc::shared, kDescending>(size);
                test_usm<int64_t,  sycl::usm::alloc::shared, kDescending>(size);
            }
            test_usm<float,    sycl::usm::alloc::shared, kDescending>(size);
        }
#endif // TEST_ALL_INPUTS
    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
