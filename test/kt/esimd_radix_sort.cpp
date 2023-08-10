// -*- C++ -*-
//===-- esimd_radix_sort.cpp -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

#include "../support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "../support/sycl_alloc_utils.h"

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

template <typename T, bool Order>
struct Compare : public std::less<T> {};

template <typename T>
struct Compare<T, false> : public std::greater<T> {};

constexpr bool Ascending = true;
constexpr bool Descending = false;

#ifndef TEST_DATA_TYPE
#define TEST_DATA_TYPE int
#endif

#ifndef TEST_DATA_PER_WORK_ITEM
#define TEST_DATA_PER_WORK_ITEM 256
#endif

#ifndef TEST_WORK_GROUP_SIZE
#define TEST_WORK_GROUP_SIZE 64
#endif

constexpr std::uint8_t RadixBits = 8;

using ParamType = oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE>;
constexpr ParamType kernel_parameters;

#if LOG_TEST_INFO
struct TypeInfo
{
    template <typename T>
    const std::string& name()
    {
        static const std::string TypeName = "unknown type name";
        return TypeName;
    }

    template <>
    const std::string& name<char>()
    {
        static const std::string TypeName = "char";
        return TypeName;
    }

    template <>
    const std::string& name<int8_t>()
    {
        static const std::string TypeName = "int8_t";
        return TypeName;
    }

    template <>
    const std::string& name<uint8_t>()
    {
        static const std::string TypeName = "uint8_t";
        return TypeName;
    }

    template <>
    const std::string& name<int16_t>()
    {
        static const std::string TypeName = "int16_t";
        return TypeName;
    }

    template <>
    const std::string& name<uint16_t>()
    {
        static const std::string TypeName = "uint16_t";
        return TypeName;
    }

    template <>
    const std::string& name<uint32_t>()
    {
        static const std::string TypeName = "uint32_t";
        return TypeName;
    }

    template <>
    const std::string& name<uint64_t>()
    {
        static const std::string TypeName = "uint64_t";
        return TypeName;
    }

    template <>
    const std::string& name<int64_t>()
    {
        static const std::string TypeName = "int64_t";
        return TypeName;
    }

    template <>
    const std::string& name<int>()
    {
        static const std::string TypeName = "int";
        return TypeName;
    }

    template <>
    const std::string& name<float>()
    {
        static const std::string TypeName = "float";
        return TypeName;
    }

    template <>
    const std::string& name<double>()
    {
        static const std::string TypeName = "double";
        return TypeName;
    }
};

struct USMAllocPresentation
{
    template <sycl::usm::alloc>
    const std::string& name()
    {
        static const std::string USMAllocTypeName = "unknown";
        return USMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::host>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::host";
        return USMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::device>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::device";
        return USMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::shared>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::shared";
        return USMAllocTypeName;
    }

    template <>
    const std::string& name<sycl::usm::alloc::unknown>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::unknown";
        return USMAllocTypeName;
    }
};
#endif // LOG_TEST_INFO

template <typename T>
typename ::std::enable_if_t<std::is_arithmetic_v<T>, void>
generate_data(T* input, std::size_t size)
{
    std::default_random_engine gen{42};
    std::size_t unique_threshold = 75 * size / 100;
    if constexpr (sizeof(T) < sizeof(short)) // no uniform_int_distribution for chars
    {
        std::uniform_int_distribution<int> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
        std::generate(input, input + unique_threshold, [&]{ return T(dist(gen)); });
    }
    else if constexpr (std::is_integral_v<T>)
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
template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void test_all_view(std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, IsAscending>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(policy, view, param).wait();
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void test_subrange_view(std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_subrange_view<T, " << IsAscending << ">(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, IsAscending>{});

    oneapi::dpl::experimental::ranges::views::subrange view(dt_input.get_data(), dt_input.get_data() + size);
    oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(policy, view, param).wait();

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

#endif // _ENABLE_RANGES_TESTING

template <typename T, bool IsAscending, std::uint8_t RadixBits, sycl::usm::alloc _alloc_type, typename KernelParam>
void test_usm(std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>() << ", " << IsAscending << ">("<< size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<_alloc_type, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, IsAscending>{});

    oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(policy, dt_input.get_data(), dt_input.get_data() + size, param).wait();

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void test_sycl_iterators(std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, IsAscending>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), param).wait();
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void test_small_sizes(KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_small_sizes();" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    std::vector<uint32_t> ref(input);

    oneapi::dpl::experimental::kt::esimd::radix_sort<Ascending, RadixBits>(
        policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input), param).wait();
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 0");

    oneapi::dpl::experimental::kt::esimd::radix_sort<Ascending, RadixBits>(
        policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input) + 1, param).wait();
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 1");
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void test_general_cases(std::size_t size, KernelParam param)
{
#if _ENABLE_RANGES_TESTING
    test_all_view<T, IsAscending, RadixBits>(size, param);
    test_subrange_view<T, IsAscending, RadixBits>(size, param);
#endif // _ENABLE_RANGES_TESTING
    test_usm<T, IsAscending, RadixBits, sycl::usm::alloc::shared>(size, param);
    test_usm<T, IsAscending, RadixBits, sycl::usm::alloc::device>(size, param);
    test_sycl_iterators<T, IsAscending, RadixBits>(size, param);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename T, typename KernelParam>
bool
can_run_test(KernelParam param)
{
    sycl::queue q = TestUtils::get_test_queue();

    const ::std::size_t __max_slm_size = q.get_device().template get_info<sycl::info::device::local_mem_size>();

    // skip tests with error: LLVM ERROR: SLM size exceeds target limits
    return sizeof(T) * param.data_per_workitem * param.workgroup_size < __max_slm_size;
}

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const bool _can_run_test = can_run_test<TEST_DATA_TYPE>(kernel_parameters);
    if (_can_run_test)
    {
        const std::vector<std::size_t> sizes = {
            1, 6, 16, 43, 256, 316, 2048, 5072, 8192, 14001, 1<<14,
            (1<<14)+1, 50000, 67543, 100'000, 1<<17, 179'581, 250'000, 1<<18,
            (1<<18)+1, 500'000, 888'235, 1'000'000, 1<<20, 10'000'000
        };

        try
        {
#if TEST_LONG_RUN
            for(auto size: sizes)
            {
                test_general_cases<TEST_DATA_TYPE, Ascending, RadixBits>(size, kernel_parameters);
                test_general_cases<TEST_DATA_TYPE, Descending, RadixBits>(size, kernel_parameters);
            }
            test_small_sizes<TEST_DATA_TYPE, Ascending, RadixBits>(kernel_parameters);
#else
            for(auto size: sizes)
            {
                test_usm<TEST_DATA_TYPE, Ascending, RadixBits, sycl::usm::alloc::shared>(size, kernel_parameters);
                test_usm<TEST_DATA_TYPE, Descending, RadixBits, sycl::usm::alloc::shared>(size, kernel_parameters);
            }
#endif // TEST_LONG_RUN
        }
        catch (const ::std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && _can_run_test);
}
