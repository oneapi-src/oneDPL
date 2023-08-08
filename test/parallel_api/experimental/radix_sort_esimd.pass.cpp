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
#include "support/typelist.h"

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
#include <type_traits>
#include <initializer_list>

//#ifndef LOG_TEST_INFO
#define LOG_TEST_INFO 1
//#endif

#define SKIP_RUNTIME_ERRORS 1
#define SKIP_WRONG_RESULTS  1

template <typename T, bool Order>
struct Compare : public std::less<T> {};

template <typename T>
struct Compare<T, false> : public std::greater<T> {};

using AscendingType  = std::true_type;
using DescendingType = std::false_type;

constexpr ::std::uint16_t WorkGroupSize = 256;

// Test dimension 1 : data per work item
using DPWI = ::std::uint16_t;
using DataPerWorkItems = ::std::initializer_list<DPWI>;

//#define TEST_DPWI 32, 64, 96, 128, 160
#define TEST_DPWI 192, 224, 256, 288, 320
//#define TEST_DPWI  352, 384, 416, 448, 480, 512

#ifdef TEST_DPWI
#   define DataPerWorkItemsLongRun  TEST_DPWI
#   define DataPerWorkItemsShortRun TEST_DPWI
#else
#   define DataPerWorkItemsLongRun  32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512
// TODO required to exclude some DataPerWorkItem values later
#   define DataPerWorkItemsShortRun 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512
#endif // TEST_DPWI

#define TEST_DATA_TYPE double

// Test dimension 2 : types
#ifdef TEST_DATA_TYPE
using TypeListLongRunRanges = TestUtils::TList<TEST_DATA_TYPE>;
using TypeListLongRunUSM    = TestUtils::TList<TEST_DATA_TYPE>;
using TypeListLongRunSyclIt = TestUtils::TList<TEST_DATA_TYPE>;
using TypeListShortRunAsc   = TestUtils::TList<TEST_DATA_TYPE>;
using TypeListShortRunDesc  = TestUtils::TList<TEST_DATA_TYPE>;
using TypeListSmallSizes    = TestUtils::TList<TEST_DATA_TYPE>;
#else
using TypeListLongRunRanges = TestUtils::TList<char, int8_t, uint8_t, int16_t, uint16_t, int, uint32_t, float, int64_t, uint64_t, double>;
using TypeListLongRunUSM    = TestUtils::TList<char, int8_t, uint8_t, int16_t, uint16_t, int, uint32_t, float, int64_t, uint64_t, double>;
using TypeListLongRunSyclIt = TestUtils::TList<char, int8_t, uint8_t, int16_t, uint16_t, int, uint32_t, float, int64_t, uint64_t, double>;
using TypeListShortRunAsc   = TestUtils::TList<char,                                     int, uint32_t, float,                    double>;
using TypeListShortRunDesc  = TestUtils::TList<                       int16_t,           int,           float,          uint64_t, double>;
using TypeListSmallSizes    = TestUtils::TList<                                               uint32_t                                  >;
#endif // TEST_DATA_TYPE

// test types :           char, int8_t,      uint8_t,       int16_t, uint16_t,       int, uint32_t,     float, int64_t, uint64_t,      double
// compiler named types : char, signed char, unsigned char, short,   unsigned short, int, unsigned int, float, long,    unsigned long, double

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
template <typename T, typename OrderType, DPWI dpwi>
void test_all_view(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << ", DataPerWorkItem = " << dpwi << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, OrderType::value>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, dpwi, OrderType::value>(policy, view);
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, typename OrderType, DPWI dpwi>
void test_subrange_view(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_subrange_view<T, " << OrderType::value << ">(" << size << ") : " << TypeInfo().name<T>() << ", DataPerWorkItem = " << dpwi << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, OrderType::value>{});

    oneapi::dpl::experimental::ranges::views::subrange view(dt_input.get_data(), dt_input.get_data() + size);
    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, dpwi, OrderType::value>(policy, view);

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

#endif // _ENABLE_RANGES_TESTING

template <typename T, sycl::usm::alloc USMAllocType, typename OrderType, DPWI dpwi>
void test_usm(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", DataPerWorkItem = " << dpwi << ", " << USMAllocPresentation().name<USMAllocType>() << ", " << OrderType::value << ">("<< size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<USMAllocType, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, OrderType::value>{});

    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, dpwi, OrderType::value>(policy, dt_input.get_data(), dt_input.get_data() + size);

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, typename OrderType, DPWI dpwi>
void
test_usm(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<T, " << OrderType::value << ">(" << size << ") : " << TypeInfo().name<T>() << ", DataPerWorkItem = " << dpwi << std::endl;
#endif

    test_usm<T, sycl::usm::alloc::shared, OrderType, dpwi>(size);
    test_usm<T, sycl::usm::alloc::device, OrderType, dpwi>(size);
}

template <typename T, typename OrderType, DPWI dpwi>
void test_sycl_iterators(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ", DataPerWorkItem = " << dpwi << ">(" << size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, OrderType::value>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, dpwi, OrderType::value>(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, typename OrderType, DPWI dpwi>
void test_small_sizes(std::size_t /*size*/)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_small_sizes();" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    std::vector<uint32_t> ref(input);

    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, dpwi, AscendingType::value>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input));
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 0");
    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, dpwi, AscendingType::value>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input) + 1);
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 1");
}

enum class GeneralCases
{
    eRanges = 0,
    eUSM,
    eSyclIterators
};

template <typename T, DPWI dpwi>
void
test_general_cases(std::size_t size, GeneralCases kind)
{
    switch (kind)
    {
    case GeneralCases::eRanges:
#if _ENABLE_RANGES_TESTING
        test_all_view<T, AscendingType,  dpwi>(size);
        test_all_view<T, DescendingType, dpwi>(size);

        test_subrange_view<T, AscendingType,  dpwi>(size);
        test_subrange_view<T, DescendingType, dpwi>(size);
#endif // _ENABLE_RANGES_TESTING
        break;

    case GeneralCases::eUSM:
        test_usm<T, AscendingType,  dpwi>(size);
        test_usm<T, DescendingType, dpwi>(size);
        break;

    case GeneralCases::eSyclIterators:
        test_sycl_iterators<T, AscendingType,  dpwi>(size);
        test_sycl_iterators<T, DescendingType, dpwi>(size);
        break;
    }
}

template <GeneralCases kind>
struct test_general_cases_runner
{
    template <typename TKey, DPWI dpwi>
    bool
    can_run_test(std::size_t /*size*/)
    {
        // RTE - run-time error
        // WTR - wrong test results
        // SF  - segmentation fault
        // H   - hang

        //              32      64     96    128     160     192     224     256     288     320     352     384     416     448     480     512
        // char         
        // int8_t       
        // uint8_t      
        // int16_t      H       H            H       H       H
        // uint16_t     H       H            H       H       H
        // int          
        // uint32_t     
        // int64_t      
        // uint64_t     
        // float        
        // double       

        // char : <>

        // int8_t : <>

        // uint8_t : <32, 64, 128>

        // int16_t : <32, 64, 128, 160, 192>

        // uint16_t : <32, 64, 128, 160, 192>

        return true;
    }

    template <typename TKey, DPWI dpwi>
    void
    run_test(std::size_t size)
    {
        test_general_cases<TKey, dpwi>(size, kind);
    }
};

template <DPWI required_dpwi, typename Pred>
inline bool
check_dpwi_size_if(DPWI dpwi, std::size_t size, Pred pred)
{
    return required_dpwi == dpwi && pred(size);
}

template <DPWI required_dpwi>
inline bool
check_dpwi_size(DPWI /*dpwi*/, std::size_t /*size*/)
{
    return false;
}

template <DPWI required_dpwi, std::size_t required_size, std::size_t... rest_of_required_sizes>
inline bool
check_dpwi_size(DPWI dpwi, std::size_t size)
{
    return (required_dpwi == dpwi && required_size == size)
        || check_dpwi_size<required_dpwi, rest_of_required_sizes...>(dpwi, size);
}


template <sycl::usm::alloc USMAllocType, typename OrderType>
struct test_usm_runner
{
    template <typename TKey, DPWI dpwi>
    bool
    can_run_test(std::size_t size)
    {
        //              32      64     96    128     160     192     224     256     288     320     352     384     416     448     480     512
        // char                 N            N               N                                                                               N
        // int8_t               N            N               N                                                                               N
        // uint8_t              N            N               N                                                                               N
        // int16_t              N            N               N               N                                       N                       N
        // uint16_t             N            N               N               N                                       N                       N
        // int          N       N      N     N       N       N       N       N       N       N                       N                       N
        // uint32_t     N       N      N     N       N       N       N       N       N       N                       N                       N
        // int64_t      
        // uint64_t     
        // float        
        // double       

        // char, int8_t, uint8_t - runtime errors
#if SKIP_RUNTIME_ERRORS
        if ((::std::is_same_v<TKey, char> || ::std::is_same_v<TKey, int8_t> || ::std::is_same_v<TKey, uint8_t>)
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                                     |                     one_sweep                     |
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 64,           5072, 14001                                                                                                                              >(dpwi, size) ||
                check_dpwi_size<128,           8192, 14001, 16384                                                                                                                       >(dpwi, size)))
            return false;
#endif // SKIP_RUNTIME_ERRORS

        // char, int8_t, uint8_t - wrong test results
#if SKIP_WRONG_RESULTS
        if ((::std::is_same_v<TKey, char> || ::std::is_same_v<TKey, int8_t> || ::std::is_same_v<TKey, uint8_t>)
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                                     |                     one_sweep                     |
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 64,                  8192, 16384,                      16385, 50000, 100000, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<128,                        16384                                                                                                                       >(dpwi, size) ||
                check_dpwi_size<512,                                                                                                  262145, 500000, 888235, 1000000, 1048576, 10000000>(dpwi, size)))
            return false;
#endif // SKIP_WRONG_RESULTS

        // int16_t, uint16_t - runtime errors
#if SKIP_RUNTIME_ERRORS
        if ((::std::is_same_v<TKey, int16_t> || ::std::is_same_v<TKey, uint16_t>)
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                                     |                     one_sweep                     |
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 64,           5072, 14001, 16384                                                                                                                       >(dpwi, size) ||
                check_dpwi_size<128,                 14001, 16384                                                                                                                       >(dpwi, size) ||
                check_dpwi_size<192,                 14001, 16384                                                                                                                       >(dpwi, size) ||
                check_dpwi_size<256,           8192, 14001, 16384                                                                                                                       >(dpwi, size) ||
                check_dpwi_size<416,           8192, 14001, 16384                                                                                                                       >(dpwi, size) ||
                check_dpwi_size<512,                 14001, 16384                                                                                                                       >(dpwi, size)))
            return false;
#endif // SKIP_RUNTIME_ERRORS

        // int16_t, uint16_t - wrong test results
#if SKIP_WRONG_RESULTS
        if ((::std::is_same_v<TKey, int16_t> || ::std::is_same_v<TKey, uint16_t>)
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                                     |                     one_sweep                     |
            // +-------------------+-----------------------------+------------------------------------------------------------------+---------------------------------------------------+
            &&(check_dpwi_size< 64,            8192,              16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                           >(dpwi, size) ||
                check_dpwi_size<192,                              16385                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<256,                              16385, 50000                                                                                                          >(dpwi, size) ||
                check_dpwi_size<416,                              16385, 50000, 67543,        100000,         179581                                                                    >(dpwi, size) ||
                check_dpwi_size<512,                              16385,        67543, 50000, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size))) 
            return false;
#endif // SKIP_WRONG_RESULTS

        // int, uint32_t - runtime errors
#if SKIP_RUNTIME_ERRORS
        if ((::std::is_same_v<TKey, int> || ::std::is_same_v<TKey, uint32_t>)
            // +-------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                               |                     one_sweep                     |
            // +-------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 64,     5072,       14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size< 96,           8192                                                                                                                               >(dpwi, size) ||
                check_dpwi_size<128,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<160,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<192,           8192, 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<224,                        16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<256,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<416,           8192, 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<512,                 14001, 16384                                                                                                                 >(dpwi, size)))
            return false;
#endif // SKIP_RUNTIME_ERRORS

        // int, uint32_t - wrong test results
#if SKIP_WRONG_RESULTS
        if ((::std::is_same_v<TKey, int> || ::std::is_same_v<TKey, uint32_t>)
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                               |                     one_sweep                     |
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 32,     5072, 8192, 14001, 16384, 16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size< 64,           8192,               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<160,                               16385, 50000                                                                                                   >(dpwi, size) ||
                check_dpwi_size<192,                               16385, 50000, 67543, 100000, 131072, 179581,         262144                                                    >(dpwi, size) ||
                check_dpwi_size<224,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<256,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<288,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<320,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<352,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<384,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<416,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<448,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<480,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<512,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144, 262145, 500000, 888235, 1000000, 1048576, 10000000>(dpwi, size)))
            return false;
#endif // SKIP_WRONG_RESULTS

        // float - runtime errors
#if SKIP_RUNTIME_ERRORS
        if ((::std::is_same_v<TKey, float>)
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                               |                     one_sweep                     |
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            && (//  32
                check_dpwi_size< 64,     5072,       14001                                                                                                                        >(dpwi, size) ||
                check_dpwi_size< 96,           8192                                                                                                                               >(dpwi, size) ||
                check_dpwi_size<128,                 14001                                                                                                                        >(dpwi, size) ||
                check_dpwi_size<160,                        16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<192,                        16384                                                                                                                 >(dpwi, size)))
            return false;
#endif // SKIP_RUNTIME_ERRORS

        // float - wrong test results
#if SKIP_WRONG_RESULTS
        if (::std::is_same_v<TKey, float>
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                               |                     one_sweep                     |
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 32,     5072, 8192, 14001, 16384, 16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size< 64,           8192,        16384, 16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size< 96,                 14001, 16384,        50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<128,                        16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<160,                 14001,        16385                                                                                                          >(dpwi, size) ||
                check_dpwi_size<192,                 14001,        16385, 50000, 67543, 100000, 131072, 179581,         262144                                                    >(dpwi, size) ||
                check_dpwi_size<224,                        16384, 16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<256,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<288,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<320,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<352,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<384,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<416,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<448,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<480,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                    >(dpwi, size) ||
                check_dpwi_size<512,                               16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144, 262145, 500000, 888235, 1000000, 1048576, 10000000>(dpwi, size)))
            return false;
#endif // SKIP_WRONG_RESULTS

        // double - runtime errors
#if SKIP_RUNTIME_ERRORS
        if ((::std::is_same_v<TKey, double>)
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                               |                     one_sweep                     |
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 32,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size< 64,     5072,       14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size< 96,           8192, 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<128,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<160,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<192,                 14001, 16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size<224,                        16384                                                                                                                 >(dpwi, size) ||
                check_dpwi_size_if<256                                                                                        /* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */ >(dpwi, size, [](std::size_t size) { return 262145 <= size; }) ||
                check_dpwi_size_if<288 /* XXXXXXXXXXXXXXXXXXXX */                                                                                                                 >(dpwi, size, [](std::size_t size) { return size <= 16384; }) ||
                check_dpwi_size_if<320 /* XXXXXXXXXXXXXXXXXXXX */                                                                                                                 >(dpwi, size, [](std::size_t size) { return size <= 16384; }) ||
                check_dpwi_size<352,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<384,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<416,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<448,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<480,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<512,  0                                                                                                                                           >(dpwi, size)))

            return false;
#endif // SKIP_RUNTIME_ERRORS

        // double - wrong test results
#if SKIP_WRONG_RESULTS
        if (::std::is_same_v<TKey, double>
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            // |  DataPweWorkItem  |           one_wg            |                  cooperative                               |                     one_sweep                     |
            // --------------------+-----------------------------+------------------------------------------------------------+---------------------------------------------------+
            && (check_dpwi_size< 32,     5072, 8192,              16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                     >(dpwi, size) ||
                check_dpwi_size< 64,           8192,              16385, 50000, 67543, 100000, 131072, 179581, 250000, 262144                                                     >(dpwi, size) ||
                check_dpwi_size< 96,                                     50000, 67543, 100000, 131072, 179581, 250000, 262144                                                     >(dpwi, size) ||
                check_dpwi_size<128,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<160,                                                                           250000                                                             >(dpwi, size) ||
                check_dpwi_size<192,                              16385, 50000, 67543,                         250000, 262144                                                     >(dpwi, size) ||
                check_dpwi_size<224,                              16385, 50000, 67543,                         250000, 262144                                                     >(dpwi, size) ||
                check_dpwi_size<256,                              16385, 50000, 67543,                         250000, 262144                                                     >(dpwi, size) ||
                check_dpwi_size<288,                              16385, 50000, 67543, 100000,                 250000, 262144, 262145, 500000, 888235, 1000000, 1048576, 10000000 >(dpwi, size) ||
                check_dpwi_size<320,                              16385, 50000, 67543, 100000, 179581,         250000, 262144, 262145, 500000, 888235, 1000000, 1048576, 10000000 >(dpwi, size) ||
                check_dpwi_size<352,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<384,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<416,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<448,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<480,  0                                                                                                                                           >(dpwi, size) ||
                check_dpwi_size<512,  0                                                                                                                                           >(dpwi, size)))
            return false;
#endif // SKIP_WRONG_RESULTS

        return true;
    }

    template <typename TKey, DPWI dpwi>
    void
    run_test(std::size_t size)
    {
        test_usm<TKey, USMAllocType, OrderType, dpwi>(size);
    }
};

struct test_small_sizes_runner
{
    template <typename TKey, DPWI dpwi>
    bool
    can_run_test(std::size_t /*size*/)
    {
        return true;
    }

    template <typename TKey, DPWI dpwi>
    void
    run_test(std::size_t size)
    {
        test_small_sizes<TKey, AscendingType, dpwi>(size);
    }
};

template <typename TestRunner, typename ListOfTypes>
void
iterate_all_params(std::size_t size)
{
    // Implementation not requierd because dpwiItem template param is absent
}

template <typename TestRunner, typename ListOfTypes, DPWI dpwiItem, DPWI... dpwiVariants>
void
iterate_all_params(std::size_t size)
{
    if constexpr (TestUtils::type_list_is_empty<ListOfTypes>())
    {
        return;
    }

    using TKey = typename TestUtils::GetHeadType<ListOfTypes>;

#if LOG_TEST_INFO
    std::cout << "\t\ttest for type " << TypeInfo().name<TKey>() << ", DataPerWorkItem = " << dpwiItem << ", size = " << size << " : ";
#endif

    {
        TestRunner runnerObj;
        if (runnerObj.template can_run_test<TKey, dpwiItem>(size))
        {
#if LOG_TEST_INFO
            std::cout << "starting..." << std::endl;
#endif
            // Start test for the current pair <TKey, dpwiItem>
            runnerObj.template run_test<TKey, dpwiItem>(size);
        }
        else
        {
#if LOG_TEST_INFO
            std::cout << "skip due run-time errors" << std::endl;
#endif
        }
    }

    // 1. Recursive call for all rest key types
    using RestTypeList = typename TestUtils::GetRestTypes<ListOfTypes>;
    if constexpr (!TestUtils::type_list_is_empty<RestTypeList>())
    {
        iterate_all_params<TestRunner, RestTypeList, dpwiItem>(size);
    }

    // 2. Recursive call for all rest DataPerWorkItem's values
    iterate_all_params<TestRunner, ListOfTypes, dpwiVariants...>(size);
};

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
#if TEST_LONG_RUN
        for(auto size: sizes)
        {
            iterate_all_params<test_general_cases_runner<GeneralCases::eRanges>,        TypeListLongRunRanges, DataPerWorkItemsLongRun>(size);
            iterate_all_params<test_general_cases_runner<GeneralCases::eUSM>,           TypeListLongRunUSM,    DataPerWorkItemsLongRun>(size);
            iterate_all_params<test_general_cases_runner<GeneralCases::eSyclIterators>, TypeListLongRunSyclIt, DataPerWorkItemsLongRun>(size);
        }
        iterate_all_params<test_small_sizes_runner, TypeListSmallSizes, DataPerWorkItemsLongRun>(1 /* this param ignored inside test_small_sizes function */);
#else
        for(auto size: sizes)
        {
            iterate_all_params<test_usm_runner<sycl::usm::alloc::shared, AscendingType>,  TypeListShortRunAsc,  DataPerWorkItemsShortRun>(size);
            iterate_all_params<test_usm_runner<sycl::usm::alloc::shared, DescendingType>, TypeListShortRunDesc, DataPerWorkItemsShortRun>(size);
        }
#endif // TEST_LONG_RUN
    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
