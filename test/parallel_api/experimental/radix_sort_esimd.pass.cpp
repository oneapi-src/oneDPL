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

//#ifndef LOG_TEST_INFO
#define LOG_TEST_INFO 1
//#endif

template <typename T, bool Order>
struct Compare : public std::less<T> {};

template <typename T>
struct Compare<T, false> : public std::greater<T> {};

using AscendingType  = std::true_type;
using DescendingType = std::false_type;

constexpr ::std::uint16_t WorkGroupSize = 256;

using USMAllocShared = ::std::integral_constant<sycl::usm::alloc, sycl::usm::alloc::shared>;
using USMAllocDevice = ::std::integral_constant<sycl::usm::alloc, sycl::usm::alloc::device>;

// Test dimension 1 : data per work item
template <::std::uint16_t count>
using DPWI = ::std::integral_constant<::std::uint16_t, count>;
using DataPerWorkItemListLongRun  = TestUtils::TList<DPWI<32>, DPWI<64>, DPWI<96>, DPWI<128>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<256>, DPWI<288>, DPWI<320>, DPWI<352>, DPWI<384>, DPWI<416>, DPWI<448>, DPWI<480>, DPWI<512>>;
using DataPerWorkItemListShortRun = TestUtils::TList<DPWI<32>, DPWI<64>,           DPWI<128>,            DPWI<192>,            DPWI<256>,                                             DPWI<416>,                       DPWI<512>>;

// Test dimension 2 : types
using TypeListLongRunRanges = TestUtils::TList<char, int8_t, uint8_t, int16_t, uint16_t, int, uint32_t, float, int64_t, uint64_t, double>;
using TypeListLongRunUSM    = TestUtils::TList<char, int8_t, uint8_t, int16_t, uint16_t, int, uint32_t, float, int64_t, uint64_t, double>;
using TypeListLongRunSyclIt = TestUtils::TList<char, int8_t, uint8_t, int16_t, uint16_t, int, uint32_t, float, int64_t, uint64_t, double>;
using TypeListShortRunAsc   = TestUtils::TList<char,                                     int, uint32_t, float,                    double>;
using TypeListShortRunDesc  = TestUtils::TList<                       int16_t,           int,           float,          uint64_t, double>;
using TypeListSmallSizes    = TestUtils::TList<                                               uint32_t                                  >;

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
template <typename T, typename OrderType, typename DataPerWorkItem>
void test_all_view(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << ", DataPerWorkItem = " << DataPerWorkItem::value << std::endl;
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
        oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, DataPerWorkItem::value, OrderType::value>(policy, view);
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, typename OrderType, typename DataPerWorkItem>
void test_subrange_view(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_subrange_view<T, " << OrderType::value << ">(" << size << ") : " << TypeInfo().name<T>() << ", DataPerWorkItem = " << DataPerWorkItem::value << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, OrderType::value>{});

    oneapi::dpl::experimental::ranges::views::subrange view(dt_input.get_data(), dt_input.get_data() + size);
    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, DataPerWorkItem::value, OrderType::value>(policy, view);

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

#endif // _ENABLE_RANGES_TESTING

template <typename T, typename USMAllocType, typename OrderType, typename DataPerWorkItem>
void test_usm(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", DataPerWorkItem = " << DataPerWorkItem::value << ", " << USMAllocPresentation().name<USMAllocType::value>() << ", " << OrderType::value << ">("<< size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> expected(size);
    generate_data(expected.data(), size);

    TestUtils::usm_data_transfer<USMAllocType::value, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, OrderType::value>{});

    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, DataPerWorkItem::value, OrderType::value>(policy, dt_input.get_data(), dt_input.get_data() + size);

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, typename OrderType, typename DataPerWorkItem>
void
test_usm(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_usm<T, " << OrderType::value << ">(" << size << ") : " << TypeInfo().name<T>() << ", DataPerWorkItem = " << DataPerWorkItem::value << std::endl;
#endif

    test_usm<T, USMAllocShared, OrderType, DataPerWorkItem>(size);
    test_usm<T, USMAllocDevice, OrderType, DataPerWorkItem>(size);
}

template <typename T, typename OrderType, typename DataPerWorkItem>
void test_sycl_iterators(std::size_t size)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ", DataPerWorkItem = " << DataPerWorkItem::value << ">(" << size << ");" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<T> input(size);
    generate_data(input.data(), size);
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, OrderType::value>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, DataPerWorkItem::value, OrderType::value>(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, typename OrderType, typename DataPerWorkItem>
void test_small_sizes(std::size_t /*size*/)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_small_sizes();" << std::endl;
#endif

    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    std::vector<uint32_t> ref(input);

    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, DataPerWorkItem::value, AscendingType::value>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input));
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 0");
    oneapi::dpl::experimental::esimd::radix_sort<WorkGroupSize, DataPerWorkItem::value, AscendingType::value>(policy, oneapi::dpl::begin(input), oneapi::dpl::begin(input) + 1);
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 1");
}

enum class GeneralCases
{
    eRanges = 0,
    eUSM,
    eSyclIterators
};

template <typename T, typename DataPerWorkItem>
void
test_general_cases(std::size_t size, GeneralCases kind)
{
    switch (kind)
    {
    case GeneralCases::eRanges:
#if _ENABLE_RANGES_TESTING
        test_all_view<T, AscendingType,  DataPerWorkItem>(size);
        test_all_view<T, DescendingType, DataPerWorkItem>(size);

        test_subrange_view<T, AscendingType,  DataPerWorkItem>(size);
        test_subrange_view<T, DescendingType, DataPerWorkItem>(size);
#endif // _ENABLE_RANGES_TESTING
        break;

    case GeneralCases::eUSM:
        test_usm<T, AscendingType,  DataPerWorkItem>(size);
        test_usm<T, DescendingType, DataPerWorkItem>(size);
        break;

    case GeneralCases::eSyclIterators:
        test_sycl_iterators<T, AscendingType,  DataPerWorkItem>(size);
        test_sycl_iterators<T, DescendingType, DataPerWorkItem>(size);
        break;
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <GeneralCases kind>
struct test_general_cases_runner
{
    template <typename TKey, typename DataPerWorkItem>
    static constexpr bool
    can_compile_test()
    {
        if constexpr (kind == GeneralCases::eRanges                 // Fast solution: use kind == GeneralCases::eUSM without each case check
                      || kind == GeneralCases::eUSM                 // Checked all disabled case of type + <data per work item> value
                      || kind == GeneralCases::eSyclIterators)      // Fast solution: use kind == GeneralCases::eUSM without each case check
        {
            //              32   64   96  128     160     192     224     256     288     320     352     384     416     448     480     512
            // char              N    N           N       N       N                               N               N       N       N
            // int8_t                 N           N       N       N                               N               N       N       N
            // uint8_t                N           N       N       N                               N               N       N       N
            // int16_t                N                           N                               N                               N
            // uint16_t               N                           N       N                       N                               N       N
            // int64_t      N    N    N   N       N       N       N       N       N       N       N       N       N       N       N       N
            // uint64_t     N    N    N   N       N       N       N       N       N       N       N       N       N       N       N       N
            // double       N    N    N   N       N       N       N       N       N       N       N       N       N       N       N       N
            // 
            // int          ?    ?    ?   ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?
            // uint32_t     ?    ?    ?   ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?
            // float        ?    ?    ?   ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?

            // char : <96, 160, 192, 224, 352, 416, 448, 480>
            using skip_dpwi_for_char = TestUtils::TList<DPWI<96>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<352>, DPWI<416>, DPWI<448>, DPWI<480>>;
            if constexpr (::std::is_same_v<TKey, char> &&
                            TestUtils::type_list_contain<skip_dpwi_for_char, DataPerWorkItem>())
            {
                return false;
            }

            // int8_t : <96, 160, 192, 224, 352, 416, 448, 480>
            using skip_dpwi_for_int8_t = TestUtils::TList<DPWI<96>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<352>, DPWI<416>, DPWI<448>, DPWI<480>>;
            if constexpr (::std::is_same_v<TKey, int8_t> &&
                            TestUtils::type_list_contain<skip_dpwi_for_int8_t, DataPerWorkItem>())
            {
                return false;
            }

            // uint8_t : <96, 160, 192, 224, 352, 416, 448, 480>
            using skip_dpwi_for_uint8_t = TestUtils::TList<DPWI<96>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<352>, DPWI<416>, DPWI<448>, DPWI<480>>;
            if constexpr (::std::is_same_v<TKey, uint8_t> &&
                            TestUtils::type_list_contain<skip_dpwi_for_uint8_t, DataPerWorkItem>())
            {
                return false;
            }

            // int16_t : <96, 224, 352, 480>
            using skip_dpwi_for_int16_t = TestUtils::TList<DPWI<96>, DPWI<224>, DPWI<352>, DPWI<480>>;
            if constexpr (::std::is_same_v<TKey, int16_t> &&
                          TestUtils::type_list_contain<skip_dpwi_for_int16_t, DataPerWorkItem>())
            {
                return false;
            }

            // uint16_t : <96, 224, 256, 352, 480, 512>
            using skip_dpwi_for_uint16_t = TestUtils::TList<DPWI<96>, DPWI<224>, DPWI<256>, DPWI<352>, DPWI<480>, DPWI<512>>;
            if constexpr (::std::is_same_v<TKey, uint16_t> &&
                            TestUtils::type_list_contain<skip_dpwi_for_uint16_t, DataPerWorkItem>())
            {
                return false;
            }

            // int64_t : <32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512>
            using skip_dpwi_for_int64_t = TestUtils::TList<DPWI<32>, DPWI<64>, DPWI<96>, DPWI<128>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<256>, DPWI<288>, DPWI<320>, DPWI<352>, DPWI<384>, DPWI<416>, DPWI<448>, DPWI<480>, DPWI<512>>;
            if constexpr (::std::is_same_v<TKey, int64_t> &&
                            TestUtils::type_list_contain<skip_dpwi_for_int64_t, DataPerWorkItem>())
            {
                return false;
            }

            // uint64_t : <32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512>
            using skip_dpwi_for_uint64_t = TestUtils::TList<DPWI<32>, DPWI<64>, DPWI<96>, DPWI<128>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<256>, DPWI<288>, DPWI<320>, DPWI<352>, DPWI<384>, DPWI<416>, DPWI<448>, DPWI<480>, DPWI<512>>;
            if constexpr (::std::is_same_v<TKey, uint64_t> &&
                            TestUtils::type_list_contain<skip_dpwi_for_uint64_t, DataPerWorkItem>())
            {
                return false;
            }

            // double : <32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512>
            using skip_dpwi_for_double = TestUtils::TList<DPWI<32>, DPWI<64>, DPWI<96>, DPWI<128>, DPWI<160>, DPWI<192>, DPWI<224>, DPWI<256>, DPWI<288>, DPWI<320>, DPWI<352>, DPWI<384>, DPWI<416>, DPWI<448>, DPWI<480>, DPWI<512>>;
            if constexpr (::std::is_same_v<TKey, double> &&
                            TestUtils::type_list_contain<skip_dpwi_for_double, DataPerWorkItem>())
            {
                return false;
            }
        }

        return true;
    }

    template <typename TKey, typename DataPerWorkItem>
    bool
    can_run_test(std::size_t /*size*/)
    {
        // RTE - run-time error
        // WTR - wrong test results
        // SF  - segmentation fault

        //              32      64     96    128     160     192     224     256     288     320     352     384     416     448     480     512
        // char         
        // int8_t       
        // uint8_t      
        // int16_t      
        // uint16_t     
        // int          
        // uint32_t     
        // int64_t      
        // float        
        // uint64_t     
        // double       

        //// char : <>
        //using skip_dpwi_for_char = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, char> &&
        //                TestUtils::type_list_contain<skip_dpwi_for_char, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int8_t : <>
        //using skip_dpwi_for_int8_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int8_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int8_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint8_t : <>
        //using skip_dpwi_for_uint8_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint8_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint8_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int16_t : <>
        //using skip_dpwi_for_int16_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int16_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int16_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint16_t : <>
        //using skip_dpwi_for_uint16_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint16_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint16_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int : <>
        //using skip_dpwi_for_int = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint32_t : <>
        //using skip_dpwi_for_uint32_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint32_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint32_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int64_t : <>
        //using skip_dpwi_for_int64_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int64_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int64_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint64_t : <>
        //using skip_dpwi_for_uint64_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint64_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint64_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// float : <>
        //using skip_dpwi_for_float= TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, float> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_float, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// double : <>
        //using skip_dpwi_for_double= TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, double> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_double, DataPerWorkItem>())
        //{
        //    return false;
        //}

        return true;
    }

    template <typename TKey, typename DataPerWorkItem>
    void
    run_test(std::size_t size)
    {
        test_general_cases<TKey, DataPerWorkItem>(size, kind);
    }
};

template <typename USMAllocType, typename OrderType>
struct test_usm_runner
{
    template <typename TKey, typename DataPerWorkItem>
    static constexpr bool
    can_compile_test()
    {
        //              32   64   96  128     160     192     224     256     288     320     352     384     416     448     480     512
        // char                                       N                                                       N

        // char : <192, 416>
        using skip_dpwi_for_char = TestUtils::TList<DPWI<192>, DPWI<416>>;
        if constexpr (::std::is_same_v<TKey, char> &&
                        TestUtils::type_list_contain<skip_dpwi_for_char, DataPerWorkItem>())
        {
            return false;
        }

        return true;
    }

    template <typename TKey, typename DataPerWorkItem>
    bool
    can_run_test(std::size_t /*size*/)
    {
        // RTE - run-time error
        // WTR - wrong test results
        
        //              32      64     96    128     160     192     224     256     288     320     352     384     416     448     480     512
        // char         
        // int8_t       
        // uint8_t      
        // int16_t      
        // uint16_t     
        // int          
        // uint32_t     
        // int64_t      
        // float        
        // uint64_t     
        // double       

        //// char : <>
        //using skip_dpwi_for_char = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, char> &&
        //                TestUtils::type_list_contain<skip_dpwi_for_char, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int8_t : <>
        //using skip_dpwi_for_int8_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int8_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int8_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint8_t : <>
        //using skip_dpwi_for_uint8_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint8_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint8_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int16_t : <>
        //using skip_dpwi_for_int16_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int16_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int16_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint16_t : <>
        //using skip_dpwi_for_uint16_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint16_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint16_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int : <>
        //using skip_dpwi_for_int = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint32_t : <>
        //using skip_dpwi_for_uint32_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint32_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint32_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int64_t : <>
        //using skip_dpwi_for_int64_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int64_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int64_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint64_t : <>
        //using skip_dpwi_for_uint64_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint64_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint64_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// float : <>
        //using skip_dpwi_for_float= TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, float> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_float, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// double : <>
        //using skip_dpwi_for_double= TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, double> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_double, DataPerWorkItem>())
        //{
        //    return false;
        //}

        return true;
    }

    template <typename TKey, typename DataPerWorkItem>
    void
    run_test(std::size_t size)
    {
        test_usm<TKey, USMAllocType, OrderType, DataPerWorkItem>(size);
    }
};

struct test_small_sizes_runner
{
    template <typename TKey, typename DataPerWorkItem>
    static constexpr bool
    can_compile_test()
    {
        return true;
    }

    template <typename TKey, typename DataPerWorkItem>
    bool
    can_run_test(std::size_t /*size*/)
    {
        // RTE - run-time error
        // WTR - wrong test results
        // SF  - segmentation fault

        //              32      64     96    128     160     192     224     256     288     320     352     384     416     448     480     512
        // char         
        // int8_t       
        // uint8_t      
        // int16_t      
        // uint16_t     
        // int          
        // uint32_t     
        // int64_t      
        // float        
        // uint64_t     
        // double       

        //// char : <>
        //using skip_dpwi_for_char = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, char> &&
        //                TestUtils::type_list_contain<skip_dpwi_for_char, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int8_t : <>
        //using skip_dpwi_for_int8_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int8_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int8_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint8_t : <>
        //using skip_dpwi_for_uint8_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint8_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint8_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int16_t : <>
        //using skip_dpwi_for_int16_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int16_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int16_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint16_t : <>
        //using skip_dpwi_for_uint16_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint16_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint16_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int : <>
        //using skip_dpwi_for_int = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint32_t : <>
        //using skip_dpwi_for_uint32_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint32_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint32_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// int64_t : <>
        //using skip_dpwi_for_int64_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, int64_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_int64_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// uint64_t : <>
        //using skip_dpwi_for_uint64_t = TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, uint64_t> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_uint64_t, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// float : <>
        //using skip_dpwi_for_float= TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, float> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_float, DataPerWorkItem>())
        //{
        //    return false;
        //}
        //
        //// double : <>
        //using skip_dpwi_for_double= TestUtils::TList<>;
        //if constexpr (::std::is_same_v<TKey, double> &&
        //              TestUtils::type_list_contain<skip_dpwi_for_double, DataPerWorkItem>())
        //{
        //    return false;
        //}

        return true;
    }

    template <typename TKey, typename DataPerWorkItem>
    void
    run_test(std::size_t size)
    {
        test_small_sizes<TKey, AscendingType, DataPerWorkItem>(size);
    }
};

template <typename TestRunner, typename ListOfTypes, typename DataPerWorkItemList>
void
iterate_all_params(std::size_t size)
{
    if constexpr (TestUtils::type_list_is_empty<ListOfTypes>() || TestUtils::type_list_is_empty<DataPerWorkItemList>())
    {
        return;
    }

    using TKey = typename TestUtils::GetHeadType<ListOfTypes>;
    using DataPerWorkItem = typename TestUtils::GetHeadType<DataPerWorkItemList>;

#if LOG_TEST_INFO
    std::cout << "\t\ttest for type " << TypeInfo().name<TKey>() << " and DataPerWorkItem = " << DataPerWorkItem::value << " : ";
#endif

    // Check that we are ablue to run test for the current pair <TKey, DataPerWorkItem>
    if constexpr (TestRunner::template can_compile_test<TKey, DataPerWorkItem>())
    {
        TestRunner runnerObj;
        if (runnerObj.template can_run_test<TKey, DataPerWorkItemList>(size))
        {
#if LOG_TEST_INFO
            std::cout << "starting..." << std::endl;
#endif
            // Start test for the current pair <TKey, DataPerWorkItem>
            runnerObj.template run_test<TKey, DataPerWorkItem>(size);
        }
        else
        {
#if LOG_TEST_INFO
            std::cout << "skip due run-time errors" << std::endl;
#endif
        }
    }
    else
    {
#if LOG_TEST_INFO
        std::cout << "skip due compile errors" << std::endl;
#endif
    }

    // 1. Recursive call for all rest values of DataPerWorkItem
    using RestDataPerWorkItemList = typename TestUtils::GetRestTypes<DataPerWorkItemList>;
    if constexpr (!TestUtils::type_list_is_empty<RestDataPerWorkItemList>())
    {
        iterate_all_params<TestRunner, ListOfTypes, RestDataPerWorkItemList>(size);
    }

    // 2. Recursive call for all rest key types
    using RestTypeList = typename TestUtils::GetRestTypes<ListOfTypes>;
    if constexpr (!TestUtils::type_list_is_empty<RestTypeList>())
    {
        iterate_all_params<TestRunner, RestTypeList, DataPerWorkItemList>(size);
    }
};

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
            iterate_all_params<test_general_cases_runner<GeneralCases::eRanges>,        TypeListLongRunRanges, DataPerWorkItemListLongRun>(size);
            iterate_all_params<test_general_cases_runner<GeneralCases::eUSM>,           TypeListLongRunUSM,    DataPerWorkItemListLongRun>(size);
            iterate_all_params<test_general_cases_runner<GeneralCases::eSyclIterators>, TypeListLongRunSyclIt, DataPerWorkItemListLongRun>(size);
        }
        iterate_all_params<test_small_sizes_runner, TypeListSmallSizes, DataPerWorkItemListLongRun>(1 /* this param ignored inside test_small_sizes function */);
#else
        for(auto size: sizes)
        {
            using test_runner = test_usm_runner<USMAllocShared, AscendingType>;
            iterate_all_params<test_runner, TypeListShortRunAsc,  DataPerWorkItemListShortRun>(size);
            iterate_all_params<test_runner, TypeListShortRunDesc, DataPerWorkItemListShortRun>(size);
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
