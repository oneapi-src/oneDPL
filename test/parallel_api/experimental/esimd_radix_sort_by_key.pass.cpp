// -*- C++ -*-
//===-- rasix_sort_esimd.pass.cpp -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include <algorithm>
#include <numeric>
#include <vector>
#include <random>
#include <string>
#include <limits>
#include <cmath>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

constexpr bool Ascending = true;
constexpr bool Descending = false;

template<bool Order>
struct CompareKey
{
    template<typename T, typename U>
    bool operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }
};

template<>
struct CompareKey<false>
{
    template<typename T, typename U>
    bool operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) > std::get<0>(rhs);
    }
};

template <typename T>
typename ::std::enable_if_t<std::is_arithmetic_v<T>, void>
generate_data(T* input, std::size_t size, std::uint32_t seed)
{
    std::default_random_engine gen{seed};
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

template<typename KeyT, typename ValueT, std::uint32_t WorkGroupSize, std::uint32_t DataPerWorkItem, bool Order,
         std::uint32_t RadixBits>
void test_sycl_iterators(std::size_t size)
{
    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    generate_data(expected_keys.data(), size, 6);
    generate_data(expected_values.data(), size, 7);

    std::vector<KeyT> actual_keys(expected_keys);
    std::vector<ValueT> actual_values(expected_values);
    {
        sycl::buffer<KeyT> keys(actual_keys.data(), actual_keys.size());
        sycl::buffer<ValueT> values(actual_values.data(), actual_values.size());
        oneapi::dpl::experimental::esimd::radix_sort_by_key<WorkGroupSize, DataPerWorkItem, Order, RadixBits>(
            policy, oneapi::dpl::begin(keys), oneapi::dpl::end(keys), oneapi::dpl::begin(values));
    }

    auto expected_first  = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<Order>{});


    std::string parameters_msg = ", n: " + std::to_string(size) +
                                 ", sizeof(key): " + std::to_string(sizeof(KeyT)) +
                                 ", sizeof(value): " + std::to_string(sizeof(ValueT)) +
                                 ", WorkGroupSize: " +  std::to_string(WorkGroupSize) +
                                 ", DataPerWorkItem: " +  std::to_string(DataPerWorkItem) +
                                 ", Order: " +  std::to_string(Order) +
                                 ", RadixBits: " + std::to_string(RadixBits);
    std::string msg = "wrong results with oneapi::dpl::begin/end (keys)" + parameters_msg;
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    msg = "wrong results with oneapi::dpl::begin/end (values)" + parameters_msg;
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
}

template<typename KeyT, typename ValueT, std::uint32_t WorkGroupSize, std::uint32_t DataPerWorkItem, bool Order,
         std::uint32_t RadixBits, sycl::usm::alloc _alloc_type>
void test_usm(std::size_t size)
{
    sycl::queue q = TestUtils::get_test_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    generate_data(expected_keys.data(), size, 6);
    generate_data(expected_values.data(), size, 7);

    TestUtils::usm_data_transfer<_alloc_type, KeyT> keys(q, expected_keys.begin(), expected_keys.end());
    TestUtils::usm_data_transfer<_alloc_type, ValueT> values(q, expected_values.begin(), expected_values.end());

    auto expected_first  = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<Order>{});

    oneapi::dpl::experimental::esimd::radix_sort_by_key<WorkGroupSize, DataPerWorkItem, Order, RadixBits>(
        policy, keys.get_data(), keys.get_data() + size, values.get_data());

    std::vector<KeyT> actual_keys(size);
    std::vector<ValueT> actual_values(size);
    keys.retrieve_data(actual_keys.begin());
    values.retrieve_data(actual_values.begin());

    std::string parameters_msg = ", n: " + std::to_string(size) +
                                 ", sizeof(key): " + std::to_string(sizeof(KeyT)) +
                                 ", sizeof(value): " + std::to_string(sizeof(ValueT)) +
                                 ", WorkGroupSize: " +  std::to_string(WorkGroupSize) +
                                 ", DataPerWorkItem: " +  std::to_string(DataPerWorkItem) +
                                 ", Order: " +  std::to_string(Order) +
                                 ", RadixBits: " + std::to_string(RadixBits);
    std::string msg = "wrong results with USM (keys)" + parameters_msg;
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    msg = "wrong results with USM (values)" + parameters_msg;
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
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

    for(auto size: sizes)
    {
        test_usm<int16_t,  char,    128,  256, true,  8, sycl::usm::alloc::shared>(size);
        test_usm<float,    uint32_t, 64,  192, true,  8, sycl::usm::alloc::shared>(size);
        test_usm<int32_t,  float,    64,  192, true,  8, sycl::usm::alloc::shared>(size);
        test_usm<uint32_t, int32_t,  64,  160, true,  8, sycl::usm::alloc::shared>(size);
        test_usm<int16_t,  uint64_t, 64,  128, true,  8, sycl::usm::alloc::shared>(size);
        test_usm<uint32_t, double,   64,  96,  true,  8, sycl::usm::alloc::shared>(size);
        test_usm<int64_t,  int16_t,  32,  64,  true,  8, sycl::usm::alloc::shared>(size);
        test_usm<uint32_t, uint16_t, 64,  32,  true,  8, sycl::usm::alloc::shared>(size);

        test_usm<int32_t,  uint16_t, 32,  256, false, 8, sycl::usm::alloc::shared>(size);
        test_usm<uint32_t, int32_t,  64,  192, false, 8, sycl::usm::alloc::shared>(size);
        test_usm<float,    float,    64,  128, false, 8, sycl::usm::alloc::shared>(size);
        test_usm<int64_t,  double,   64,  64,  false, 8, sycl::usm::alloc::shared>(size);

        test_sycl_iterators<uint32_t, int32_t, 64,  192, true,   8>(size);
        test_sycl_iterators<int32_t,  double,  48,  128, false,  8>(size);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}