// -*- C++ -*-
//===-- esimd_radix_sort_by_key.cpp -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../support/test_config.h"
#include "../support/utils.h"
#include "esimd_radix_sort_utils.h"

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include <vector>
#include <string>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif


template<typename KeyT, typename ValueT, bool isAscending, std::uint32_t RadixBits, typename KernelParam>
void test_sycl_iterators(std::size_t size, KernelParam param)
{
    sycl::queue q = TestUtils::get_test_queue();

    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    generate_data(expected_keys.data(), size, 6);
    generate_data(expected_values.data(), size, 7);

    std::vector<KeyT> actual_keys(expected_keys);
    std::vector<ValueT> actual_values(expected_values);
    {
        sycl::buffer<KeyT> keys(actual_keys.data(), actual_keys.size());
        sycl::buffer<ValueT> values(actual_values.data(), actual_values.size());
        oneapi::dpl::experimental::kt::esimd::radix_sort_by_key<isAscending, RadixBits>(
            q, oneapi::dpl::begin(keys), oneapi::dpl::end(keys), oneapi::dpl::begin(values), param).wait();
    }

    auto expected_first  = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<isAscending>{});

    std::string parameters_msg = ", n: " + std::to_string(size) +
                                 ", sizeof(key): " + std::to_string(sizeof(KeyT)) +
                                 ", sizeof(value): " + std::to_string(sizeof(ValueT)) +
                                 ", isAscending: " +  std::to_string(isAscending) +
                                 ", RadixBits: " + std::to_string(RadixBits);
    std::string msg = "wrong results with oneapi::dpl::begin/end (keys)" + parameters_msg;
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    msg = "wrong results with oneapi::dpl::begin/end (values)" + parameters_msg;
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
}

template<typename KeyT, typename ValueT, bool isAscending, std::uint32_t RadixBits, sycl::usm::alloc _alloc_type, typename KernelParam>
void test_usm(std::size_t size, KernelParam param)
{
    sycl::queue q = TestUtils::get_test_queue();

    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    generate_data(expected_keys.data(), size, 6);
    generate_data(expected_values.data(), size, 7);

    TestUtils::usm_data_transfer<_alloc_type, KeyT> keys(q, expected_keys.begin(), expected_keys.end());
    TestUtils::usm_data_transfer<_alloc_type, ValueT> values(q, expected_values.begin(), expected_values.end());

    auto expected_first  = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<isAscending>{});

    oneapi::dpl::experimental::kt::esimd::radix_sort_by_key<isAscending, RadixBits>(
        q, keys.get_data(), keys.get_data() + size, values.get_data(), param).wait();

    std::vector<KeyT> actual_keys(size);
    std::vector<ValueT> actual_values(size);
    keys.retrieve_data(actual_keys.begin());
    values.retrieve_data(actual_values.begin());

    std::string parameters_msg = ", n: " + std::to_string(size) +
                                 ", sizeof(key): " + std::to_string(sizeof(KeyT)) +
                                 ", sizeof(value): " + std::to_string(sizeof(ValueT)) +
                                 ", isAscending: " +  std::to_string(isAscending) +
                                 ", RadixBits: " + std::to_string(RadixBits);
    std::string msg = "wrong results with USM (keys)" + parameters_msg;
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    msg = "wrong results with USM (values)" + parameters_msg;
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
}

template <std::uint16_t DataPerWorkItem, std::uint16_t WorkGroupSize>
using param_type = oneapi::dpl::experimental::kt::kernel_param<DataPerWorkItem, WorkGroupSize>;

int main()
{
    const std::vector<std::size_t> sizes = {
        6, 16, 43, 256, 316, 2048, 5072, 8192, 14001, 1<<14,
        (1<<14)+1, 50000, 67543, 100'000, 1<<17, 179'581, 250'000, 1<<18,
        (1<<18)+1, 500'000, 888'235, 1'000'000, 1<<20, 10'000'000
    };

    for(auto size: sizes)
    {
        test_usm<int16_t,  char,     Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<512, 64>{});
        test_usm<float,    uint32_t, Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<192, 64>{});
        test_usm<int32_t,  float,    Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<160, 64>{});
        test_usm<uint32_t, int32_t,  Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<160, 32>{});
        test_usm<int16_t,  uint64_t, Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<128, 64>{});
        test_usm<int64_t,  int16_t,  Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<96, 32>{});
        test_usm<uint32_t, double,   Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<64, 32>{});
        test_usm<uint32_t, uint16_t, Ascending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<32, 64>{});

        test_usm<int32_t,  uint16_t, Descending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<256, 32>{});
        test_usm<uint32_t, int32_t,  Descending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<192, 64>{});
        test_usm<float,    float,    Descending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<128, 64>{});
        test_usm<int64_t,  double,   Descending, TestRadixBits, sycl::usm::alloc::shared>(size, param_type<64, 64>{});

        test_sycl_iterators<uint32_t, int32_t, Ascending,   TestRadixBits>(size, param_type<192, 32>{});
        test_sycl_iterators<int32_t,  double,  Descending,  TestRadixBits>(size, param_type<128, 64>{});
    }
    return TestUtils::done();
}
