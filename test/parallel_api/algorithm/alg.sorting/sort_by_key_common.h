// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _SORT_BY_KEY_COMMON_H
#define _SORT_BY_KEY_COMMON_H

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#if TEST_DPCPP_BACKEND_PRESENT
#   include "support/sycl_alloc_utils.h"
#endif

#include <vector>
#include <algorithm>
#include <type_traits>

struct StableSort{};
struct UnstableSort{};

struct AscendingSort{};
struct DiscendingSort{};

template<typename TestType, typename StabilityTag, typename DirectionTag>
struct KernelName{};

template<typename _KeyIt, typename _ValIt, typename _Size, typename DirectionTag>
void
generate_data(_KeyIt keys_begin, _ValIt vals_begin, _Size n, DirectionTag direction_tag)
{
    if constexpr (std::is_same_v<DirectionTag, AscendingSort>)
    {
        // Generated example for n = 10:
        // Keys:    1 2 1 2 1 2 1 2 1 2
        // Values:  0 1 2 3 4 5 6 7 8 9
        //
        // Sorted example for n = 10, stable sort:
        // Keys:    1 1 1 1 1 2 2 2 2 2
        // Values:  0 2 4 6 8 1 3 5 7 9
        auto counting_begin = oneapi::dpl::counting_iterator<int>{0};
        std::transform(counting_begin, counting_begin + n, keys_begin,
                    [](int i) { return i % 2 + 1; });
        std::copy(counting_begin, counting_begin + n, vals_begin);
    }
    else
    {
        // Generated example for n = 10:
        // Keys:    0 10 20 30 40 50 60 70 80 90
        // Values:  9  8  7  6  5  4  3  2  1  0
        //
        // Sorted example for n = 10, stable sort:
        // Keys:    90 80 70 60 50 40 30 20 10  0
        // Values:   0  1  2  3  4  5  6  7  8  9
        for (int i = 0; i < n; i++)
        {
            keys_begin[i] = i * 10;
            vals_begin[i] = n - i - 1;
        }
    }
}

template<typename _Policy, typename _KeyIt, typename _ValIt, typename _Size, typename StabilityTag, typename DirectionTag>
void
call_sort(_Policy&& policy, _KeyIt keys_begin, _ValIt vals_begin, _Size n, StabilityTag, DirectionTag)
{
    if constexpr (std::is_same_v<DirectionTag, AscendingSort>)
    {
        // Do not pass the comparator to check the API with a default comparator; radix sort may be used
        if constexpr (std::is_same_v<StabilityTag, StableSort>)
            oneapi::dpl::stable_sort_by_key(policy, keys_begin, keys_begin + n, vals_begin);
        else
            oneapi::dpl::sort_by_key(policy, keys_begin, keys_begin + n, vals_begin);
    }
    else
    {
        // Pass a custom comparator to check the corresponding implementation; merge-sort may be used
        auto greater = [](const auto& lhs, const auto& rhs) { return lhs > rhs; };
        if constexpr (std::is_same_v<StabilityTag, StableSort>)
            oneapi::dpl::stable_sort_by_key(policy, keys_begin, keys_begin + n, vals_begin, greater);
        else
            oneapi::dpl::sort_by_key(policy, keys_begin, keys_begin + n, vals_begin, greater);
    }
}

template<typename _KeysIt, typename _ValsIt, typename _Size, typename StabilityTag, typename DirectionTag>
void
check_sort(const _KeysIt& keys_begin, const _ValsIt& vals_begin, _Size n, StabilityTag, DirectionTag)
{
    if constexpr (std::is_same_v<DirectionTag, AscendingSort>)
    {
        const int k = (n - 1) / 2 + 1;
        for (int i = 0; i < n; ++i)
        {
            if(i - k < 0)
            {
                //a key should be 1 and value should be even
                EXPECT_TRUE(keys_begin[i] == 1 && vals_begin[i] % 2 == 0, "wrong result with a standard policy");
            }
            else
            {
                //a key should be 2 and value should be odd
                EXPECT_TRUE(keys_begin[i] == 2 && vals_begin[i] % 2 == 1, "wrong result with a standard policy");
            }
        }

        if constexpr (std::is_same_v<StabilityTag, StableSort>)
        {
            EXPECT_TRUE(std::is_sorted(vals_begin, vals_begin + k),
                        "wrong result with a standard policy, sort stability issue");
            EXPECT_TRUE(std::is_sorted(vals_begin + k, vals_begin + n),
                        "wrong result with a standard policy, sort stability issue");
        }
    }
    else
    {
        EXPECT_TRUE(std::is_sorted(keys_begin, keys_begin + n, std::greater<void>()), "wrong result with hetero policy, USM data");
        if constexpr (std::is_same_v<StabilityTag, StableSort>)
        {
            EXPECT_TRUE(std::is_sorted(vals_begin, vals_begin + n),
                        "wrong result with hetero policy, USM data, sort stability issue");
        }
    }
}

template<typename _Policy, typename StabilityTag, typename DirectionTag>
void
test_with_std_policy(_Policy&& policy, StabilityTag stability_tag, DirectionTag direction_tag)
{
    constexpr int n = 1000000;
    std::vector<int> keys_buf(n);
    std::vector<int> vals_buf(n);

    generate_data(keys_buf.begin(), vals_buf.begin(), n, direction_tag);
    call_sort(policy, keys_buf.begin(), vals_buf.begin(), n, stability_tag, direction_tag);
    check_sort(keys_buf.begin(), vals_buf.begin(), n, stability_tag, direction_tag);
}

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type, typename StabilityTag, typename DirectionTag>
void
test_with_usm(sycl::queue& q, StabilityTag stability_tag, DirectionTag direction_tag)
{
    constexpr int n = (1 << 12) + 42;
    int host_keys[n] = {};
    int host_vals[n] = {};

    generate_data(host_keys, host_vals, n, direction_tag);

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_keys(q, host_keys, host_keys + n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_vals(q, host_vals, host_vals + n);

    // calling sort
    int* device_keys = dt_helper_keys.get_data();
    int* device_vals = dt_helper_vals.get_data();
    auto policy = TestUtils::make_device_policy<KernelName<class USM, StabilityTag, DirectionTag>>(q);
    call_sort(policy, device_keys, device_vals, n, stability_tag, direction_tag);

    // checking results
    int host_out_keys[n] = {};
    int host_out_vals[n] = {};
    dt_helper_keys.retrieve_data(host_out_keys);
    dt_helper_vals.retrieve_data(host_out_vals);
   // sort_by_key with device policy guarantees stability, hence StableSort{} is passed
    check_sort(host_out_keys, host_out_vals, n, StableSort{}, direction_tag);
}

template <typename StabilityTag, typename DirectionTag>
void
test_with_buffers(sycl::queue& q, StabilityTag stability_tag, DirectionTag direction_tag)
{
    constexpr int n = 1000000;
    sycl::buffer<int> keys_buf{n};
    sycl::buffer<int> vals_buf{n};

    // generating data
    {
        sycl::host_accessor host_keys(keys_buf, sycl::write_only);
        sycl::host_accessor host_vals(vals_buf, sycl::write_only);
        generate_data(host_keys.begin(), host_vals.begin(), n, direction_tag);
    }

    // calling the algorithm
    auto keys_begin = oneapi::dpl::begin(keys_buf);
    auto vals_begin = oneapi::dpl::begin(vals_buf);
    auto policy = TestUtils::make_device_policy<KernelName<class Buffer, StabilityTag, DirectionTag>>(q);
    call_sort(policy, keys_begin, vals_begin, n, stability_tag, direction_tag);

   // checking results
   // sort_by_key with device policy guarantees stability, hence StableSort{} is passed
   {
        sycl::host_accessor host_keys(keys_buf, sycl::read_only);
        sycl::host_accessor host_vals(vals_buf, sycl::read_only);
        check_sort(host_keys.begin(), host_vals.begin(), n, StableSort{}, direction_tag);
   }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <typename StabilityTag>
void
test_device_polcies(StabilityTag stability_tag)
{
    sycl::queue q = TestUtils::get_test_queue();
    test_with_usm<sycl::usm::alloc::shared>(q, stability_tag, AscendingSort{});
    test_with_usm<sycl::usm::alloc::device>(q, stability_tag, AscendingSort{});
    test_with_buffers(q, stability_tag, DiscendingSort{});
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename StabilityTag, typename DirectionTag>
void
test_std_polcies(StabilityTag stability_tag, DirectionTag direciton_tag)
{
    test_with_std_policy(oneapi::dpl::execution::seq, stability_tag, direciton_tag);
    test_with_std_policy(oneapi::dpl::execution::unseq, stability_tag, direciton_tag);
    test_with_std_policy(oneapi::dpl::execution::par, stability_tag, direciton_tag);
    test_with_std_policy(oneapi::dpl::execution::par_unseq, stability_tag, direciton_tag);
}

template <typename StabilityTag>
void
test_all_policies(StabilityTag stability_tag)
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_device_polcies(stability_tag);
#endif // TEST_DPCPP_BACKEND_PRESENT

#if !TEST_DPCPP_BACKEND_PRESENT
    test_std_polcies(stability_tag, AscendingSort{});
    test_std_polcies(stability_tag, DiscendingSort{});
#endif // !TEST_DPCPP_BACKEND_PRESENT
}

#endif // _SORT_BY_KEY_COMMON_H
