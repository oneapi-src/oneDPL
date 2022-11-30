// -*- C++ -*-
//===-- reduce_by_segment.pass.cpp --------------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"
#include "support/reduce_serial_impl.h"

#include <iostream>
#include <iomanip>
#include <oneapi/dpl/complex>

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/sycl_alloc_utils.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

// This macro may be used to analyze source data and test results in test_inclusive_scan_by_segment
// WARNING: in the case of using this macro debug output is very large.
//#define DUMP_CHECK_RESULTS

template <typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Size, typename T,
          typename BinaryPredCheck = oneapi::dpl::__internal::__pstl_equal,
          typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
void
check_values(size_t num_segments_returned, Iterator1 host_keys, Iterator2 host_vals, Iterator3 key_res,
             Iterator4 val_res, Size n, T init, BinaryPredCheck pred = BinaryPredCheck(),
             BinaryOperationCheck op = BinaryOperationCheck())
{
    // https://docs.oneapi.io/versions/latest/onedpl/extension_api.html
    // keys:   [ 0, 0, 0, 1, 1, 1 ]
    // values: [ 1, 2, 3, 4, 5, 6 ]
    // result: [ 1 + 2 + 3 = 6, 4 + 5 + 6 = 15]

    if (n < 1)
        return;

    using ValT = typename ::std::decay<decltype(val_res[0])>::type;
    using KeyT = typename ::std::decay<decltype(key_res[0])>::type;

    std::vector<KeyT> expected_key_res(n);
    std::vector<ValT> expected_val_res(n);
    ::std::size_t num_segments =
        reduce_by_segment_serial(host_keys, host_vals, expected_key_res, expected_val_res, n, init, pred, op);

#ifdef DUMP_CHECK_RESULTS
    std::cout << "NumSegments: " << num_segments << "n: " << n << std::endl;
#endif //DUMP_CHECK_RESULTS

    for (Size i = 0; i < num_segments; ++i)
    {
#ifdef DUMP_CHECK_RESULTS
        if (val_res[i] != expected_val_res[i])
            std::cout << "Failed: " << i << ": actual(" << key_res[i] << ", " << val_res[i] << ") != expected("
                      << expected_key_res[i] << ", " << expected_val_res[i] << ")" << std::endl;
        else
            std::cout << "Success: " << i << ": actual(" << key_res[i] << ", " << val_res[i] << ") == expected("
                      << expected_key_res[i] << ", " << expected_val_res[i] << ")" << std::endl;
#endif //DUMP_CHECK_RESULTS

        EXPECT_TRUE(val_res[i] == expected_val_res[i], "wrong effect from reduce_by_segment");
    }
    EXPECT_EQ(num_segments, num_segments_returned, "incorrect return value from reduce_by_segment");
}

#if TEST_DPCPP_BACKEND_PRESENT

template <typename KernelName, typename T>
void
test_with_buffers()
{
    constexpr size_t n = 13;
    // create buffers
    sycl::buffer<T, 1> key_buf{sycl::range<1>(n)};
    sycl::buffer<T, 1> val_buf{sycl::range<1>(n)};
    sycl::buffer<T, 1> key_res_buf{sycl::range<1>(n)};
    sycl::buffer<T, 1> val_res_buf{sycl::range<1>(n)};

    {
        auto keys = key_buf.template get_access<sycl::access::mode::read_write>();
        auto vals = val_buf.template get_access<sycl::access::mode::read_write>();
        auto keys_res = key_res_buf.template get_access<sycl::access::mode::read_write>();
        auto vals_res = val_res_buf.template get_access<sycl::access::mode::read_write>();

        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
        //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };

        // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, 0};
        // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, 0};

        // Initialize data
        for (int i = 0; i < n - 1; ++i)
        {
            keys[i] = i % 4 + 1;
            vals[i] = i % 4 + 1;
            keys_res[i] = 9;
            vals_res[i] = 1;
            if (i > 3)
            {
                ++i;
                keys[i] = keys[i - 1];
                vals[i] = vals[i - 1];
                keys_res[i] = 9;
                vals_res[i] = 1;
            }
        }
        keys[n - 1] = 0;
        vals[n - 1] = 0;
    }

    // create sycl iterators
    auto key_beg = oneapi::dpl::begin(key_buf);
    auto key_end = oneapi::dpl::end(key_buf);
    auto val_beg = oneapi::dpl::begin(val_buf);
    auto key_res_beg = oneapi::dpl::begin(key_res_buf);
    auto val_res_beg = oneapi::dpl::begin(val_res_buf);

    // create named policy from existing one
    auto new_policy = oneapi::dpl::execution::make_device_policy<KernelName>(oneapi::dpl::execution::dpcpp_default);

    // call algorithm
    auto res1 = oneapi::dpl::reduce_by_segment(new_policy, key_beg, key_end, val_beg, key_res_beg, val_res_beg);

    size_t segments_key_ret = std::distance(key_beg, res1.first);
    size_t segments_val_ret = std::distance(val_beg, res1.second);
    EXPECT_EQ(segments_key_ret, segments_val_ret, "inconsistent return value from reduce_by_segment");

    auto keys_acc = key_buf.template get_access<sycl::access::mode::read>();
    auto vals_acc = val_buf.template get_access<sycl::access::mode::read>();
    auto keys_res_acc = key_res_buf.template get_access<sycl::access::mode::read>();
    auto vals_res_acc = val_res_buf.template get_access<sycl::access::mode::read>();

    check_values(segments_key_ret, keys_acc, vals_acc, keys_res_acc, vals_res_acc, n, T(0));
}

struct PrepTrivialData
{
    template <typename KeyT, typename ValT>
    void
    operator()(size_t n, KeyT* key_head, ValT* val_head, KeyT* key_res_head, ValT* val_res_head)
    {
        for (size_t i = 0; i < n; i++)
        {
            key_head[i] = KeyT(i);
            val_head[i] = ValT(i);
        }
    }
};

struct PrepRandomData
{
    template <typename KeyT, typename ValT>
    void
    operator()(size_t n, KeyT* key_head, ValT* val_head, KeyT* key_res_head, ValT* val_res_head)
    {
        size_t i = 0;
        size_t segment = 1;
        size_t seg_length = 1;

        while (i < n)
        {
            // reasonable length segment
            seg_length = std::rand() % 10000;
            //random label
            segment = std::rand();
            for (size_t j = 0; i < n && j < seg_length; j++, i++)
            {
                key_head[i] = KeyT(segment);
                //small random number to prevent overflow
                val_head[i] = ValT(std::rand() % 500);
            }
        }
    }
};

struct PrepData
{
    template <typename KeyT, typename ValT>
    void
    operator()(size_t n, KeyT* key_head, ValT* val_head, KeyT* key_res_head, ValT* val_res_head)
    {
        size_t i = 0;
        size_t segment = 1;
        size_t seg_length = 1;

        while (i < n)
        {
            for (size_t j = 0; i < n && j < seg_length; j++, i++)
            {
                key_head[i] = KeyT(segment);
                val_head[i] = ValT(1);
            }
            segment = (segment % 4) + 1;
            if (segment == 1)
            {
                seg_length++;
            }
        }
    }
};

struct PrepDataFlagPred
{
    template <typename KeyT, typename ValT>
    void
    operator()(size_t n, KeyT* key_head, ValT* val_head, KeyT* key_res_head, ValT* val_res_head)
    {
        size_t i = 0;
        size_t segment = 1;
        size_t seg_length = 1;

        while (i < n)
        {
            //mark the beginning of each segment with a "1"
            key_head[i] = KeyT(1);
            val_head[i] = ValT(1);
            i++;
            for (size_t j = 1; i < n && j < seg_length; j++, i++)
            {
                key_head[i] = KeyT(0);
                val_head[i] = ValT(1);
            }
            seg_length++;
        }
    }
};

template <sycl::usm::alloc alloc_type, typename KernelName, typename T, typename BinaryPred, typename BinaryOp,
          typename PrepareData>
void
test_with_usm(size_t n, BinaryPred pred, BinaryOp op, PrepareData prepare_data)
{
    sycl::queue q;

    // Initialize data
    T* key_head_on_host = (T*)std::malloc(n * sizeof(T));
    T* val_head_on_host = (T*)std::malloc(n * sizeof(T));
    T* key_res_head_on_host = (T*)std::malloc(n * sizeof(T));
    T* val_res_head_on_host = (T*)std::malloc(n * sizeof(T));
    T* expected_res_head_on_host = (T*)std::malloc(n * sizeof(T));

    prepare_data(n, key_head_on_host, val_head_on_host, key_res_head_on_host, val_res_head_on_host);

    //allocate USM memory and copying data to USM shared / device memory
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper1(q, key_head_on_host, n);
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper2(q, val_head_on_host, n);
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper3(q, key_res_head_on_host, n);
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper4(q, val_res_head_on_host, n);
    auto key_head = dt_helper1.get_data();
    auto val_head = dt_helper2.get_data();
    auto key_res_head = dt_helper3.get_data();
    auto val_res_head = dt_helper4.get_data();

    //call algorithm
    auto new_policy = oneapi::dpl::execution::make_device_policy<TestUtils::unique_kernel_name<
        TestUtils::unique_kernel_name<KernelName, 1>, TestUtils::uniq_kernel_index<alloc_type>()>>(q);
    auto res1 = oneapi::dpl::reduce_by_segment(new_policy, key_head, key_head + n, val_head, key_res_head, val_res_head,
                                               pred, op);

    //retrieve result on the host and check the result
    dt_helper3.retrieve_data(key_res_head_on_host);
    dt_helper4.retrieve_data(val_res_head_on_host);

    //check values

    size_t segments_key_ret = std::distance(key_res_head, res1.first);
    size_t segments_val_ret = std::distance(val_res_head, res1.second);
    EXPECT_EQ(segments_key_ret, segments_val_ret, "inconsistent return value from reduce_by_segment");

    check_values(segments_key_ret, key_head_on_host, val_head_on_host, key_res_head_on_host, val_res_head_on_host, n,
                 T(0), pred, op);

    std::free(key_head_on_host);
    std::free(val_head_on_host);
    std::free(key_res_head_on_host);
    std::free(val_res_head_on_host);
    std::free(expected_res_head_on_host);
}

void
test_with_vector()
{
    auto policy = oneapi::dpl::execution::dpcpp_default;

    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());

    for (int key_val = -1; key_val < 2; ++key_val)
    {
        // Check on interval from 0 till 4096 * 3.5 (+1024)
        for (int destLength = 0; destLength <= 14336; destLength += 100)
        {
            std::vector<int, decltype(alloc)> keys(destLength, key_val, alloc);
            std::vector<int, decltype(alloc)> values(destLength, 1, alloc);
            std::vector<int, decltype(alloc)> output_keys(destLength, alloc);
            std::vector<int, decltype(alloc)> output_values(destLength, alloc);

            auto new_end =
                oneapi::dpl::reduce_by_segment(policy, keys.begin(), keys.end(), values.begin(), output_keys.begin(),
                                               output_values.begin(), std::equal_to<int>(), std::plus<int>());

            const size_t size = new_end.first - output_keys.begin();
            EXPECT_TRUE((destLength == 0 && size == 0) || size == 1, "reduce_by_segment: wrong result");

            for (size_t i = 0; i < size; i++)
            {
                EXPECT_EQ(key_val, output_keys[i], "reduce_by_segment: wrong key");
                EXPECT_EQ(destLength, output_values[i], "reduce_by_segment: wrong value");
            }
        }
    }
}
#endif

template <typename T>
void
test_on_host()
{
    const int N = 7;
    T A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    T B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    T C[N];                         // output keys
    T D[N];                         // output values

    std::pair<T*, T*> new_end;
    new_end = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par, A, A + N, B, C, D, std::equal_to<T>(),
                                             std::plus<T>());

    size_t segments_key_ret = std::distance(C, new_end.first);
    size_t segments_val_ret = std::distance(D, new_end.second);
    EXPECT_EQ(segments_key_ret, segments_val_ret, "inconsistent return value from reduce_by_segment run on host");

    check_values(segments_key_ret, A, B, C, D, N, T(0));
}

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers<class KernelName1, std::uint64_t>();
    test_with_buffers<class KernelName2, std::complex<float>>();
    size_t n = 10000;

    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared, class KernelName3, std::uint64_t>(n, std::equal_to<>(), sycl::plus<>(),
                                                                              PrepData());
    // No known identity (legacy range algorithm)
    test_with_usm<sycl::usm::alloc::shared, class KernelName4, std::complex<float>>(n, std::equal_to<>(),
                                                                                    sycl::plus<>(), PrepData());
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device, class KernelName5, std::uint64_t>(n, std::equal_to<>(), sycl::plus<>(),
                                                                              PrepData());
    // No known identity (legacy range algorithm)
    test_with_usm<sycl::usm::alloc::device, class KernelName6, std::complex<float>>(n, std::equal_to<>(),
                                                                                    sycl::plus<>(), PrepData());

    //Mark each segment start with a 1
    auto flag_pred = [](auto a, auto b) {
        using KeyT = ::std::decay_t<decltype(b)>;
        return b != KeyT(1);
    };
    test_with_usm<sycl::usm::alloc::device, class KernelName10, std::int32_t>(n, flag_pred, sycl::plus<>(),
                                                                              PrepDataFlagPred());

    // Use maximum as binary op
    test_with_usm<sycl::usm::alloc::device, class KernelName11, std::int32_t>(n, std::equal_to<>(), sycl::maximum<>(),
                                                                              PrepRandomData());

    n = 100000;
    // Random Data
    test_with_usm<sycl::usm::alloc::device, class KernelName7, std::int32_t>(n, std::equal_to<>(), sycl::plus<>(),
                                                                             PrepRandomData());

    n = 1000;
    //Trivial Data (1000 segments of length 1)
    test_with_usm<sycl::usm::alloc::device, class KernelName8, std::int32_t>(n, std::equal_to<>(), sycl::plus<>(),
                                                                             PrepTrivialData());
    n = 1;
    //Trivial Data (1 segment of length 1)
    test_with_usm<sycl::usm::alloc::device, class KernelName9, std::int32_t>(n, std::equal_to<>(), sycl::plus<>(),
                                                                             PrepTrivialData());

    test_with_vector();
#endif
    test_on_host<int>();
    test_on_host<dpl::complex<float>>();

    return TestUtils::done();
}
