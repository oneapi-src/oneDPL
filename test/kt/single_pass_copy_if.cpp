// -*- C++ -*-
//===-- single_pass_copy_if.cpp -------------------------------------------===//
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

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>

#if LOG_TEST_INFO
#    include <iostream>
#endif

#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"
#include "../support/scan_serial_impl.h"

#include "esimd_radix_sort_utils.h"

#include <random>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>
#include <type_traits>

inline const std::vector<std::size_t> copy_if_sizes = {
    1,       6,         16,      43,        256,           316,           2048,
    5072,    8192,      14001,   1 << 14,   (1 << 14) + 1, 50000,         67543,
    100'000, 1 << 17,   179'581, 250'000,   1 << 18,       (1 << 18) + 1, 500'000,
    888'235, 1'000'000, 1 << 20, 10'000'000};

template <typename T>
struct __less_than_val
{
    const T __val;
    bool
    operator()(const T& __v) const
    {
        return __v < __val;
    }
};

template <typename T>
auto
generate_copy_if_data(T* input, std::size_t size, std::uint32_t seed)
{
    // Integer numbers are generated even for floating point types in order to avoid rounding errors,
    // and simplify the final check
    std::default_random_engine gen{seed};

    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        std::generate(input, input + size, [&] { return dist(gen); });
    }
    else
    {
        std::uniform_real_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        std::generate(input, input + size, [&] { return dist(gen); });
    }
}

#if _ENABLE_RANGES_TESTING
template <typename T, typename Predicate, typename KernelParam>
void
test_all_view(sycl::queue q, std::size_t size, Predicate pred, KernelParam param)
{
#    if LOG_TEST_INFO
    std::cout << "\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#    endif
    std::vector<T> input(size);
    generate_copy_if_data(input.data(), size, 42);
    std::vector<T> ref(input);
    std::vector<T> out_ref(size);
    sycl::buffer<T> buf_out(input.size());
    std::size_t num_copied = 0;
    sycl::buffer<std::size_t> buf_num_copied(&num_copied, 1);
    auto out_end = std::copy_if(std::begin(ref), std::end(ref), std::begin(out_ref), pred);
    std::size_t num_copied_ref = out_end - std::begin(out_ref);
    sycl::buffer<T> buf(input.data(), input.size());

    oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read> view(buf);
    oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::write> view_out(buf_out);
    oneapi::dpl::experimental::ranges::all_view<std::size_t, sycl::access::mode::write> view_num_copied(buf_num_copied);
    oneapi::dpl::experimental::kt::gpu::copy_if(q, view, view_out, view_num_copied, pred, param).wait();

    auto acc = buf_out.get_host_access();
    auto num_copied_acc = buf_num_copied.get_host_access();

    std::string msg1 = "wrong num copied with all_view, n: " + std::to_string(size);
    EXPECT_EQ(num_copied_ref, num_copied_acc[0], msg1.c_str());
    std::string msg2 = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_N(out_ref.begin(), acc.begin(), num_copied_ref, msg2.c_str());
}

template <typename T, typename Predicate, typename KernelParam>
void
test_buffer(sycl::queue q, std::size_t size, Predicate pred, KernelParam param)
{
#    if LOG_TEST_INFO
    std::cout << "\ttest_buffer(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#    endif
    std::vector<T> input(size);
    generate_copy_if_data(input.data(), size, 42);
    std::vector<T> ref(input);
    std::vector<T> out_ref(size);
    sycl::buffer<T> buf_out(size);
    std::size_t num_copied = 0;
    sycl::buffer<std::size_t> buf_num_copied(&num_copied, 1);
    auto out_end = std::copy_if(std::begin(ref), std::end(ref), std::begin(out_ref), pred);
    std::size_t num_copied_ref = out_end - std::begin(out_ref);
    {
        sycl::buffer<T> buf(input.data(), input.size());

        oneapi::dpl::experimental::kt::gpu::copy_if(q, buf, buf_out, buf_num_copied, pred, param).wait();
    }

    auto acc = buf_out.get_host_access();
    auto num_copied_acc = buf_num_copied.get_host_access();

    std::string msg1 = "wrong num copied with buffer, n: " + std::to_string(size);
    EXPECT_EQ(num_copied_ref, num_copied_acc[0], msg1.c_str());
    std::string msg2 = "wrong results with buffer, n: " + std::to_string(size);
    EXPECT_EQ_N(out_ref.begin(), acc.begin(), num_copied_ref, msg2.c_str());
}
#endif

template <typename T, sycl::usm::alloc _alloc_type, typename Predicate, typename KernelParam>
void
test_usm(sycl::queue q, std::size_t size, Predicate pred, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>() << ">("
              << size << ");" << std::endl;
#endif
    std::vector<T> in_ref(size);
    generate_copy_if_data(in_ref.data(), size, 42);
    std::vector<T> out_ref(size);

    TestUtils::usm_data_transfer<_alloc_type, T> dt_input(q, in_ref.begin(), in_ref.end());
    TestUtils::usm_data_transfer<_alloc_type, T> dt_output(q, size);
    TestUtils::usm_data_transfer<_alloc_type, std::size_t> dt_num_copied(q, 1);

    std::size_t num_copied = 0;
    auto out_end = std::copy_if(std::begin(in_ref), std::end(in_ref), std::begin(out_ref), pred);
    std::size_t num_copied_ref = out_end - std::begin(out_ref);

    oneapi::dpl::experimental::kt::gpu::copy_if(q, dt_input.get_data(), dt_input.get_data() + size,
                                                dt_output.get_data(), dt_num_copied.get_data(), pred, param)
        .wait();

    std::vector<T> actual(size);
    dt_output.retrieve_data(actual.begin());
    std::vector<std::size_t> num_copied_host(1);
    dt_num_copied.retrieve_data(num_copied_host.begin());

    std::string msg1 = "wrong num copied with USM, n: " + std::to_string(size);
    EXPECT_EQ(num_copied_ref, num_copied_host[0], msg1.c_str());
    std::string msg2 = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(out_ref.begin(), actual.begin(), num_copied_ref, msg2.c_str());
}

template <typename T, typename Predicate, typename KernelParam>
void
test_sycl_iterators(sycl::queue q, std::size_t size, Predicate pred, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif
    std::vector<T> input(size);
    std::vector<T> output(size);
    generate_copy_if_data(input.data(), size, 42);
    std::vector<T> ref(input);
    std::vector<T> out_ref(size);
    std::size_t num_copied = 0;
    auto out_end = std::copy_if(std::begin(ref), std::end(ref), std::begin(out_ref), pred);
    std::size_t num_copied_ref = out_end - std::begin(out_ref);
    {
        sycl::buffer<T> buf(input.data(), input.size());
        sycl::buffer<T> buf_out(output.data(), output.size());
        sycl::buffer<std::size_t> buf_num(&num_copied, 1);
        oneapi::dpl::experimental::kt::gpu::copy_if(q, oneapi::dpl::begin(buf), oneapi::dpl::end(buf),
                                                    oneapi::dpl::begin(buf_out), oneapi::dpl::begin(buf_num), pred,
                                                    param)
            .wait();
    }

    std::string msg1 = "wrong num copied with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ(num_copied_ref, num_copied, msg1.c_str());
    std::string msg2 = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_N(out_ref.begin(), output.begin(), num_copied_ref, msg2.c_str());
}

template <typename T, typename Predicate, typename KernelParam>
void
test_general_cases(sycl::queue q, std::size_t size, Predicate pred, KernelParam param)
{
    test_usm<T, sycl::usm::alloc::shared>(q, size, pred, TestUtils::get_new_kernel_params<0>(param));
    test_usm<T, sycl::usm::alloc::device>(q, size, pred, TestUtils::get_new_kernel_params<1>(param));
    test_sycl_iterators<T>(q, size, pred, TestUtils::get_new_kernel_params<2>(param));
#if _ENABLE_RANGES_TESTING
    test_all_view<T>(q, size, pred, TestUtils::get_new_kernel_params<3>(param));
    test_buffer<T>(q, size, pred, TestUtils::get_new_kernel_params<4>(param));
#endif
}

template <typename T, typename Predicate, typename KernelParam>
void
test_all_cases(sycl::queue q, std::size_t size, Predicate pred, KernelParam param)
{
    test_general_cases<T>(q, size, pred, TestUtils::get_new_kernel_params<0>(param));
}

int
main()
{
#if LOG_TEST_INFO
    std::cout << "TEST_DATA_PER_WORK_ITEM : " << TEST_DATA_PER_WORK_ITEM << "\n"
              << "TEST_WORK_GROUP_SIZE    : " << TEST_WORK_GROUP_SIZE << "\n"
              << "TEST_SINGLE_WG_OPTOUT   : " << TEST_SINGLE_WG_OPTOUT << "\n"
              << "TEST_TYPE               : " << TypeInfo().name<TEST_TYPE>() << std::endl;
#endif

    constexpr oneapi::dpl::experimental::kt::kernel_param<
        TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE,
        /*opt_out_single_wg=*/std::bool_constant<TEST_SINGLE_WG_OPTOUT>>
        params;
    auto q = TestUtils::get_test_queue();
    bool run_test = can_run_test<decltype(params), TEST_TYPE>(q, params);

    TEST_TYPE cutoff = std::is_signed_v<TEST_TYPE> ? TEST_TYPE{0} : std::numeric_limits<TEST_TYPE>::max() / 2;
    auto __predicate = __less_than_val<TEST_TYPE>{cutoff};
    if (run_test)
    {

        try
        {
            for (auto size : copy_if_sizes)
                test_all_cases<TEST_TYPE>(q, size, __predicate, params);
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    return TestUtils::done(run_test);
}
