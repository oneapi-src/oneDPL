// -*- C++ -*-
//===-- transform_output_iterator.pass.cpp ---------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

#include <tuple>

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT

DEFINE_TEST(test_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy)

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Size, typename Iterator3>
    void operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Size n, Iterator3 expected_values)
    {
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        ::std::copy(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);

        host_vals.retrieve_data();
        auto res_begin = host_vals.get();
        
        EXPECT_EQ_N(expected_values, res_begin, n, "Wrong result from copy with transform_output_iterator");
    }
}; // struct test_copy


DEFINE_TEST(test_copy_typeshift)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy_typeshift)

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Size, typename Iterator3>
    void operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Size n, Iterator3 expected_values)
    {

        ::std::copy(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);

      
        EXPECT_EQ_N(expected_values, first2, n, "Wrong result from copy with transform_output_iterator");
    }
}; // struct test_copy_typeshift


void test_simple_copy(size_t buffer_size)
{
    // 1. create buffers - usm to test copying to transform output iterator
    sycl::queue q{};

    using TestBaseData = test_base_data_usm<sycl::usm::alloc::shared, int>;
    TestBaseData test_base_data(q, {{ buffer_size, 0 }, 
                                    { buffer_size, 0 }});

    // 2. create iterators over source buffer 
    auto sycl_source_begin = test_base_data.get_start_from(UDTKind::eKeys);
    auto sycl_result_begin = test_base_data.get_start_from(UDTKind::eVals);

    // 3. run algorithms
    auto transformation = [](int item) { return item + 1; };

    int identity = 0;
    ::std::fill_n(sycl_source_begin, buffer_size, identity);
    auto tr_host_result_begin = oneapi::dpl::make_transform_output_iterator(sycl_result_begin, transformation); 

    ::std::vector<int> expected_res(buffer_size, identity + 1);

    test_copy<int> test(test_base_data);
    TestUtils::invoke_on_all_hetero_policies<1>()(test, sycl_source_begin, sycl_source_begin + buffer_size, 
        tr_host_result_begin, buffer_size, expected_res.begin());
}

void test_multi_transform_copy(size_t buffer_size)
{
    // 1. create buffers
    sycl::queue q{};

    using TestBaseData = test_base_data_usm<sycl::usm::alloc::shared, int>;
    TestBaseData test_base_data(q, {{buffer_size, 0}, {buffer_size, 0}});

    // 2. create iterators over buffers
    auto sycl_source_begin = test_base_data.get_start_from(UDTKind::eKeys);
    auto sycl_result_begin = test_base_data.get_start_from(UDTKind::eVals);

    // 3. Run the algorithm

    auto transformation1 = [](int item) { return item + 1; };
    auto transformation2 = [](int item) { return item * 2; };
    auto transformation3 = [](int item) { return item + 3; };

    int identity = 0;
    ::std::fill_n(sycl_source_begin, buffer_size, identity);

    auto tr_sycl_result_begin = oneapi::dpl::make_transform_output_iterator(sycl_result_begin, transformation1);
    auto tr2_sycl_result_begin = oneapi::dpl::make_transform_output_iterator(tr_sycl_result_begin, transformation2);
    auto tr3_sycl_result_begin = oneapi::dpl::make_transform_output_iterator(tr2_sycl_result_begin, transformation3);

    test_copy<int> test(test_base_data);

    ::std::vector<int> expected_res(buffer_size, ((identity + 3) * 2) + 1);
    
    TestUtils::invoke_on_all_hetero_policies<2>()(test, sycl_source_begin, sycl_source_begin + buffer_size,
                                                  tr3_sycl_result_begin, buffer_size, expected_res.begin());
}

void test_fill_transform(size_t buffer_size)
{
    // 1. create buffers
    using TestBaseData = test_base_data_buffer<int>;
    TestBaseData test_base_data({ { buffer_size, 0 },
                                  { buffer_size, 0 } });

    // 2. create iterators over source buffer 
    auto sycl_source_begin = test_base_data.get_start_from(UDTKind::eKeys);

    // 3. run algorithms
    auto transformation = [](int item) { return item + 1; };

    int identity = 0;
    {
        auto& sycl_src_buf = test_base_data.get_buffer(UDTKind::eKeys);
        auto host_source_acc = sycl_src_buf.get_access<sycl::access::mode::write>();
        auto host_source_begin = host_source_acc.get_pointer();
        auto tr_host_source_begin = oneapi::dpl::make_transform_output_iterator(host_source_begin, transformation);
        ::std::fill_n(tr_host_source_begin, buffer_size, identity);
    }
    auto& sycl_result_buf = test_base_data.get_buffer(UDTKind::eVals);
    auto sycl_result_begin = oneapi::dpl::begin(sycl_result_buf);

    ::std::vector<int> expected_res(buffer_size, identity + 1);

    test_copy<int> test(test_base_data);
    TestUtils::invoke_on_all_hetero_policies<0>()(test, sycl_source_begin, sycl_source_begin + buffer_size, sycl_result_begin, buffer_size, expected_res.begin());
}

void
test_type_shift(size_t buffer_size)
{

    // 1. create buffers
    sycl::queue q{};

    using TestBaseOutputData = test_base_data_usm<sycl::usm::alloc::shared, int>;
    TestBaseOutputData test_base_output_data(q, {{buffer_size, 0}});
    using TestBaseInputData = test_base_data_usm<sycl::usm::alloc::shared, double>;
    TestBaseInputData test_base_input_data(q, {{buffer_size, 0}});

    // 2. create iterators over source buffer
    auto sycl_source_begin = test_base_input_data.get_start_from(UDTKind::eKeys);
    auto sycl_result_begin = test_base_output_data.get_start_from(UDTKind::eKeys);

    // 3. run algorithms
    auto transformation1 = [](float item) { return (int)(item * 2.0f); };
    auto transformation2 = [](double item) { return (float)(item + 1.0); };

    double init = 0.5;

    ::std::fill_n(sycl_source_begin, buffer_size, init);

    auto tr1_host_result_begin = oneapi::dpl::make_transform_output_iterator(sycl_result_begin, transformation1);
    auto tr2_host_result_begin = oneapi::dpl::make_transform_output_iterator(tr1_host_result_begin, transformation2);

    ::std::vector<int> expected_res(buffer_size, (int)((float)(init + 1.0) * 2.0f));

    test_copy_typeshift<int> test(test_base_output_data);
    TestUtils::invoke_on_all_hetero_policies<3>()(test, sycl_source_begin, sycl_source_begin + buffer_size,
                                                   tr2_host_result_begin, buffer_size, expected_res.begin());
}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    size_t max_n = 10000;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_simple_copy(n);
        test_multi_transform_copy(n);
        test_fill_transform(n);
        test_type_shift(n);
    }
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
