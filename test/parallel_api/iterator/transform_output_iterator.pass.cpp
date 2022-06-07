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

// These test cases are modified versions of transform_iterator.pass.cpp

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

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Size, typename TExpectedValue>
    void operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Size n, TExpectedValue expected_value)
    {
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        ::std::copy(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);

        host_vals.retrieve_data();
        auto res_begin = host_vals.get();
        for(int i = 0; i != n; ++i) {
            EXPECT_EQ(expected_value, *res_begin, "Wrong result from copy with transform_output_iterator");
            ++res_begin;
        }
    }
}; // struct test_copy

void test_simple_copy(size_t buffer_size)
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
    auto& sycl_src_buf = test_base_data.get_buffer(UDTKind::eKeys);
    auto host_source_begin = sycl_src_buf.get_access<sycl::access::mode::write>().get_pointer();

    ::std::fill_n(host_source_begin, buffer_size, identity);
    
    auto& sycl_result_buf = test_base_data.get_buffer(UDTKind::eVals);
    auto host_result_begin = sycl_result_buf.get_access<sycl::access::mode::write>().get_pointer();
    auto tr_host_result_begin = oneapi::dpl::make_transform_output_iterator(host_result_begin, transformation);

    // transformation should occur upon copy to the result
    test_copy<int> test(test_base_data);
    TestUtils::invoke_on_all_hetero_policies<0>()(test, sycl_source_begin, sycl_source_begin + buffer_size, tr_host_result_begin, buffer_size, identity + 1);
}

void test_multi_transform_copy(size_t buffer_size)
{
    // 1. create buffers
    using TestBaseData = test_base_data_buffer<int>;
    TestBaseData test_base_data({ { buffer_size, 0 },
                                  { buffer_size, 0 } });

    // 2. create iterators over buffers
    sycl::buffer<int>& source_buf = test_base_data.get_buffer(UDTKind::eKeys);
    sycl::buffer<int>& result_buf = test_base_data.get_buffer(UDTKind::eVals);

    auto host_source_begin = source_buf.template get_access<sycl::access::mode::write>().get_pointer();
    auto sycl_result_begin = oneapi::dpl::begin(result_buf);

    // 3. Run the algorithm

    auto transformation1 = [](int item) { return item + 1; };
    auto transformation2 = [](int item) { return item + 2; };
    auto transformation3 = [](int item) { return item + 3; };

    // Since transformation occurs on write, transformation3 should be the only function called. The other derefences
    // should have no effect.
    auto tr_host_source_begin = oneapi::dpl::make_transform_output_iterator(host_source_begin, transformation1);
    auto tr2_host_source_begin = oneapi::dpl::make_transform_output_iterator(tr_host_source_begin, transformation2);
    auto tr3_host_source_begin = oneapi::dpl::make_transform_output_iterator(tr2_host_source_begin, transformation3);
    auto tr3_host_source_end = tr3_host_source_begin + buffer_size;

    int identity = 0;
    ::std::fill_n(tr3_host_source_begin, buffer_size, identity);
    test_copy<int> test(test_base_data);

    TestUtils::invoke_on_all_hetero_policies<2>()(test, tr3_host_source_begin, tr3_host_source_end, sycl_result_begin, buffer_size, identity + 3);
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
    auto& sycl_src_buf = test_base_data.get_buffer(UDTKind::eKeys);
    auto host_source_begin = sycl_src_buf.get_access<sycl::access::mode::write>().get_pointer();
    auto tr_host_source_begin = oneapi::dpl::make_transform_output_iterator(host_source_begin, transformation);

    ::std::fill_n(tr_host_source_begin, buffer_size, identity);
    
    auto& sycl_result_buf = test_base_data.get_buffer(UDTKind::eVals);
    auto sycl_result_begin = oneapi::dpl::begin(sycl_result_buf);

    // verify the data has been transformed in the result buffer.
    test_copy<int> test(test_base_data);
    TestUtils::invoke_on_all_hetero_policies<0>()(test, sycl_source_begin, sycl_source_begin + buffer_size, sycl_result_begin, buffer_size, identity + 1);
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
    }
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
