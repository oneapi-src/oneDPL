// -*- C++ -*-
//===-- transform_iterator.pass.cpp ---------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#include "support/utils.h"
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include <tuple>

#if _PSTL_BACKEND_SYCL

template <typename Iterator>
class test_copy {
    using result_type = typename ::std::iterator_traits<Iterator>::value_type;

    size_t buffer_size;
    Iterator result_begin;
    result_type expected_value;
public:
    test_copy(size_t buf_size, Iterator res_begin, const result_type& val)
        : buffer_size(buf_size), result_begin(res_begin), expected_value(val) {}

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2) {
        ::std::copy(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);
        auto host_first = TestUtils::get_host_pointer(result_begin);

        auto res_begin = host_first;
        for(int i = 0; i != buffer_size; ++i) {
            EXPECT_EQ(expected_value, *res_begin, "Wrong result from copy with transform_iterator");
            ++res_begin;
        }
    }
}; // struct test_copy

template <typename Iterator>
struct test_copy_if {
    size_t expected_buffer_size;
    Iterator result_begin;
public:
    test_copy_if(size_t buf_size, Iterator res_begin)
        : expected_buffer_size(buf_size), result_begin(res_begin) {}

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2) {
        using value_type = typename ::std::iterator_traits<Iterator1>::value_type;
        auto predicate = [](value_type val) { return val <= 10; };

        Iterator res_begin = result_begin;
        for(int i = 0; i != expected_buffer_size; ++i) {
            if ((i + 1) <= 10 ) {
                EXPECT_EQ(i + 1, *res_begin, "Wrong result from copy_if with transform_iterator");
                ++res_begin;
            }
        }
    }
}; // struct test_copy_if

void test_simple_copy(size_t buffer_size) {
    cl::sycl::buffer<int> source_buf{ buffer_size };
    cl::sycl::buffer<int> result_buf{ buffer_size };
    auto host_source_begin = source_buf.template get_access<cl::sycl::access::mode::write>().get_pointer();

    auto sycl_source_begin = oneapi::dpl::begin(source_buf);
    auto sycl_result_begin = oneapi::dpl::begin(result_buf);

    auto transformation = [](int item) { return item + 1; };

    auto tr_sycl_source_begin = oneapi::dpl::make_transform_iterator(sycl_source_begin, transformation);
    auto tr_sycl_source_end = tr_sycl_source_begin + buffer_size;

    int identity = 0;
    ::std::fill_n(host_source_begin, buffer_size, identity);

    test_copy<decltype(sycl_result_begin)> test(buffer_size, sycl_result_begin, identity + 1);
    TestUtils::invoke_on_all_hetero_policies<0>()(test, tr_sycl_source_begin, tr_sycl_source_end, sycl_result_begin);
}

void test_ignore_copy(size_t buffer_size) {
    cl::sycl::buffer<int> source_buf{ buffer_size };
    cl::sycl::buffer<int> result_buf{ buffer_size };
    auto host_source_begin = source_buf.template get_access<cl::sycl::access::mode::write>().get_pointer();
    auto host_result_begin = result_buf.template get_access<cl::sycl::access::mode::write>().get_pointer();

    auto sycl_source_begin = oneapi::dpl::begin(source_buf);
    auto sycl_source_end = oneapi::dpl::end(source_buf);
    auto sycl_result_begin = oneapi::dpl::begin(result_buf);

    auto transformation = [](int) { return ::std::ignore; };

    auto tr_sycl_result_begin = oneapi::dpl::make_transform_iterator(sycl_result_begin, transformation);

    int ignored = -100;
    ::std::fill_n(host_source_begin, buffer_size, 1);
    ::std::fill_n(host_result_begin, buffer_size, ignored);

    test_copy<decltype(sycl_result_begin)> test(buffer_size, sycl_result_begin, ignored);
    TestUtils::invoke_on_all_hetero_policies<1>()(test, sycl_source_begin, sycl_source_end, tr_sycl_result_begin);
}

void test_multi_transform_copy(size_t buffer_size) {
    cl::sycl::buffer<int> source_buf{ buffer_size };
    cl::sycl::buffer<int> result_buf{ buffer_size };
    auto host_source_begin = source_buf.template get_access<cl::sycl::access::mode::write>().get_pointer();

    auto sycl_source_begin = oneapi::dpl::begin(source_buf);
    auto sycl_source_end = sycl_source_begin + buffer_size;
    auto sycl_result_begin = oneapi::dpl::begin(result_buf);

    auto transformation = [](int item) { return item + 1; };

    auto tr_sycl_source_begin = oneapi::dpl::make_transform_iterator(sycl_source_begin, transformation);
    auto tr2_sycl_source_begin = oneapi::dpl::make_transform_iterator(tr_sycl_source_begin, transformation);
    auto tr3_sycl_source_begin = oneapi::dpl::make_transform_iterator(tr2_sycl_source_begin, transformation);
    auto tr3_sycl_source_end = tr3_sycl_source_begin + buffer_size;

    int identity = 0;
    ::std::fill_n(host_source_begin, buffer_size, identity);

    test_copy<decltype(sycl_result_begin)> test(buffer_size, sycl_result_begin, identity + 3);
    TestUtils::invoke_on_all_hetero_policies<2>()(test, tr3_sycl_source_begin, tr3_sycl_source_end, sycl_result_begin);
}

#endif // _PSTL_BACKEND_SYCL

int32_t main() {
#if _PSTL_BACKEND_SYCL
    size_t max_n = 10000;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        test_simple_copy(n);
        test_ignore_copy(n);
        test_multi_transform_copy(n);
    }
#endif
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
