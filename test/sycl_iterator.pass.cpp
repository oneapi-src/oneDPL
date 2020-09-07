// -*- C++ -*-
//===-- sycl_iterator.pass.cpp --------------------------------------------===//
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

#include <cmath>
#include <type_traits>

#include "support/pstl_test_config.h"

#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(memory)
#include _PSTL_TEST_HEADER(iterator)

#include "oneapi/dpl/pstl/utils.h"

using namespace TestUtils;

//This macro is required for the tests to work correctly in CI with tbb-backend.
#if _PSTL_BACKEND_SYCL
#include "support/utils_sycl.h"

template <class KernelName1, typename KernelName2>
class ForComplexName
{
};

struct Flip
{
    int32_t val;
    Flip(int32_t y) : val(y) {}
    template <typename T>
    T
    operator()(const T& x) const
    {
        return val - x;
    }
};
struct Plus
{
    template <typename T, typename U>
    T
    operator()(const T x, const U y) const
    {
        return x + y;
    }
};

struct Inc
{
    template <typename T>
    void
    operator()(T& x) const
    {
        ++x;
    }
};

template <typename T>
struct Generator_count
{
    T def_val;
    Generator_count(const T& val) : def_val(val) {}
    T
    operator()()
    {
        return def_val;
    }
    T
    default_value() const
    {
        return def_val;
    }
};

// created just to check destroy and destroy_n correctness
template <typename T>
struct SyclTypeWrapper
{
    T __value;

    explicit SyclTypeWrapper(const T& value = T{4}) : __value(value) {}
    ~SyclTypeWrapper() { __value = -2; }
    bool
    operator==(const SyclTypeWrapper& other) const
    {
        return __value == other.__value;
    }
};

// this wrapper is needed to take into account not only kernel name,
// but also other types (for example, iterator's value type)
template<typename... T>
struct policy_name_wrapper{};

using namespace oneapi::dpl::execution;

struct test_uninitialized_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto value = IteratorValueType(42);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, IteratorValueType{-1});

        ::std::uninitialized_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from uninitialized_copy");
    }
};

struct test_uninitialized_copy_n
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto value = IteratorValueType(42);
        ::std::fill_n(host_first1, n, value);
        ::std::fill_n(host_first2, n, IteratorValueType{0});

        ::std::uninitialized_copy_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from uninitialized_copy_n");
    }
};

struct test_uninitialized_move
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto value = IteratorValueType(42);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, IteratorValueType{-1});

        ::std::uninitialized_move(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from uninitialized_move");
    }
};

struct test_uninitialized_move_n
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto value = IteratorValueType(42);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, IteratorValueType{-1});

        ::std::uninitialized_move_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from uninitialized_move_n");
    }
};

struct test_uninitialized_fill
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                                value);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1 + (n / 3), host_first1 + (n / 2), value),
                    "wrong effect from uninitialized_fill");
    }
};

struct test_uninitialized_fill_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from uninitialized_fill_n");
    }
};

struct test_uninitialized_default_construct
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        T1 exp_value; // default-constructed value
        auto value = T1{2};
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::uninitialized_default_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                             first1 + (n / 3), first1 + (n / 2));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1 + (n / 3), host_first1 + (n / 2), exp_value),
                    "wrong effect from uninitialized_default_construct");
    }
};

struct test_uninitialized_default_construct_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        T1 exp_value; // default-constructed value
        auto value = T1{2};
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::uninitialized_default_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1,
                                               n);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, exp_value),
                    "wrong effect from uninitialized_default_construct_n");
    }
};

struct test_uninitialized_value_construct
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::uninitialized_value_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                           first1 + (n / 3), first1 + (n / 2));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1 + (n / 3), host_first1 + (n / 2), T1{}),
                    "wrong effect from uninitialized_value_construct");
    }
};

struct test_uninitialized_value_construct_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::uninitialized_value_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, T1{}),
                    "wrong effect from uninitialized_value_construct_n");
    }
};

struct test_destroy
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{2};
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::destroy(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec),
            first1 + (n / 3), first1 + (n / 2));
        if(!::std::is_trivially_destructible<T1>::value)
            value = T1{-2};
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1 + (n / 3), host_first1 + (n / 2), value), "wrong effect from destroy");
    }
};

struct test_destroy_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{2};
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::destroy_n(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1, n);
        if(!::std::is_trivially_destructible<T1>::value)
            value = T1{-2};
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value), "wrong effect from destroy_n");
    }
};

struct test_fill
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), value);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1 + (n / 3), host_first1 + (n / 2), value), "wrong effect from fill");
    }
};

struct test_fill_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from fill_n");
    }
};

struct test_generate
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                      Generator_count<T1>(value));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1 + (n / 3), host_first1 + (n / 2), value), "wrong effect from generate");
    }
};

struct test_generate_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);
        ::std::generate_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Generator_count<T1>(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from generate_n");
    }
};

struct test_for_each
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(6);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first1 + (n / 3), host_first1 + (n / 2), value - 1);

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), Inc());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        // We call due to SYCL 1.2.1: 4.7.2.3.
        // If the host memory is modified by the host,
        // or mapped to another buffer or image during the lifetime of this buffer,
        // then the results are undefined
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value), "wrong effect from for_each");
    }
};

struct test_for_each_n
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::for_each_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Inc());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from for_each_n");
    }
};

struct test_transform_unary
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, value + 1);
        ::std::transform(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + n / 2, last1, first2 + n / 2, Flip(7));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n / 2, value + 1), "wrong effect from transform_unary (1)");
        EXPECT_TRUE(check_values(host_first2 + n / 2, host_first2 + n, T1(5)), "wrong effect from transform_unary (2)");
    }
};

struct test_transform_binary
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(3);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::transform(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first1, first2, Plus());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, T1(6)), "wrong effect from transform_binary");
    }
};

struct test_replace
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::replace(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, value, T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from replace");
    }
};

struct test_replace_if
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::replace_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                        oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from replace_if");
    }
};

struct test_replace_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::replace_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, value, T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value + 1), "wrong effect from replace_copy");
    }
};

struct test_replace_copy_if
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::replace_copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2,
                             oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value + 1), "wrong effect from replace_copy_if");
    }
};

struct test_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, IteratorValueType{0});

        ::std::copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from copy");
    }
};

struct test_copy_n
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, IteratorValueType{0});

        ::std::copy_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from copy_n");
    }
};

struct test_move
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, IteratorValueType{0});

        ::std::move(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(check_values(host_first2, host_first2 + n, value), "wrong effect from move");
    }
};

struct test_adjacent_difference
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        // init
        Iterator1ValueType fill_value{1};
        Iterator2ValueType blank_value{0};
        auto __f = [](Iterator1ValueType& a, Iterator1ValueType& b) -> Iterator2ValueType { return a + b; };
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        ::std::for_each(host_first1, host_first1 + n,
                      [&fill_value](Iterator1ValueType& val) { val = (fill_value++ % 10) + 1; });

        // test with custom functor
        ::std::fill(host_first2, host_first2 + n, blank_value);
        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, __f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_first1 = get_host_pointer(first1);
        host_first2 = get_host_pointer(first2);
        bool is_correct = *host_first1 == *host_first2; // for the first element
        for (int i = 1; i < n; ++i)                     // for subsequent elements
            is_correct = is_correct && *(host_first2 + i) == __f(*(host_first1 + i), *(host_first1 + i - 1));

        EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #1");

        // test with default functor
        ::std::fill(host_first2, host_first2 + n, blank_value);
        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_first1 = get_host_pointer(first1);
        host_first2 = get_host_pointer(first2);
        is_correct = *host_first1 == *host_first2; // for the first element
        for (int i = 1; i < n; ++i)                // for subsequent elements
            is_correct = is_correct && *(host_first2 + i) == (*(host_first1 + i) - *(host_first1 + i - 1));

        EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #2");
    }
};

struct test_reduce
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);
        auto value = T1(2);
        ::std::fill(host_first1, host_first1 + n, T1(0));
        ::std::fill(host_first1 + (n / 3), host_first1 + (n / 2), value);

        // without initial value
        auto result1 = ::std::reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result1 == value * (n / 2 - n / 3), "wrong effect from reduce (1)");

        // with initial value
        auto init = T1(42);
        auto result2 = ::std::reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), init);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result2 == init + value * (n / 2 - n / 3), "wrong effect from reduce (2)");
    }
};

struct test_transform_reduce_unary
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);
        auto value = T1(1);
        ::std::fill(host_first1, host_first1 + n, value);

        auto result = ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, T1(42),
                                            Plus(), ::std::negate<T1>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == 42 - n, "wrong effect from transform_reduce (unary + binary)");
    }
};

struct test_transform_reduce_binary
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);
        auto value = T1(1);
        ::std::fill(host_first1, host_first1 + n, value);

        auto result =
            ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first1, T1(42));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == n + 42, "wrong effect from transform_reduce (2 binary)");
    }
};

struct test_min_element
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{static_cast<IteratorValueType>(::std::distance(first, last))};
        auto host_first1 = get_host_pointer(first);
        ::std::for_each(host_first1, host_first1 + n,
                      [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t min_dis = n;
        if (min_dis)
        {
            *(host_first1 + min_dis / 2) = IteratorValueType{/*min_val*/ 0};
            *(host_first1 + n - 1) = IteratorValueType{/*2nd min*/ 0};
        }

        auto result_min = ::std::min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto expected_min = ::std::min_element(host_first1, host_first1 + n);

        EXPECT_TRUE(result_min - first == expected_min - host_first1, "wrong effect from min_element");
    }
};

struct test_adjacent_find
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::equal_to<ValueType>{};
        auto host_first1 = get_host_pointer(first);

        ValueType fill_value{0};
        ::std::for_each(host_first1, host_first1 + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });

        // check with no adjacent equal elements
        Iterator result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #1 no elements)");

        // check with the last adjacent element
        ::std::size_t max_dis = n;
        host_first1 = get_host_pointer(first);
        if (max_dis > 1)
            *(host_first1 + n - 1) = *(host_first1 + n - 2);
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 2 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #2 the last element)");

        // check with an adjacent element
        max_dis = n;
        Iterator it{last};
        host_first1 = get_host_pointer(first);
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_first1 + max_dis / 2) = *(host_first1 + max_dis / 2 - 1);
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? it - 1 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #3 middle element)");
        // check with an adjacent element (no predicate)
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #4 middle element (no predicate))");

        // check with the first adjacent element
        max_dis = n;
        host_first1 = get_host_pointer(first);
        if (max_dis > 1)
            *(host_first1 + 1) = *host_first1;
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? first : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #5 the first element)");
    }
};

struct test_max_element
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{0};
        auto host_first1 = get_host_pointer(first);
        ::std::for_each(host_first1, host_first1 + n,
                      [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t max_dis = n;
        if (max_dis)
        {
            *(host_first1 + max_dis / 2) = IteratorValueType{/*max_val*/ 777};
            *(host_first1 + n - 1) = IteratorValueType{/*2nd max*/ 777};
        }

        auto result_max = ::std::max_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        auto expected_max = ::std::max_element(host_first1, host_first1 + n);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        EXPECT_TRUE(result_max - first == expected_max - host_first1, "wrong effect from max_element");
    }
};

struct test_is_sorted_until
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{0};
        auto host_first1 = get_host_pointer(first);
        ::std::for_each(host_first1, host_first1 + n, [&fill_value](ValueType& value) { value = ++fill_value; });

        // check sorted
        Iterator result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from is_sorted_until (Test #1 sorted sequence)");

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        host_first1 = get_host_pointer(first);
        if (max_dis > 1)
            *(host_first1 + n - 1) = ValueType{0};
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 1 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #2 unsorted sequence - the last element)");

        // check unsorted: the middle element
        max_dis = n;
        Iterator it{last};
        host_first1 = get_host_pointer(first);
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_first1 + /*max_idx*/ max_dis / 2) = ValueType{0};
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected = it;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #3 unsorted sequence - the middle element)");
        // check unsorted: the middle element (no predicate)
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(
            result == expected,
            "wrong effect from is_sorted_until (Test #4 unsorted sequence - the middle element (no predicate))");

        // check unsorted: the first element
        host_first1 = get_host_pointer(first);
        if (n > 1)
            *(host_first1 + 1) = ValueType{0};
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected = n > 1 ? first + 1 : last;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #5 unsorted sequence - the first element)");
    }
};

struct test_minmax_element
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto host_first = get_host_pointer(first);
        IteratorValueType fill_value = IteratorValueType{0};
        ::std::for_each(host_first, host_first + n, [&fill_value](IteratorValueType& it) { it = fill_value++ % 10 + 1; });

        ::std::size_t dis = n;
        if (dis > 1)
        {
            auto min_it = host_first + /*min_idx*/ dis / 2 - 1;
            *(min_it) = IteratorValueType{/*min_val*/ 0};

            auto max_it = host_first + /*max_idx*/ dis / 2;
            *(max_it) = IteratorValueType{/*max_val*/ 777};
        }

        auto expected = ::std::minmax_element(host_first, host_first + n);
        auto result = ::std::minmax_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);

        auto result_min = result.first - first;
        auto result_max = result.second - first;

        auto expected_min = expected.first - host_first;
        auto expected_max = expected.second - host_first;

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_min == expected_min && result_max == expected_max, "wrong effect from minmax_element");
        if (!(result_min == expected_min && result_max == expected_max))
        {
            auto got_min = get_host_pointer(result.first);
            auto got_max = get_host_pointer(result.second);
            ::std::cout << "MIN got: " << got_min << "[" << result_min << "], "
                      << "expected: " << *expected.first << "[" << expected_min << "]" << ::std::endl;
            ::std::cout << "MAX got: " << got_max << "[" << result_max << "], "
                      << "expected: " << *expected.second << "[" << expected_max << "]" << ::std::endl;
        }
    }
};

struct test_is_sorted
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto host_first = get_host_pointer(first);
        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_first, host_first + n, [&fill_value](ValueType& value) { value = ++fill_value; });

        // check sorted
        bool result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        bool expected_bool = true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_sorted (Test #1 sorted sequence)");
        host_first = get_host_pointer(first);
        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
            *(host_first + n - 1) = ValueType{0};
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #2 unsorted sequence - the last element)");

        // check unsorted: the middle element
        max_dis = n;
        if (max_dis > 1)
        {
            host_first = get_host_pointer(first);
            *(host_first + max_dis / 2) = ValueType{0};
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #3 unsorted sequence - the middle element)");
        // check unsorted: the middle element (no predicate)
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #4 unsorted sequence - the middle element (no predicate))");

        // check unsorted: the first element
        max_dis = n;
        if (max_dis > 1)
        {
            host_first = get_host_pointer(first);
            *(host_first + 1) = ValueType{0};
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted Test #5 unsorted sequence - the first element");
    }
};

struct test_count
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        auto host_first1 = get_host_pointer(first);
        ValueType fill_value{0};
        ::std::for_each(host_first1, host_first1 + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{0});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count (Test #1 arbitrary to count)");

        // check when none should be counted
        expected = 0;
        result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{12});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count (Test #2 none to count)");

        // check when all should be counted
        host_first1 = get_host_pointer(first);
        ::std::fill(host_first1, host_first1 + n, ValueType{7});
        expected = n;
        result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{7});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count (Test #3 all to count)");
    }
};

struct test_count_if
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        auto host_first1 = get_host_pointer(first);
        ValueType fill_value{0};
        ::std::for_each(host_first1, host_first1 + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last,
                                          [](ValueType const& value) { return value % 10 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #1 arbitrary to count)");

        // check when none should be counted
        expected = 0;
        result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last,
                               [](ValueType const& value) { return value > 10; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #2 none to count)");

        // check when all should be counted
        expected = n;
        result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                               [](ValueType const& value) { return value < 10; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #3 all to count)");
    }
};

struct test_is_partitioned
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};
        if (n < 2)
            return;

        auto host_first1 = get_host_pointer(first);
        ValueType fill_value{0};
        ::std::for_each(host_first1, host_first1 + n, [&fill_value](ValueType& value) { value = ++fill_value; });
        // check sorted
        auto less_than = [](const ValueType& value) -> bool { return value < 10; };
        bool result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, less_than);
        bool expected_bool = ::std::is_partitioned(host_first1, host_first1 + n, less_than);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_partitioned (Test #1 less than)");

        auto is_odd = [](const ValueType& value) -> bool { return value % 2; };
        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, is_odd);
        expected_bool = ::std::is_partitioned(host_first1, host_first1 + n, is_odd);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_partitioned (Test #2 is odd)");

        // The code as below was added to prevent accessor destruction working with host memory
        {
            auto host_access = get_host_access(first);
            host_first1 = get_host_pointer(host_access);
            ::std::partition(host_first1, host_first1 + n, is_odd);
        }
        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, is_odd);
        expected_bool = ::std::is_partitioned(host_first1, host_first1 + n, is_odd);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_partitioned (Test #3 is odd after partition)");
    }
};

struct test_any_all_none_of
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);
        ::std::iota(host_first1, host_first1 + n, T1(0));

        // empty sequence case
        if (n == 1)
        {
            auto res0 =
                ::std::any_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(!res0, "wrong effect from any_of_0");
            res0 = ::std::none_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1, [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0, "wrong effect from none_of_0");
            res0 = ::std::all_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, first1, [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0, "wrong effect from all_of_0");
        }
        // any_of
        auto res1 =
            ::std::any_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1, "wrong effect from any_of_1");
        auto res2 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(!res2, "wrong effect from any_of_2");
        auto res3 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res3, "wrong effect from any_of_3");

        //none_of
        auto res4 = ::std::none_of(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1, [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res4, "wrong effect from none_of");

        //all_of
        auto res5 = ::std::all_of(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1, [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(n == 1 || !res5, "wrong effect from all_of");
    }
};

struct test_equal
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using T = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = T(42);
        auto new_start = n / 3;
        auto new_end = n / 2;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        ::std::fill(host_first1, host_first1 + n, value);
        ::std::fill(host_first2, host_first2 + n, T{0});
        ::std::fill(host_first2 + new_start, host_first2 + new_end, value);

        auto expected  = new_end - new_start > 0;
        auto result =
            ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + new_start, first1 + new_end, first2 + new_start);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(expected == result, "wrong effect from equal with 3 iterators");
        result = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1 + new_start, first1 + new_end,
                            first2 + new_start, first2 + new_end);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(expected == result, "wrong effect from equal with 4 iterators");
    }
};

struct test_find_if
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);
        ::std::iota(host_first1, host_first1 + n, T1(0));
        // empty sequence case
        if (n == 1)
        {
            auto res0 =
                ::std::find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from find_if_0");
            res0 = ::std::find(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1, T1(1));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from find_0");
        }
        // find_if
        auto res1 =
            ::std::find_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, [n](T1 x) { return x == n - 1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE((res1 - first1) == n - 1, "wrong effect from find_if_1");

        auto res2 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, [](T1 x) { return x == -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res2 == last1, "wrong effect from find_if_2");

        auto res3 =
            ::std::find_if(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, [](T1 x) { return x % 2 == 0; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res3 == first1, "wrong effect from find_if_3");

        //find
        auto res4 = ::std::find(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, T1(-1));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res4 == last1, "wrong effect from find");
    }
};

struct test_find_first_of
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;

        // Reset values after previous execution
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::fill(host_first1, host_first1 + n, T1(0));

        if (n < 2)
        {
            {
                ::std::iota(host_first2, host_first2 + n, T1(5));

                auto res =
                    ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1, "Wrong effect from find_first_of_1");
            }
        }
        else if (n >= 2 && n < 10)
        {
            {
                auto res =
                    ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_2");
            }

            // No matches
            {
                host_first2 = get_host_pointer(first2);
                ::std::iota(host_first2, host_first2 + n, T1(5));

                auto res =
                    ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_3");
            }
        }
        else if (n >= 10)
        {
            ::std::iota(host_first2, host_first2 + n, T1(5));

            auto pos1 = n / 5;
            auto pos2 = 3 * n / 5;
            auto num = 3;
            {

                ::std::iota(host_first1 + pos2, host_first1 + pos2 + num, T1(7));

                auto res =
                    ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1 + pos2, "Wrong effect from find_first_of_4");
            }

            // Add second match
            {
                host_first1 = get_host_pointer(first1);
                ::std::iota(host_first1 + pos1, host_first1 + pos1 + num, T1(6));

                auto res =
                    ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1 + pos1, "Wrong effect from find_first_of_5");
            }
        }
    }
};

struct test_search
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::iota(host_first1, host_first1 + n, T1(5));
        ::std::iota(host_first2, host_first2 + n, T1(0));
        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from search_00");
            res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0 == first1, "wrong effect from search_01");
        }
        auto res1 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1 == last1, "wrong effect from search_1");
        if (n > 10)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::search(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 10, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res2 - first1 == 5, "wrong effect from search_2");
        }
        // subsequence consists of one element (last one)
        auto res3 = ::std::search(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, last1 - 1, last1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(last1 - res3 == 1, "wrong effect from search_3");

        // first sequence contains 2 almost similar parts
        if (n > 5)
        {
            ::std::iota(host_first1 + n / 2, host_first1 + n, T1(5));
            auto res4 = ::std::search(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2 + 5, first2 + 6);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res4 == first1, "wrong effect from search_4");
        }
    }
};

struct test_search_n
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator>::value_type T;
        auto host_first1 = get_host_pointer(first);
        ::std::iota(host_first1, host_first1 + n, T(5));

        // Search for sequence at the end
        {
            auto start = (n > 3) ? (n / 3 * 2) : (n - 1);

            ::std::fill(host_first1 + start, host_first1 + n, T(11));
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, n - start, T(11));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start, "wrong effect from search_1");
        }
        // Search for sequence in the middle
        {
            auto start = (n > 3) ? (n / 3) : (n - 1);
            auto end = (n > 3) ? (n / 3 * 2) : n;
            host_first1 = get_host_pointer(first);
            ::std::fill(host_first1 + start, host_first1 + end, T(22));
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, end - start, T(22));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start, "wrong effect from search_20");

            // Search for sequence of lesser size
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                                ::std::max(end - start - 1, (size_t)1), T(22));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start, "wrong effect from search_21");
        }
        // Search for sequence at the beginning
        {
            auto end = n / 3;
            host_first1 = get_host_pointer(first);
            ::std::fill(host_first1, host_first1 + end, T(33));
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last, end, T(33));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first, "wrong effect from search_3");
        }
        // Search for sequence that covers the whole range
        {
            host_first1 = get_host_pointer(first);
            ::std::fill(host_first1, host_first1 + n, T(44));
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, n, T(44));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first, "wrong effect from search_4");
        }
        // Search for sequence which is not there
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 5>>(exec), first, last, 2, T(55));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last, "wrong effect from search_50");

            // Sequence is there but of lesser size(see search_n_3)
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 6>>(exec), first, last, (n / 3 + 1), T(33));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last, "wrong effect from search_51");
        }

        // empty sequence case
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 7>>(exec), first, first, 1, T(5 + n - 1));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first, "wrong effect from search_6");
        }
        // 2 distinct sequences, must find the first one
        if (n > 10)
        {
            auto start1 = n / 6;
            auto end1 = n / 3;

            auto start2 = (2 * n) / 3;
            auto end2 = (5 * n) / 6;
            host_first1 = get_host_pointer(first);
            ::std::fill(host_first1 + start1, host_first1 + end1, T(66));
            ::std::fill(host_first1 + start2, host_first1 + end2, T(66));

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 8>>(exec), first, last,
                                     ::std::min(end1 - start1, end2 - start2), T(66));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res - first == start1, "wrong effect from search_7");
        }

        if (n == 10)
        {
            auto seq_len = 3;
            host_first1 = get_host_pointer(first);
            // Should fail when searching for sequence which is placed before our first iterator.
            ::std::fill(host_first1, host_first1 + seq_len, T(77));
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 9>>(exec), first + 1, last, seq_len, T(77));
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last, "wrong effect from search_8");
        }
    }
};

struct test_mismatch
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::iota(host_first1, host_first1 + n, T1(5));
        ::std::iota(host_first2, host_first2 + n, T1(0));
        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_00");
            res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_01");
        }
        auto res1 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1.first == first1 && res1.second == first2, "wrong effect from mismatch_1");
        if (n > 5)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 5, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res2.first == last1 - 5 && res2.second == last2, "wrong effect from mismatch_2");
        }
    }
};

struct test_transform_inclusive_scan
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, T1(1));

        auto res1 = ::std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2,
                                                  ::std::plus<T1>(), [](T1 x) { return x * 2; }, value);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last2, "wrong result from transform_inclusive_scan_1");
        auto host_first2 = get_host_pointer(first2);
        auto ii = value;
        for (int i = 0; i < last2 - first2; ++i)
        {
            ii += 2*host_first1[i];
            if (host_first2[i] != ii)
            {
                ::std::cout << "Error in scan_1: i = " << i << ", expected " << ii << ", got " << host_first2[i]
                          << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == ii, "wrong effect from transform_inclusive_scan_1");
        }

        // without initial value
        auto res2 = ::std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                                  ::std::plus<T1>(), [](T1 x) { return x * 2; });
        EXPECT_TRUE(res2 == last2, "wrong result from transform_inclusive_scan_2");
        host_first2 = get_host_pointer(first2);
        ii = 0;
        for (int i = 0; i < last2 - first2; ++i)
        {
            ii += 2*host_first1[i];
            if (host_first2[i] != ii)
            {
                ::std::cout << "Error in scan_2: i = " << i << ", expected " << ii << ", got " << host_first2[i]
                          << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == ii, "wrong effect from transform_inclusive_scan_2");
        }
    }
};

struct test_transform_exclusive_scan
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::fill(host_first1, host_first1 + n, T1(1));

        auto res1 =
            ::std::transform_exclusive_scan(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2,
                                          T1{}, ::std::plus<T1>(), [](T1 x) { return x * 2; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last2, "wrong result from transform_exclusive_scan");

        auto ii = T1(0);
        host_first2 = get_host_pointer(first2);
        for (size_t i = 0; i < last2 - first2; ++i)
        {
            if (host_first2[i] != ii)
            {
                ::std::cout << "Error: i = " << i << ", expected " << ii << ", got " << host_first2[i] << ::std::endl;
            }
            //EXPECT_TRUE(host_first2[i] == ii, "wrong effect from transform_exclusive_scan");
            ii += 2*host_first1[i];
        }
    }
};

struct test_copy_if
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        auto host_first1 = get_host_pointer(first1);
        ::std::iota(host_first1, host_first1 + n, T1(222));

        auto res1 =
            ::std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, [](T1 x) { return x > -1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last2, "wrong result from copy_if_1");
        auto host_first2 = get_host_pointer(first2);
        for (int i = 0; i < res1 - first2; ++i)
        {
            auto exp = i + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_1");
        }
        auto res2 = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                 [](T1 x) { return x % 2 == 1; });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res2 == first2 + (last2 - first2) / 2, "wrong result from copy_if_2");
        host_first2 = get_host_pointer(first2);
        for (int i = 0; i < res2 - first2; ++i)
        {
            auto exp = 2 * i + 1 + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_2: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_2");
        }
    }
};

struct test_remove
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator>::value_type T1;
        auto host_first1 = get_host_pointer(first);
        ::std::iota(host_first1, host_first1 + n, T1(222));
        auto pos = (last - first) / 2;
        auto res1 = ::std::remove(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, T1(222 + pos));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last - 1, "wrong result from remove");
        host_first1 = get_host_pointer(first);
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove");
        }
    }
};

struct test_remove_if
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator>::value_type T1;
        auto host_first1 = get_host_pointer(first);
        ::std::iota(host_first1, host_first1 + n, T1(222));
        auto pos = (last - first) / 2;
        auto res1 = ::std::remove_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last,
                                   [=](T1 x) { return x == T1(222 + pos); });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res1 == last - 1, "wrong result from remove_if");
        host_first1 = get_host_pointer(first);
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove_if");
        }
    }
};

struct test_unique_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto f = [](Iterator1ValueType a, Iterator1ValueType b) { return a == b; };
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        // init
        int index = 0;
        ::std::for_each(host_first1, host_first1 + n, [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });

        ::std::fill(host_first2, host_first2 + n, Iterator1ValueType{-1});

        // invoke
        auto result_first = first2;
        auto result_last =
            ::std::unique_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, result_first, f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto result_size = result_last - result_first;

        int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
        host_first2 = get_host_pointer(first2);

        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first2 + i) != i + 1)
            {
                is_correct = false;
                ::std::cout << "got: " << *(host_first2 + i) << "[" << i << "], "
                          << "expected: " << i + 1 << "[" << i << "]" << ::std::endl;
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique_copy");
        }
    }
};

struct test_unique
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto f = [](IteratorValueType a, IteratorValueType b) { return a == b; };
        auto host_first1 = get_host_pointer(first);

        // init
        int index = 0;
        ::std::for_each(host_first1, host_first1 + n, [&index](IteratorValueType& value) { value = (index++ + 4) / 4; });

        // invoke
        auto result_last = ::std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto result_size = result_last - first;

        int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
        host_first1 = get_host_pointer(first);
        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first1 + i) != i + 1)
            {
                is_correct = false;
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique");
        }
    }
};

struct test_partition_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;
        using Iterator3ValueType = typename ::std::iterator_traits<Iterator3>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto host_first3 = get_host_pointer(first3);
        auto f = [](Iterator1ValueType value) { return (value % 3 == 0) && (value % 2 == 0); };

        // init
        ::std::iota(host_first1, host_first1 + n, Iterator1ValueType{0});
        ::std::fill(host_first2, host_first2 + n, Iterator2ValueType{-1});
        ::std::fill(host_first3, host_first3 + n, Iterator3ValueType{-2});

        // invoke
        auto res =
            ::std::partition_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first3, f);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        host_first3 = get_host_pointer(first3);
        // init for expected

        ::std::vector<Iterator2ValueType> exp_true(n, -1);
        ::std::vector<Iterator3ValueType> exp_false(n, -2);
        auto exp_true_first = exp_true.begin();
        auto exp_false_first = exp_false.begin();

        // invoke for expected
        auto exp = ::std::partition_copy(host_first1, host_first1 + n, exp_true_first, exp_false_first, f);

        // check
        bool is_correct = (exp.first - exp_true_first) == (res.first - first2) &&
                          (exp.second - exp_false_first) == (res.second - first3);

        for (int i = 0; i < ::std::min(exp.first - exp_true_first, res.first - first2) && is_correct; ++i)
        {
            if (*(exp_true_first + i) != *(host_first2 + i))
            {
                is_correct = false;
            }
        }

        for (int i = 0; i < ::std::min(exp.second - exp_false_first, res.second - first3) && is_correct; ++i)
        {
            if (*(exp_false_first + i) != *(host_first3 + i))
            {
                is_correct = false;
            }
        }

        EXPECT_TRUE(is_correct, "wrong effect from partition_copy");
    }
};

struct test_partition
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto unary_op = [](IteratorValueType value) { return (value % 3 == 0) && (value % 2 == 0); };
        auto host_first1 = get_host_pointer(first);

        // init
        ::std::iota(host_first1, host_first1 + n, IteratorValueType{0});

        // invoke partition
        auto res = ::std::partition(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, unary_op);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first);
        EXPECT_TRUE(::std::all_of(host_first1, host_first1 + (res - first), unary_op) &&
                        !::std::any_of(host_first1 + (res - first), host_first1 + n, unary_op),
                    "wrong effect from partition");
        // init
        ::std::iota(host_first1, host_first1 + n, IteratorValueType{0});

        // invoke stable_partition
        res = ::std::stable_partition(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, unary_op);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_first1 = get_host_pointer(first);
        EXPECT_TRUE(::std::all_of(host_first1, host_first1 + (res - first), unary_op) &&
                        !::std::any_of(host_first1 + (res - first), host_first1 + n, unary_op) &&
                        ::std::is_sorted(host_first1, host_first1 + (res - first)) &&
                        ::std::is_sorted(host_first1 + (res - first), host_first1 + n),
                    "wrong effect from stable_partition");
    }
};

struct test_is_heap_until
{

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto host_first1 = get_host_pointer(first);
        ::std::iota(host_first1, host_first1 + n, ValueType(0));

        ::std::make_heap(host_first1, host_first1);
        auto actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        // first element is always a heap
        EXPECT_TRUE(actual == first + 1, "wrong result of is_heap_until_1");

        if (n <= 5)
            return;
        host_first1 = get_host_pointer(first);
        ::std::make_heap(host_first1, host_first1 + n / 2);
        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == (first + n / 2), "wrong result of is_heap_until_2");

        host_first1 = get_host_pointer(first);
        ::std::make_heap(host_first1, host_first1 + n);
        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(actual == last, "wrong result of is_heap_until_3");
    }
};

struct test_is_heap
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto host_first1 = get_host_pointer(first);
        ::std::iota(host_first1, host_first1 + n, ValueType(0));
        {
            ::std::make_heap(host_first1, host_first1);

            auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
            // True only when n == 1
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(actual == (n == 1), "wrong result of is_heap_11");

            actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, first);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(actual == true, "wrong result of is_heap_12");
        }

        if (n <= 5)
            return;

        {
            host_first1 = get_host_pointer(first);
            auto end = first + n / 2;
            ::std::make_heap(host_first1, host_first1 + n / 2);

            auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(actual == false, "wrong result of is_heap_21");

            actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, end);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(actual == true, "wrong result of is_heap_22");
        }

        {
            host_first1 = get_host_pointer(first);
            ::std::make_heap(host_first1, host_first1 + n);
            auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(actual == true, "wrong result of is_heap_3");
        }
    }
};

struct test_inplace_merge
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator>::value_type T;
        auto host_first1 = get_host_pointer(first);
        auto value = T(0);
        ::std::iota(host_first1, host_first1 + n, value);
        ::std::vector<T> exp(n);
        ::std::iota(exp.begin(), exp.end(), value);

        auto middle = ::std::stable_partition(host_first1, host_first1 + n, [](const T& x) { return x % 2; });

        ::std::inplace_merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + (middle - host_first1), last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        host_first1 = get_host_pointer(first);
        for (size_t i = 0; i < n; ++i)
        {
            if (host_first1[i] != exp[i])
            {
                ::std::cout << "Error: i = " << i << ", expected " << exp[i] << ", got " << host_first1[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first1[i] == exp[i], "wrong effect from inplace_merge");
        }
    }
};

struct test_merge
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;
        typedef typename ::std::iterator_traits<Iterator3>::value_type T3;
        auto value = T1(0);
        auto x = n > 1 ? n / 2 : n;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::iota(host_first1, host_first1 + n, value);
        ::std::iota(host_first2, host_first2 + n, T2(value));
        ::std::vector<T3> exp(2 * n);
        auto res1 = ::std::merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2 + x, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto exp1 = ::std::merge(host_first1, host_first1 + n, host_first2, host_first2 + x, exp.begin());
        auto host_first3 = get_host_pointer(first3);
        EXPECT_TRUE(res1 - first3 == exp1 - exp.begin(), "wrong result from merge_1");
        EXPECT_TRUE(::std::is_sorted(host_first3, host_first3 + (res1 - first3)), "wrong effect from merge_1");
    }
};

struct test_sort
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);
        auto host_first1 = get_host_pointer(first1);
        ::std::iota(host_first1, host_first1 + n, value);

        ::std::sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from sort_1");

        ::std::sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, ::std::greater<T1>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T1>()), "wrong effect from sort_2");
    }
};

struct test_stable_sort
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);
        auto host_first1 = get_host_pointer(first1);
        ::std::iota(host_first1, host_first1 + n, value);

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from stable_sort_1");

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, ::std::greater<T1>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T1>()),
                    "wrong effect from stable_sort_3");
    }
};

struct test_partial_sort
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        if (n <= 1)
            return;

        auto value = T1(333);
        auto init = value;
        auto host_first1 = get_host_pointer(first1);
        ::std::generate(host_first1, host_first1 + n, [&init]() { return init--; });

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end1 = first1 + end_idx;
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, end1, last1);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif

            // Make sure that elements up to end are sorted and remaining elements are bigger
            // than the last sorted one.
            host_first1 = get_host_pointer(first1);
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + end_idx), "wrong effect from partial_sort_1");

            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n,
                                   [&](T1 val) { return val >= *(host_first1 + end_idx - 1); });
            EXPECT_TRUE(res, "wrong effect from partial_sort_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, last1);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            host_first1 = get_host_pointer(first1);
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from partial_sort_2");
        }
    }
};

struct test_partial_sort_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        if (n <= 1)
            return;

        auto value = T1(333);
        auto init = value;
        auto host_first1 = get_host_pointer(first1);
        ::std::generate(host_first1, host_first1 + n, [&init]() { return init--; });

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end2 = first2 + end_idx;

            auto last_sorted =
                ::std::partial_sort_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, end2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(last_sorted == end2, "wrong effect from partial_sort_copy_1");
            auto host_first2 = get_host_pointer(first2);
            // Make sure that elements up to end2 are sorted
            EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + end_idx), "wrong effect from partial_sort_copy_1");

            // Now ensure that the original sequence wasn't changed by partial_sort_copy
            init = value;
            auto res = ::std::all_of(host_first1, host_first1 + n, [&init](T1 val) { return val == init--; });
            EXPECT_TRUE(res, "original sequence was changed by partial_sort_copy_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            auto last_sorted =
                ::std::partial_sort_copy(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif

            EXPECT_TRUE(last_sorted == last2, "wrong effect from partial_sort_copy_2");
            auto host_first2 = get_host_pointer(first2);
            EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + n), "wrong effect from partial_sort_copy_2");

            // Now ensure that partial_sort_copy hasn't change the unsorted part of original sequence
            init = value - end_idx;
            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n, [&init](T1 val) { return val == init--; });
            EXPECT_TRUE(res, "original sequence was changed by partial_sort_copy_2");
        }
    }
};

struct test_find_end
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;

        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        // Reset after previous run
        ::std::fill(host_first1, host_first1 + n, T1(0));

        if (n <= 2)
        {
            ::std::iota(host_first2, host_first2 + n, T2(10));

            // Empty subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last1, "Wrong effect from find_end_1");

            return;
        }

        if (n > 2 && n < 10)
        {
            // re-write the sequence after previous run
            ::std::iota(host_first1, host_first1 + n, T1(0));

            ::std::iota(host_first2, host_first2 + n, T2(10));

            // No subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2 + n / 2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == last1, "Wrong effect from find_end_2");

            // Whole sequence is matched
            host_first1 = get_host_pointer(first1);
            ::std::iota(host_first1, host_first1 + n, T1(10));
            res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            EXPECT_TRUE(res == first1, "Wrong effect from find_end_3");

            return;
        }

        if (n >= 10)
        {
            ::std::iota(host_first2, host_first2 + n / 5, T2(20));

            // Match at the beginning
            {
                auto start = host_first1;
                auto end = host_first1 + n / 5;

                ::std::iota(start, end, T1(20));

                auto res =
                    ::std::find_end(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2, first2 + n / 5);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1, "Wrong effect from find_end_4");
            }

            // 2 matches: at the beginning and in the middle, should return the latter
            {
                host_first1 = get_host_pointer(first1);
                auto start = host_first1 + 2 * n / 5;
                auto end = host_first1 + 3 * n / 5;

                ::std::iota(start, end, T1(20));
                auto res =
                    ::std::find_end(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2, first2 + n / 5);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1 + 2 * n / 5, "Wrong effect from find_end_5");
            }

            // 3 matches: at the beginning, in the middle and at the end, should return the latter
            {
                host_first1 = get_host_pointer(first1);
                auto start = host_first1 + 4 * n / 5;
                auto end = host_first1 + n;

                ::std::iota(start, end, T1(20));
                auto res =
                    ::std::find_end(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2, first2 + n / 5);
#if _PSTL_SYCL_TEST_USM
                exec.queue().wait_and_throw();
#endif
                EXPECT_TRUE(res == first1 + 4 * n / 5, "Wrong effect from find_end_6");
            }
        }
    }
};

// TODO: move unique cases to test_lexicographical_compare
struct test_lexicographical_compare
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        // INIT
        ValueType fill_value1{0};
        ::std::for_each(host_first1, host_first1 + n, [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
        ValueType fill_value2{0};
        ::std::for_each(host_first2, host_first2 + n, [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });

        auto comp = [](ValueType const& first, ValueType const& second) { return first < second; };

        // CHECK 1.1: S1 == S2 && len(S1) == len(S2)
        bool is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                                                        first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 1.1: S1 == S2 && len(S1) == len(S2)");

        // CHECK 1.2: S1 == S2 && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 1)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 1" << ::std::endl;
        EXPECT_TRUE(is_less_res == 1, "wrong effect from lex_compare Test 1.2: S1 == S2 && len(S1) < len(S2)");

        // CHECK 1.3: S1 == S2 && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 1.3: S1 == S2 && len(S1) > len(S2)");

        host_first2 = get_host_pointer(first2);
        if (n > 1)
            *(host_first2 + n - 2) = 222;

        // CHECK 2.1: S1 < S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        bool is_less_exp = n > 1 ? 1 : 0;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 2.1: S1 < S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 2.2: S1 < S2 (PRE-LAST ELEMENT) && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 2.2: S1 < S2 (PRE-LAST) && len(S1) > len(S2)");

        host_first1 = get_host_pointer(first1);
        if (n > 1)
            *(host_first1 + n - 2) = 333;

        // CHECK 3.1: S1 > S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0,
                    "wrong effect from lex_compare Test 3.1: S1 > S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 3.2: S1 > S2 (PRE-LAST ELEMENT) && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        is_less_exp = n > 1 ? 0 : 1;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 3.2: S1 > S2 (PRE-LAST) && len(S1) < len(S2)");
        host_first2 = get_host_pointer(first2);
        *host_first2 = 444;

        // CHECK 4.1: S1 < S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 1)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 1" << ::std::endl;
        EXPECT_TRUE(is_less_res == 1, "wrong effect from lex_compare Test 4.1: S1 < S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 4.2: S1 < S2 (FIRST ELEMENT) && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 8>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        is_less_exp = n > 1 ? 1 : 0;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 4.2: S1 < S2 (FIRST) && len(S1) > len(S2)");
        host_first1 = get_host_pointer(first1);
        *host_first1 = 555;

        // CHECK 5.1: S1 > S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 9>>(exec), first1, last1, first2,
                                                   last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 5.1: S1 > S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 5.2: S1 > S2 (FIRST ELEMENT) && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 10>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        is_less_exp = n > 1 ? 0 : 1;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 5.2: S1 > S2 (FIRST) && len(S1) < len(S2)");
    }
};

struct test_swap_ranges
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using value_type = typename ::std::iterator_traits<Iterator1>::value_type;
        using reference = typename ::std::iterator_traits<Iterator1>::reference;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        ::std::iota(host_first1, host_first1 + n, value_type(0));
        ::std::iota(host_first2, host_first2 + n, value_type(n));

        Iterator2 actual_return = ::std::swap_ranges(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        bool check_return = (actual_return == last2);
        EXPECT_TRUE(check_return, "wrong result of swap_ranges");
        if (check_return)
        {
            ::std::size_t i = 0;
            host_first1 = get_host_pointer(first1);
            host_first2 = get_host_pointer(first2);
            bool check =
                ::std::all_of(host_first2, host_first2 + n, [&i](reference a) { return a == value_type(i++); }) &&
                ::std::all_of(host_first1, host_first1 + n, [&i](reference a) { return a == value_type(i++); });

            EXPECT_TRUE(check, "wrong effect of swap_ranges");
        }
    }
};

struct test_nth_element
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using T1 = typename ::std::iterator_traits<Iterator1>::value_type;
        using T2 = typename ::std::iterator_traits<Iterator2>::value_type;

        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto comp = ::std::less<T1>{};

        // init
        auto value1 = T1(0);
        auto value2 = T2(0);
        ::std::for_each(host_first1, host_first1 + n, [&value1](T1& val) { val = (value1++ % 10) + 1; });
        ::std::for_each(host_first2, host_first2 + n, [&value2](T2& val) { val = (value2++ % 10) + 1; });

        auto middle1 = first1 + n / 2;
        auto middle2 = first2 + n / 2;

        // invoke
        ::std::nth_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, middle1, last1, comp);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        ::std::nth_element(host_first2, host_first2 + n / 2, host_first2 + n, comp);

        // check
        host_first1 = get_host_pointer(first1);
        auto median = *(host_first1 + n / 2);
        bool is_correct = median == *(host_first2 + n / 2);
        if (!is_correct)
        {
            ::std::cout << "wrong nth element value got: " << median << ", expected: " << *(host_first2 + n / 2)
                      << ::std::endl;
        }
        is_correct =
            ::std::find_first_of(host_first1, host_first1 + n / 2, host_first1 + n / 2, host_first1 + n,
                               [comp](T1& x, T2& y) { return comp(y, x); }) ==
                     host_first1 + n / 2;
        EXPECT_TRUE(is_correct, "wrong effect from nth_element");
    }
};

struct test_reverse
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        auto host_first1 = get_host_pointer(first);
        ::std::vector<IteratorValyeType> local_copy(host_first1, host_first1 + n);

        ::std::reverse(local_copy.begin(), local_copy.end());
        ::std::reverse(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first);
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from reverse");
    }
};

struct test_reverse_copy
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator1 result_first, Iterator1 result_last, Size n)
    {
        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        auto host_first1 = get_host_pointer(first);
        ::std::vector<IteratorValyeType> local_copy(host_first1, host_first1 + n);

        ::std::reverse(local_copy.begin(), local_copy.end());
        ::std::reverse_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, result_first);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first2 = get_host_pointer(result_first);
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(local_copy[i] == host_first2[i], "wrong effect from reverse_copy");
    }
};

struct test_rotate
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        auto host_first1 = get_host_pointer(first);
        ::std::vector<IteratorValyeType> local_copy(host_first1, host_first1 + n);

        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());
        ::std::rotate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first);

        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from rotate");
    }
};

struct test_rotate_copy
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator1 result_first, Iterator1 result_last, Size n)
    {
        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        auto host_first1 = get_host_pointer(first);
        ::std::vector<IteratorValyeType> local_copy(host_first1, host_first1 + n);

        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());
        ::std::rotate_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last, result_first);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto host_first2 = get_host_pointer(result_first);
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(local_copy[i] == host_first2[i], "wrong effect from rotate_copy");
    }
};

int a[] = {0, 0, 1, 1, 2, 6, 6, 9, 9};
int b[] = {0, 1, 1, 6, 6, 9};
int c[] = {0, 1, 6, 6, 6, 9, 9};
int d[] = {7, 7, 7, 8};
int e[] = {11, 11, 12, 16, 19};
constexpr auto na = sizeof(a) / sizeof(a[0]);
constexpr auto nb = sizeof(b) / sizeof(b[0]);
constexpr auto nc = sizeof(c) / sizeof(c[0]);
constexpr auto nd = sizeof(d) / sizeof(d[0]);
constexpr auto ne = sizeof(e) / sizeof(e[0]);

struct test_includes
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        //first test case
        last1 = first1 + na;
        last2 = first2 + nb;
        ::std::copy(a, a + na, host_first1);
        ::std::copy(b, b + nb, host_first2);
        auto result = ::std::includes(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result, "wrong effect from includes a, b");

        host_first2 = get_host_pointer(first2);
        ::std::copy(c, c + nc, host_first2);
        result = ::std::includes(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        EXPECT_TRUE(!result, "wrong effect from includes a, c");
    }
};

struct test_set_intersection
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        //first test case
        last1 = first1 + na;
        last2 = first2 + nb;
        ::std::copy(a, a + na, host_first1);
        ::std::copy(b, b + nb, host_first2);

        last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2,
                                      first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto nres = last3 - first3;

        EXPECT_TRUE(nres == 6, "wrong size of intersection of a, b");

        auto host_first3 = get_host_pointer(first3);
        auto result = ::std::includes(host_first1, host_first1 + na, host_first3, host_first3 + nres) &&
                      ::std::includes(host_first2, host_first2 + nb, host_first3, host_first3 + nres);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(result, "wrong effect from set_intersection a, b");

        { //second test case

            last2 = first2 + nd;
            host_first1 = get_host_pointer(first1);
            host_first2 = get_host_pointer(first2);

            ::std::copy(a, a + na, host_first1);
            ::std::copy(d, d + nd, host_first2);

            last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                          last2, first3);
#if _PSTL_SYCL_TEST_USM
            exec.queue().wait_and_throw();
#endif
            auto nres = last3 - first3;
            EXPECT_TRUE(nres == 0, "wrong size of intersection of a, d");
        }
    }
};

struct test_set_difference
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {

        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        last1 = first1 + na;
        last2 = first2 + nb;
        ::std::copy(a, a + na, host_first1);
        ::std::copy(b, b + nb, host_first2);

        last3 = ::std::set_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto nres = last3 - first3;

        int res_expect[na];
        auto host_first3 = get_host_pointer(first3);
        auto nres_expect =
            ::std::set_difference(host_first1, host_first1 + na, host_first2, host_first2 + nb, res_expect) - res_expect;
        EXPECT_EQ_N(host_first3, res_expect, nres_expect, "wrong effect from set_difference a, b");
    }
};

struct test_set_union
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        last1 = first1 + na;
        last2 = first2 + nb;
        ::std::copy(a, a + na, host_first1);
        ::std::copy(b, b + nb, host_first2);

        last3 = ::std::set_union(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto nres = last3 - first3;

        int res_expect[na + nb];
        auto host_first3 = get_host_pointer(first3);
        auto nres_expect =
            ::std::set_union(host_first1, host_first1 + na, host_first2, host_first2 + nb, res_expect) - res_expect;
        EXPECT_EQ_N(host_first3, res_expect, nres_expect, "wrong effect from set_union a, b");
    }
};

struct test_set_symmetric_difference
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        last1 = first1 + na;
        last2 = first2 + nb;
        ::std::copy(a, a + na, host_first1);
        ::std::copy(b, b + nb, host_first2);

        last3 = ::std::set_symmetric_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                                              first2, last2, first3);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto nres = last3 - first3;

        int res_expect[na + nb];
        auto host_first3 = get_host_pointer(first3);
        auto nres_expect =
            ::std::set_symmetric_difference(host_first1, host_first1 + na, host_first2, host_first2 + nb, res_expect) -
            res_expect;
        EXPECT_EQ_N(host_first3, res_expect, nres_expect, "wrong effect from set_symmetric_difference a, b");
    }
};
#endif

int32_t
main()
{
#if _PSTL_BACKEND_SYCL
    // test1buffer
    PRINT_DEBUG("test_for_each");
    test1buffer<int32_t, test_for_each>();
    PRINT_DEBUG("test_for_each_n");
    test1buffer<int32_t, test_for_each_n>();
    PRINT_DEBUG("test_replace");
    test1buffer<int32_t, test_replace>();
    PRINT_DEBUG("test_replace_if");
    test1buffer<int32_t, test_replace_if>();
    PRINT_DEBUG("test_fill");
    test1buffer<int32_t, test_fill>();
    PRINT_DEBUG("test_fill_n");
    test1buffer<int32_t, test_fill_n>();
    PRINT_DEBUG("test_generate");
    test1buffer<int32_t, test_generate>();
    PRINT_DEBUG("test_generate_n");
    test1buffer<int32_t, test_generate_n>();
    PRINT_DEBUG("test_reduce");
    test1buffer<int32_t, test_reduce>();
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<int32_t, test_transform_reduce_unary>();
    PRINT_DEBUG("test_any_all_none_of");
    test1buffer<int32_t, test_any_all_none_of>();
    PRINT_DEBUG("test_is_sorted");
    test1buffer<int32_t, test_is_sorted>();
    PRINT_DEBUG("test_count");
    test1buffer<int32_t, test_count>();
    PRINT_DEBUG("test_count_if");
    test1buffer<int32_t, test_count_if>();
    PRINT_DEBUG("test_is_partitioned");
    test1buffer<int32_t, test_is_partitioned>();
    PRINT_DEBUG("test_sort");
    test1buffer<int32_t, test_sort>();
    PRINT_DEBUG("test_min_element");
    test1buffer<int32_t, test_min_element>();
    PRINT_DEBUG("test_max_element");
    test1buffer<int32_t, test_max_element>();
    PRINT_DEBUG("test_minmax_element");
    test1buffer<int32_t, test_minmax_element>();
    PRINT_DEBUG("test_inplace_merge");
    test1buffer<int32_t, test_inplace_merge>();
    PRINT_DEBUG("test_reverse");
    test1buffer<int32_t, test_reverse>();
    PRINT_DEBUG("test_rotate");
    test1buffer<int32_t, test_rotate>();
    PRINT_DEBUG("test_partition");
    test1buffer<int32_t, test_partition>();
    PRINT_DEBUG("test_is_heap");
    test1buffer<int32_t, test_is_heap>();
    PRINT_DEBUG("test_find_if");
    test1buffer<int32_t, test_find_if>();
    PRINT_DEBUG("test_adjacent_find");
    test1buffer<int32_t, test_adjacent_find>();
    PRINT_DEBUG("test_is_sorted_until");
    test1buffer<int32_t, test_is_sorted_until>();
    PRINT_DEBUG("test_search_n");
    test1buffer<int32_t, test_search_n>();
    PRINT_DEBUG("test_remove");
    test1buffer<int32_t, test_remove>();
    PRINT_DEBUG("test_remove_if");
    test1buffer<int32_t, test_remove_if>();
    PRINT_DEBUG("test_stable_sort");
    test1buffer<int32_t, test_stable_sort>();
    PRINT_DEBUG("test_unique");
    test1buffer<int32_t, test_unique>();
    PRINT_DEBUG("test_is_heap_until");
    test1buffer<int32_t, test_is_heap_until>();
    PRINT_DEBUG("test_uninitialized_fill");
    test1buffer<int32_t, test_uninitialized_fill>();
    PRINT_DEBUG("test_uninitialized_fill_n");
    test1buffer<int32_t, test_uninitialized_fill_n>();
    PRINT_DEBUG("test_uninitialized_default_construct");
    test1buffer<SyclTypeWrapper<int32_t>, test_uninitialized_default_construct>();
    PRINT_DEBUG("test_uninitialized_default_construct_n");
    test1buffer<SyclTypeWrapper<int32_t>, test_uninitialized_default_construct_n>();
    PRINT_DEBUG("test_uninitialized_value_construct");
    test1buffer<int32_t, test_uninitialized_value_construct>();
    PRINT_DEBUG("test_uninitialized_value_construct_n");
    test1buffer<int32_t, test_uninitialized_value_construct_n>();
    print_debug("test_is_heap");
    test1buffer<int32_t, test_is_heap>();
    PRINT_DEBUG("test_destroy");
    test1buffer<SyclTypeWrapper<int32_t>, test_destroy>();
#if !_SYCL_ARITHMETIC_TYPES_DESTRUCTION_BROKEN
    test1buffer<int32_t, test_destroy>();
#endif
    PRINT_DEBUG("test_destroy_n");
    test1buffer<SyclTypeWrapper<int32_t>, test_destroy_n>();
#if !_SYCL_ARITHMETIC_TYPES_DESTRUCTION_BROKEN
    test1buffer<int32_t, test_destroy_n>();
#endif

    //test2buffers
    PRINT_DEBUG("test_nth_element");
    test2buffers<int32_t, test_nth_element>();
    PRINT_DEBUG("test_replace_copy");
    test2buffers<int32_t, test_replace_copy>();
    PRINT_DEBUG("test_replace_copy_if");
    test2buffers<int32_t, test_replace_copy_if>();
    PRINT_DEBUG("test_transform_unary");
    test2buffers<int32_t, test_transform_unary>();
    PRINT_DEBUG("test_transform_binary");
    test2buffers<int32_t, test_transform_binary>();
    PRINT_DEBUG("test_copy");
    test2buffers<int32_t, test_copy>();
    PRINT_DEBUG("test_copy_n");
    test2buffers<int32_t, test_copy_n>();
    PRINT_DEBUG("test_move");
    test2buffers<int32_t, test_move>();
    PRINT_DEBUG("test_adjacent_difference");
    test2buffers<int32_t, test_adjacent_difference>();
    PRINT_DEBUG("test_transform_reduce_binary");
    test2buffers<int32_t, test_transform_reduce_binary>();
    PRINT_DEBUG("test_equal");
    test2buffers<int32_t, test_equal>();
    PRINT_DEBUG("test_mismatch");
    test2buffers<int32_t, test_mismatch>();
    PRINT_DEBUG("test_swap_ranges");
    test2buffers<int32_t, test_swap_ranges>();
    PRINT_DEBUG("test_reverse_copy");
    test2buffers<int32_t, test_reverse_copy>();
    PRINT_DEBUG("test rotate_copy");
    test2buffers<int32_t, test_rotate_copy>();
    PRINT_DEBUG("test_uninitialized_copy");
    test2buffers<int32_t, test_uninitialized_copy>();
    PRINT_DEBUG("test_uninitialized_copy_n");
    test2buffers<int32_t, test_uninitialized_copy_n>();
    PRINT_DEBUG("test_uninitialized_move");
    test2buffers<int32_t, test_uninitialized_move>();
    PRINT_DEBUG("test_uninitialized_move_n");
    test2buffers<int32_t, test_uninitialized_move_n>();
    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<int32_t, test_lexicographical_compare>();
    PRINT_DEBUG("test_partial_sort");
    test2buffers<int32_t, test_partial_sort>();
    PRINT_DEBUG("test_partial_sort_copy");
    test2buffers<int32_t, test_partial_sort_copy>();
    PRINT_DEBUG("test_search");
    test2buffers<int32_t, test_search>();
    PRINT_DEBUG("test_transform_inclusive_scan");
    test2buffers<int32_t, test_transform_inclusive_scan>();
    PRINT_DEBUG("test_transform_exclusive_scan");
    test2buffers<int32_t, test_transform_exclusive_scan>();
    PRINT_DEBUG("test_copy_if");
    test2buffers<int32_t, test_copy_if>();
    PRINT_DEBUG("test_unique_copy");
    test2buffers<int32_t, test_unique_copy>();
    PRINT_DEBUG("test_find_end");
    test2buffers<int32_t, test_find_end>();
    PRINT_DEBUG("test_find_first_of");
    test2buffers<int32_t, test_find_first_of>();
    PRINT_DEBUG("test_includes");
    test2buffers<int32_t, test_includes>();

    //test3buffers
    PRINT_DEBUG("test_partition_copy");
    test3buffers<int32_t, test_partition_copy>();
    PRINT_DEBUG("test_set_symmetric_difference");
    test3buffers<int32_t, test_set_symmetric_difference>();
    PRINT_DEBUG("test_set_union");
    test3buffers<int32_t, test_set_union>();
    PRINT_DEBUG("test_set_difference");
    test3buffers<int32_t, test_set_difference>();
    PRINT_DEBUG("test_set_intersection");
    test3buffers<int32_t, test_set_intersection>();
    PRINT_DEBUG("test_merge");
    test3buffers<int32_t, test_merge>(2);

#endif
    ::std::cout << done() << ::std::endl;
    return 0;
}
