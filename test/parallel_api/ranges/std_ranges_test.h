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

#include <oneapi/dpl/execution>

#include "support/test_config.h"

#include "support/utils.h"

#define _ENABLE_STD_RANGES_TESTING (_ENABLE_RANGES_TESTING && _ONEDPL___cplusplus >= 202002L)
#if _ENABLE_STD_RANGES_TESTING

#include <oneapi/dpl/ranges>
#include <span>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <string>

namespace test_std_ranges
{;

template<int call_id = 0>
auto dpcpp_policy()
{
    auto exec = TestUtils::default_dpcpp_policy;
    using Policy = decltype(exec);
    return TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, call_id>>(TestUtils::default_dpcpp_policy);
}

auto host_policies() { return std::true_type{};}

enum TestDataMode
{
    data_in,
    data_in_out,
    data_in_in,
    data_in_in_out
};

template<typename Container, TestDataMode Ranges = data_in, bool RetTypeCheck = true>
struct test
{
    template<typename Policy, typename Algo, typename... Args>
    std::enable_if_t<std::is_same_v<Policy, std::true_type>>
    operator()(Policy, Algo algo, Args... args)
    {
        operator()(oneapi::dpl::execution::seq, algo, args...);
        operator()(oneapi::dpl::execution::unseq, algo, args...);
        operator()(oneapi::dpl::execution::par, algo, args...);
        operator()(oneapi::dpl::execution::par_unseq, algo, args...);
    }

    template<typename Policy, typename Algo, typename Checker, typename FunctorOrVal, typename Proj = std::identity,
             typename Transform = std::identity>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && Ranges == data_in>
    operator()(Policy&& exec, Algo algo, Checker checker, FunctorOrVal f, Proj proj = {}, Transform tr = {})
    {
        constexpr int max_n = 10;
        int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int expected[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        auto expected_view = tr(std::ranges::subrange(expected, expected + max_n));
        auto expected_res = checker(expected_view, f, proj);
        {
            Container cont(exec, data, max_n);
            typename Container::type& A = cont();

            auto res = algo(exec, tr(A), f, proj);

            //check result
            if constexpr(RetTypeCheck)
                static_assert(std::is_same_v<decltype(res), decltype(checker(tr(A), f, proj))>, "Wrong return type");

            auto bres = ret_in_val(expected_res, expected_view.begin()) == ret_in_val(res, tr(A).begin());
            EXPECT_TRUE(bres, (std::string("wrong return value from algo with ranges: ") + typeid(Algo).name()).c_str());
        }

        //check result
        EXPECT_EQ_N(expected, data, max_n, (std::string("wrong effect algo with ranges: ")
            + typeid(Algo).name() + typeid(decltype(tr(std::declval<Container&>()()))).name()).c_str());
    }

    template<typename Policy, typename Algo, typename Checker, typename Functor, typename Proj = std::identity,
             typename Transform = std::identity>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && Ranges == data_in_out>
    operator()(Policy&& exec, Algo algo, Checker checker, Functor f, Proj proj = {}, Transform tr = {})
    {
        constexpr int max_n = 10;
        int data_in[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int data_out[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int expected[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        auto src_view = tr(std::ranges::subrange(data_in, data_in + max_n));
        auto expected_res = checker(src_view, expected, f, proj);
        {
            Container cont_in(exec, data_in, max_n);
            Container cont_out(exec, data_out, max_n);

            typename Container::type& A = cont_in();
            typename Container::type& B = cont_out();

            auto res = algo(exec, tr(A), B, f, proj);

            //check result
            if constexpr(RetTypeCheck)
                static_assert(std::is_same_v<decltype(res), decltype(checker(tr(A), B, f, proj))>, "Wrong return type");

            auto bres_in = ret_in_val(expected_res, src_view.begin()) == ret_in_val(res, tr(A).begin());
            EXPECT_TRUE(bres_in, (std::string("wrong return value from algo with input range: ") + typeid(Algo).name()).c_str());

            auto bres_out = ret_out_val(expected_res, expected) == ret_out_val(res, B.begin());
            EXPECT_TRUE(bres_out, (std::string("wrong return value from algo with output range: ") + typeid(Algo).name()).c_str());
        }

        //check result
        EXPECT_EQ_N(expected, data_out, max_n, (std::string("wrong effect algo with ranges: ") + typeid(Algo).name()).c_str());
    }

    template<typename Policy, typename Algo, typename Checker, typename Functor, typename Proj = std::identity,
             typename Transform = std::identity>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && Ranges == data_in_in>
    operator()(Policy&& exec, Algo algo, Checker checker, Functor f, Proj proj = {}, Transform tr = {})
    {
        constexpr int max_n = 10;
        int data_in1[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int data_in2[max_n] = {0, 0, 2, 3, 4, 5, 0, 0, 0, 0};

        auto src_view1 = tr(std::ranges::subrange(data_in1, data_in1 + max_n));
        auto src_view2 = tr(std::ranges::subrange(data_in2, data_in2 + max_n));
        auto expected_res = checker(src_view1, src_view2, f, proj, proj);
        {
            Container cont_in1(exec, data_in1, max_n);
            Container cont_in2(exec, data_in2, max_n);

            typename Container::type& A = cont_in1();
            typename Container::type& B = cont_in2();

            auto res = algo(exec, tr(A), tr(B), f, proj, proj);

            if constexpr(RetTypeCheck)
                static_assert(std::is_same_v<decltype(res), decltype(checker(tr(A), tr(B), f, proj, proj))>, "Wrong return type");

            auto bres_in = ret_in_val(expected_res, src_view1.begin()) == ret_in_val(res, tr(A).begin());
            EXPECT_TRUE(bres_in, (std::string("wrong return value from algo: ") + typeid(Algo).name() +
                typeid(decltype(tr(std::declval<Container&>()()))).name()).c_str());
        }
    }

private:

    template<typename, typename = void>
    static constexpr bool is_iterator{};

    template<typename T>
    static constexpr
    bool is_iterator<T, std::void_t<decltype(++std::declval<T&>()), decltype(*std::declval<T&>())>> = true;

    template<typename, typename = void>
    static constexpr bool check_in{};

    template<typename T>
    static constexpr
    bool check_in<T, std::void_t<decltype(std::declval<T>().in)>> = true;

    template<typename, typename = void>
    static constexpr bool check_out{};

    template<typename T>
    static constexpr
    bool check_out<T, std::void_t<decltype(std::declval<T>().out)>> = true;


    template<typename, typename = void>
    static constexpr bool is_range{};

    template<typename T>
    static constexpr
    bool is_range<T, std::void_t<decltype(std::declval<T&>().begin())>> = true;

    template<typename Ret, typename Begin>
    auto ret_in_val(Ret&& ret, Begin&& begin)
    {
        if constexpr (check_in<Ret>)
            return std::distance(begin, ret.in);
        else if constexpr (is_iterator<Ret>)
            return std::distance(begin, ret);
        else if constexpr(is_range<Ret>)
            return std::pair{std::distance(begin, ret.begin()), std::ranges::distance(ret.begin(), ret.end())};
        else
            return ret;
    }

    template<typename Ret, typename Begin>
    auto ret_out_val(Ret&& ret, Begin&& begin)
    {
        if constexpr (check_out<Ret>)
            return std::distance(begin, ret.out);
        else if constexpr (is_iterator<Ret>)
            return std::distance(begin, ret);
        else if constexpr(is_range<Ret>)
            return std::pair{std::distance(begin, ret.begin()), std::ranges::distance(ret.begin(), ret.end())};
        else
            return ret;
    }
};

struct sycl_buffer
{
    using type = sycl::buffer<int>;
    type buf;

    template<typename Policy>
    sycl_buffer(Policy&&, int* data, int n): buf(data, sycl::range<1>(n)) {}
    type& operator()()
    {
        return buf;
    }
};

template<typename Type>
struct host_subrange_impl
{
    using type = Type;
    type view;

    template<typename Policy>
    host_subrange_impl(Policy&&, int* data, int n): view(data, data + n) {}
    type& operator()()
    {
        return view;
    }
};

using  host_subrange = host_subrange_impl<std::ranges::subrange<int*>>;
using  host_span = host_subrange_impl<std::span<int>>;

struct host_vector
{
    using type = std::vector<int>;
    type vec;
    int* p = NULL;

    template<typename Policy>
    host_vector(Policy&&, int* data, int n): vec(data, data + n), p(data) {}
    type& operator()()
    {
        return vec;
    }
    ~host_vector()
    {
        std::copy_n(vec.begin(), vec.size(), p);
    }
};

struct usm_vector
{
    using shared_allocator = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    using type = std::vector<int, shared_allocator>;

    std::vector<int, shared_allocator> vec;
    int* p = NULL;

    template<typename Policy>
    usm_vector(Policy&& exec, int* data, int n): vec(data, data + n, shared_allocator(exec.queue())), p(data)
    {
        assert(vec.size() == n);
    }
    type& operator()()
    {
        return vec;
    }
    ~usm_vector()
    {
        std::copy_n(vec.begin(), vec.size(), p);
    }
};

template<typename Type>
struct usm_subrange_impl
{
    using shared_allocator = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
    using type = Type;

    shared_allocator alloc;
    int* p = NULL;
    type view;

    template<typename Policy>
    usm_subrange_impl(Policy&& exec, int* data, int n): alloc(exec.queue()), p(data)
    {
        auto mem = alloc.allocate(n);
        view = type(mem, mem + n);
        std::copy_n(data, n, view.data());
    }

    type& operator()()
    {
        return view;
    }

    ~usm_subrange_impl()
    {
        std::copy_n(view.data(), view.size(), p);
        alloc.deallocate(view.data(), view.size());
    }
};

using  usm_subrange = usm_subrange_impl<std::ranges::subrange<int*>>;
using  usm_span = usm_subrange_impl<std::span<int>>;

template<TestDataMode TestDataMode = data_in, bool RetTypeCheck = true>
struct test_range_algo
{
    void operator()(auto algo, auto checker, auto f, auto proj)
    {

        auto subrange_view = [](auto&& v) { return std::ranges::subrange(v); };
        auto span_view = [](auto&& v) { return std::span(v); };

        auto forward_view = [](auto&& v) {
            using forward_it = TestUtils::ForwardIterator<decltype(v.begin()), ::std::forward_iterator_tag>;
            return std::ranges::subrange(forward_it(v.begin()), forward_it(v.end()));
        };

        test<host_vector, TestDataMode, RetTypeCheck>{}(host_policies(), algo, checker, f, std::identity{}, forward_view);

        test<host_vector, TestDataMode, RetTypeCheck>{}(host_policies(), algo, checker, f, std::identity{}, subrange_view);
        test<host_vector, TestDataMode, RetTypeCheck>{}(host_policies(), algo, checker, f, std::identity{}, span_view);
        test<host_vector, TestDataMode, RetTypeCheck>{}(host_policies(), algo, checker, f, proj, std::views::all);
        test<host_subrange, TestDataMode, RetTypeCheck>{}(host_policies(), algo, checker, f, proj, std::views::all);
        test<host_span, TestDataMode, RetTypeCheck>{}(host_policies(), algo, checker,  f, proj, std::views::all);

        test<usm_vector, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, proj);
        test<usm_vector, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, proj, oneapi::dpl::views::all);
        test<usm_vector, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, proj, subrange_view);
        test<usm_vector, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, std::identity{}, span_view);
        test<usm_subrange, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, proj);
        test<usm_span, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, proj);

#if 0 //sycl buffer
        test<sycl_buffer, TestDataMode, RetTypeCheck>{}(dpcpp_policy(), algo, checker, f, std::identity{}, oneapi::dpl::views::all);
#endif
    }
};

}; //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING

