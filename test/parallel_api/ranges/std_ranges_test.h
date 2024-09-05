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
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"

#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

static_assert(ONEDPL_HAS_RANGE_ALGORITHMS >= 202409L);

#if TEST_CPP20_SPAN_PRESENT
#include <span>
#endif
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <string>
#include <ranges>

namespace test_std_ranges
{

#if TEST_DPCPP_BACKEND_PRESENT
template<int call_id = 0>
auto dpcpp_policy()
{
    auto exec = TestUtils::default_dpcpp_policy;
    using Policy = decltype(exec);
    return TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, call_id>>(TestUtils::default_dpcpp_policy);
}
#endif //TEST_DPCPP_BACKEND_PRESENT

auto host_policies() { return std::true_type{};}

enum TestDataMode
{
    data_in,
    data_in_out,
    data_in_out_lim,
    data_in_in,
    data_in_in_out,
    data_in_in_out_lim
};

auto f_mutuable = [](auto&& val) { return val *= val; };
auto proj_mutuable = [](auto&& val) { return val *= 2; };

auto f = [](auto&& val) { return val * val; };
auto binary_f = [](auto&& val1, auto&& val2) { return val1 * val2; };
auto proj = [](auto&& val){ return val * 2; };
auto pred = [](auto&& val) { return val == 5; };
auto binary_pred = [](auto&& val1, auto&& val2) { return val1 == val2; };

auto pred1 = [](auto&& val) -> decltype(auto) { return val > 0; };
auto pred2 = [](auto&& val) -> decltype(auto) { return val == 4; };
auto pred3 = [](auto&& val) -> decltype(auto) { return val < 0; };

struct P2
{
    P2() {}
    P2(int v): x(v) {}
    int x = {};
    int y = {};

    int proj() const { return x; }
    friend bool operator==(const P2& a, const P2& b) { return a.x == b.x && a.y == b.y; }
};


// These are copies of __range_size and __range_size_t utilities from oneDPL
// to get a size type of a range be it sized or not
template <typename R>
struct range_size {
    using type = std::uint8_t;
};

template <std::ranges::sized_range R>
struct range_size<R> {
    using type = std::ranges::range_size_t<R>;
};

template <typename R>
using range_size_t = typename range_size<R>::type;

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
static constexpr bool check_in1{};

template<typename T>
static constexpr
bool check_in1<T, std::void_t<decltype(std::declval<T>().in1)>> = true;

template<typename, typename = void>
static constexpr bool check_in2{};

template<typename T>
static constexpr
bool check_in2<T, std::void_t<decltype(std::declval<T>().in2)>> = true;

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

template<typename DataType, typename Container, TestDataMode test_mode = data_in>
struct test
{
    template<typename Policy>
    std::enable_if_t<std::is_same_v<Policy, std::true_type>>
    operator()(Policy, auto algo, auto& checker, auto... args)
    {
        operator()(oneapi::dpl::execution::seq, algo, checker, args...);
        operator()(oneapi::dpl::execution::unseq, algo, checker, args...);
        operator()(oneapi::dpl::execution::par, algo, checker,  args...);
        operator()(oneapi::dpl::execution::par_unseq, algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && mode == data_in>
    operator()(Policy&& exec, Algo algo, Checker& checker, TransIn tr_in, TransOut, auto... args)
    {
        constexpr int max_n = 10;
        DataType data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        DataType expected[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        auto expected_view = tr_in(std::ranges::subrange(expected, expected + max_n));
        auto expected_res = checker(expected_view, args...);
        {
            Container cont(exec, std::ranges::begin(data), max_n);
            typename Container::type& A = cont();

            auto res = algo(exec, tr_in(A), args...);

            //check result
            static_assert(std::is_same_v<decltype(res), decltype(checker(tr_in(A), args...))>, "Wrong return type");

            auto bres = ret_in_val(expected_res, expected_view.begin()) == ret_in_val(res, tr_in(A).begin());
            EXPECT_TRUE(bres, (std::string("wrong return value from algo with ranges: ") + typeid(Algo).name() + 
                typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());
        }

        //check result
        EXPECT_EQ_N(expected, data, max_n, (std::string("wrong effect algo with ranges: ")
            + typeid(Algo).name() + typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());
    }

private:
    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    void
    process_data_in_out(int n_in, int n_out, Policy&& exec, Algo algo, Checker& checker, TransIn tr_in,
                        TransOut tr_out, auto... args)
    {
        static_assert(mode == data_in_out || mode == data_in_out_lim);

        constexpr int max_n = 10;
        DataType data_in[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        DataType data_out[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        DataType expected[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        assert(n_in <= max_n);
        assert(n_out <= max_n);

        auto src_view = tr_in(std::ranges::subrange(data_in, data_in + n_in));
        auto expected_res = checker(src_view, std::ranges::subrange(expected, expected + n_out), args...);
        {
            Container cont_in(exec, data_in, n_in);
            Container cont_out(exec, data_out, n_out);

            typename Container::type& A = cont_in();
            typename Container::type& B = cont_out();

            auto res = algo(exec, tr_in(A), tr_out(B), args...);

            //check result
            static_assert(std::is_same_v<decltype(res), decltype(checker(tr_in(A), B, args...))>, "Wrong return type");

            auto bres_in = ret_in_val(expected_res, src_view.begin()) == ret_in_val(res, tr_in(A).begin());
            EXPECT_TRUE(bres_in, (std::string("wrong return value from algo with input range: ") + typeid(Algo).name()).c_str());

            auto bres_out = ret_out_val(expected_res, expected) == ret_out_val(res, B.begin());
            EXPECT_TRUE(bres_out, (std::string("wrong return value from algo with output range: ") + typeid(Algo).name()).c_str());
        }

        //check result
        EXPECT_EQ_N(expected, data_out, n_out, (std::string("wrong effect algo with ranges: ") + typeid(Algo).name()).c_str());
    }

public:
    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && mode == data_in_out>
    operator()(Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = 10;
        process_data_in_out(r_size, r_size, std::forward<Policy>(exec), algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && mode == data_in_out_lim>
    operator()(Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = 10;
        process_data_in_out(r_size, r_size, std::forward<Policy>(exec), algo, checker, args...);

        //test case size of input range is less than size of output and viceversa
        process_data_in_out(r_size/2, r_size, exec, algo, checker, args...);
        process_data_in_out(r_size, r_size/2, std::forward<Policy>(exec), algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && mode == data_in_in>
    operator()(Policy&& exec, Algo algo, Checker& checker, TransIn tr_in, TransOut, auto... args)
    {
        constexpr int max_n = 10;
        DataType data_in1[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        DataType data_in2[max_n] = {0, 0, 2, 3, 4, 5, 0, 0, 0, 0};

        auto src_view1 = tr_in(std::ranges::subrange(data_in1, data_in1 + max_n));
        auto src_view2 = tr_in(std::ranges::subrange(data_in2, data_in2 + max_n));
        auto expected_res = checker(src_view1, src_view2, args...);
        {
            Container cont_in1(exec, data_in1, max_n);
            Container cont_in2(exec, data_in2, max_n);

            typename Container::type& A = cont_in1();
            typename Container::type& B = cont_in2();

            auto res = algo(exec, tr_in(A), tr_in(B), args...);

            static_assert(std::is_same_v<decltype(res), decltype(checker(tr_in(A), tr_in(B), args...))>, "Wrong return type");

            auto bres_in = ret_in_val(expected_res, src_view1.begin()) == ret_in_val(res, tr_in(A).begin());
            EXPECT_TRUE(bres_in, (std::string("wrong return value from algo: ") + typeid(Algo).name() +
                typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());
        }
    }

private:
    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    void
    process_data_in_in_out(int n_in1, int n_in2, int n_out, Policy&& exec, Algo algo, Checker& checker, TransIn tr_in, TransOut tr_out, auto... args)
    {
        static_assert(mode == data_in_in_out || mode == data_in_in_out_lim);

        constexpr int max_n = 10;
        DataType data_in1[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        DataType data_in2[max_n] = {0, 0, 2, 3, 4, 5, 6, 6, 6, 6};
        constexpr int max_n_out = max_n*2;
        DataType data_out[max_n_out] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; //TODO: size
        DataType expected[max_n_out] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        assert(n_in1 <= max_n);
        assert(n_in2 <= max_n);
        assert(n_out <= max_n_out);
        
        auto src_view1 = tr_in(std::ranges::subrange(data_in1, data_in1 + n_in1));
        auto src_view2 = tr_in(std::ranges::subrange(data_in2, data_in2 + n_in2));
        auto expected_res = checker(src_view1, src_view2, std::ranges::subrange(expected, expected + n_out), args...);
        {
            Container cont_in1(exec, data_in1, n_in1);
            Container cont_in2(exec, data_in2, n_in2);
            Container cont_out(exec, data_out, n_out);

            typename Container::type& A = cont_in1();
            typename Container::type& B = cont_in2();
            typename Container::type& C = cont_out();

            auto res = algo(exec, tr_in(A), tr_in(B), tr_out(C), args...);

            static_assert(std::is_same_v<decltype(res), decltype(checker(tr_in(A), tr_in(B), C, args...))>, "Wrong return type");

            auto bres_in = ret_in_val(expected_res, src_view1.begin()) == ret_in_val(res, tr_in(A).begin());
            EXPECT_TRUE(bres_in, (std::string("wrong return value from algo: ") + typeid(Algo).name() +
                typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());
        }
        //check result
        EXPECT_EQ_N(expected, data_out, n_out, (std::string("wrong effect algo with ranges: ") + typeid(Algo).name()).c_str());
    }

public:
    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && mode == data_in_in_out>
    operator()(Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = 10;
        process_data_in_in_out(r_size, r_size, r_size*2, std::forward<Policy>(exec), algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<!std::is_same_v<Policy, std::true_type> && mode == data_in_in_out_lim>
    operator()(Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = 10;
        process_data_in_in_out(r_size, r_size, r_size, exec, algo, checker, args...);
        process_data_in_in_out(r_size/2, r_size, r_size, exec, algo, checker, args...);
        process_data_in_in_out(r_size, r_size/2, r_size, exec, algo, checker, args...);
        process_data_in_in_out(r_size, r_size, r_size/2, std::forward<Policy>(exec), algo, checker, args...);
    }
private:

    template<typename Ret, typename Begin>
    auto ret_in_val(Ret&& ret, Begin&& begin)
    {
        if constexpr (check_in<Ret>)
            return std::distance(begin, ret.in);
        else if constexpr (check_in1<Ret>)
            return std::distance(begin, ret.in1);
        else if constexpr (check_in2<Ret>)
            return std::distance(begin, ret.in2);
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

template<typename T, typename ViewType>
struct host_subrange_impl
{
    using type = ViewType;
    ViewType view;

    template<typename Policy>
    host_subrange_impl(Policy&&, T* data, int n): view(data, data + n) {}
    ViewType& operator()()
    {
        return view;
    }
};

template<typename T>
using  host_subrange = host_subrange_impl<T, std::ranges::subrange<T*>>;

#if TEST_CPP20_SPAN_PRESENT
template<typename T>
using  host_span = host_subrange_impl<T, std::span<T>>;
#endif

template<typename T>
struct host_vector
{
    using type = std::vector<T>;
    type vec;
    T* p = NULL;

    template<typename Policy>
    host_vector(Policy&&, T* data, int n): vec(data, data + n), p(data) {}
    type& operator()()
    {
        return vec;
    }
    ~host_vector()
    {
        std::copy_n(vec.begin(), vec.size(), p);
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
template<typename T>
struct usm_vector
{
    using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
    using type = std::vector<T, shared_allocator>;

    std::vector<T, shared_allocator> vec;
    T* p = NULL;

    template<typename Policy>
    usm_vector(Policy&& exec, T* data, int n): vec(data, data + n, shared_allocator(exec.queue())), p(data)
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

template<typename T, typename ViewType>
struct usm_subrange_impl
{
    using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
    using type = ViewType;

    shared_allocator alloc;
    T* p = NULL;
    ViewType view;

    template<typename Policy>
    usm_subrange_impl(Policy&& exec, T* data, int n): alloc(exec.queue()), p(data)
    {
        auto mem = alloc.allocate(n);
        view = ViewType(mem, mem + n);
        std::copy_n(data, n, view.data());
    }

    ViewType& operator()()
    {
        return view;
    }

    ~usm_subrange_impl()
    {
        std::copy_n(view.data(), view.size(), p);
        alloc.deallocate(view.data(), view.size());
    }
};

template<typename T>
using  usm_subrange = usm_subrange_impl<T, std::ranges::subrange<T*>>;

#if TEST_CPP20_SPAN_PRESENT
template<typename T>
using  usm_span = usm_subrange_impl<T, std::span<T>>;
#endif

#endif // TEST_DPCPP_BACKEND_PRESENT

template<int call_id = 0, typename T = int, TestDataMode mode = data_in>
struct test_range_algo
{
    void operator()(auto algo, auto& checker, auto... args)
    {

        auto subrange_view = [](auto&& v) { return std::ranges::subrange(v); };
#if TEST_CPP20_SPAN_PRESENT
        auto span_view = [](auto&& v) { return std::span(v); };
#endif

        test<T, host_vector<T>, mode>{}(host_policies(), algo, checker, std::identity{}, std::identity{}, args...);
        test<T, host_vector<T>, mode>{}(host_policies(), algo, checker, subrange_view, std::identity{}, args...);
        test<T, host_vector<T>, mode>{}(host_policies(), algo, checker, std::views::all, std::identity{}, args...);
        test<T, host_subrange<T>, mode>{}(host_policies(), algo, checker, std::views::all, std::identity{}, args...);
#if TEST_CPP20_SPAN_PRESENT
        test<T, host_vector<T>, mode>{}(host_policies(), algo, checker,  span_view, std::identity{}, args...);
        test<T, host_span<T>, mode>{}(host_policies(), algo, checker, std::views::all, std::identity{}, args...);
#endif

#if TEST_DPCPP_BACKEND_PRESENT
        //Skip the cases with pointer-to-function and hetero policy because pointer-to-function is not supported within kernel code.
        if constexpr(!std::disjunction_v<std::is_member_function_pointer<decltype(args)>...>)
        {
#if _PSTL_LAMBDA_PTR_TO_MEMBER_WINDOWS_BROKEN
            if constexpr(!std::disjunction_v<std::is_member_pointer<decltype(args)>...>)
#endif
            {
                test<T, usm_vector<T>, mode>{}(dpcpp_policy<call_id + 10>(), algo, checker, subrange_view, subrange_view, args...);
                test<T, usm_subrange<T>, mode>{}(dpcpp_policy<call_id + 30>(), algo, checker, std::identity{}, std::identity{}, args...);
#if TEST_CPP20_SPAN_PRESENT
                test<T, usm_vector<T>, mode>{}(dpcpp_policy<call_id + 20>(), algo, checker, span_view, subrange_view, args...);
                test<T, usm_span<T>, mode>{}(dpcpp_policy<call_id + 40>(), algo, checker, std::identity{}, std::identity{}, args...);
#endif
            }
        }
#endif //TEST_DPCPP_BACKEND_PRESENT
    }
};

}; //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
