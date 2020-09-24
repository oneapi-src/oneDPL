// -*- C++ -*-
//===-- uninitialized_fill_destroy.pass.cpp -------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#include "support/pstl_test_config.h"
#include "support/utils.h"

#if  !defined(_PSTL_TEST_UNITIALIZED_FILL) && !defined(_PSTL_TEST_UNITIALIZED_FILL_N) &&\
     !defined(_PSTL_TEST_UNITIALIZED_DESTROY) && !defined(_PSTL_TEST_UNITIALIZED_DESTROY_N)
#define _PSTL_TEST_UNITIALIZED_FILL
#define _PSTL_TEST_UNITIALIZED_FILL_N
#define _PSTL_TEST_UNITIALIZED_DESTROY
#define _PSTL_TEST_UNITIALIZED_DESTROY_N
#endif

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(memory)

#include <memory>

using namespace TestUtils;

template <typename Type>
struct test_uninitialized_fill
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        uninitialized_fill(exec, first, last, in);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_TRUE(n == count, "wrong work of uninitialized_fill");

        destroy(oneapi::dpl::execution::seq, first, last);
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

        uninitialized_fill(exec, first, last, in);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_EQ(n, count, "wrong work of uninitialized_fill");
    }
};

template <typename Type>
struct test_uninitialized_fill_n
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        auto res = uninitialized_fill_n(exec, first, n, in);
        EXPECT_TRUE(res == last, "wrong result of uninitialized_fill_n");
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_TRUE(n == count, "wrong work of uninitialized_fill_n");

        destroy_n(oneapi::dpl::execution::seq, first, n);
    }
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

        auto res = uninitialized_fill_n(exec, first, n, in);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_EQ(n, count, "wrong work of uninitialized_fill_n");
        EXPECT_TRUE(res == last, "wrong result of uninitialized_fill_n");
    }
};

template <typename Type>
struct test_destroy
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        T::SetCount(0);
#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill(oneapi::dpl::execution::seq, first, last, in);
#else
        uninitialized_fill(first, last, in);
#endif
        destroy(exec, first, last);
        EXPECT_TRUE(T::Count() == 0, "wrong work of destroy");
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill(oneapi::dpl::execution::seq, first, last, in);
#else
        uninitialized_fill(first, last, in);
#endif
        destroy(exec, first, last);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x != in; });
        size_t tmp_n = 0;
        EXPECT_EQ(tmp_n, count, "wrong work of destroy");
    }
};

template <typename Type>
struct test_destroy_n
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        T::SetCount(0);
#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill_n(oneapi::dpl::execution::seq, first, n, in);
#else
        uninitialized_fill(first, last, in);
#endif
        auto dres = destroy_n(exec, first, n);
        EXPECT_TRUE(dres == last, "wrong result of destroy_n");
        EXPECT_TRUE(T::Count() == 0, "wrong work of destroy_n");
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill_n(oneapi::dpl::execution::seq, first, n, in);
#else
        uninitialized_fill(first, last, in);
#endif
        auto dres = destroy_n(exec, first, n);
        EXPECT_TRUE(dres == last, "wrong result of destroy_n");
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x != in; });
        size_t tmp_n = 0;
        EXPECT_EQ(tmp_n, count, "wrong work of destroy");
    }
};

template <typename T>
void
test_uninitialized_fill_destroy_by_type()
{
    ::std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
#if !_PSTL_BACKEND_SYCL
        ::std::unique_ptr<T[]> p(new T[n]);
        auto p_begin = p.get();
#else
        Sequence<T> p(n, [](size_t k){ return T{}; });
        auto p_begin = p.begin();
#endif
        auto p_end = ::std::next(p_begin, n);
#ifdef _PSTL_TEST_UNITIALIZED_FILL
        invoke_on_all_policies<>()(test_uninitialized_fill<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_FILL_N
        invoke_on_all_policies<>()(test_uninitialized_fill_n<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#if !_PSTL_BACKEND_SYCL
#ifdef _PSTL_TEST_UNITIALIZED_DESTROY
        invoke_on_all_policies<>()(test_destroy<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_DESTROY_N
        invoke_on_all_policies<>()(test_destroy_n<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#endif
    }
}

int
main()
{
#if !_PSTL_DPCPP_TEST_UNINITIALIZED_BROKEN
    // for trivial types
    test_uninitialized_fill_destroy_by_type<int32_t>();
    test_uninitialized_fill_destroy_by_type<float64_t>();
#endif

#if !_PSTL_BACKEND_SYCL
    // for user-defined types
    test_uninitialized_fill_destroy_by_type<Wrapper<::std::string>>();
    test_uninitialized_fill_destroy_by_type<Wrapper<int8_t*>>();
#endif
    ::std::cout << done() << ::std::endl;

    return 0;
}
