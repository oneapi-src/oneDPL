// -*- C++ -*-
//===-- zip_iterator.pass.cpp ---------------------------------------------===//
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

// just a temporary include and NoOp functor to check
// algorithms that require init element with zip_iterators
// (transform_reduce, scan, etc)
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

struct TupleNoOp
{
    template <typename T>
    T
    operator()(const T& val) const
    {
        return val;
    }

    template <typename T1, typename T2>
    T2
    operator()(const T1& t1, const T2& t2) const
    {
        return t2;
    }
};

using ::std::get;
template <typename Predicate, int KeyIndex>
struct TuplePredicate
{
    Predicate pred;

    template <typename... Args>
    auto
    operator()(const Args&... args) const -> decltype(pred(get<KeyIndex>(args)...))
    {
        return pred(get<KeyIndex>(args)...);
    }
};

using namespace oneapi::dpl::execution;

struct test_for_each
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto value = T1(6);
        auto f = [](T1& val) { ++val; };
        ::std::fill(host_first1, host_first1 + n, value);

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                      TuplePredicate<decltype(f), 0>{f});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        EXPECT_TRUE(check_values(host_first1, host_first1 + n, value + 1), "wrong effect from for_each(tuple)");
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

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto value = T1(1);
        ::std::fill(host_first1, host_first1 + n, value);

        auto tuple_result =
            ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                  ::std::make_tuple(T1{42}, T1{42}), TupleNoOp{}, TupleNoOp{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
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

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);

        auto value = T1(1);
        ::std::fill(host_first1, host_first1 + n, value);

        auto tuple_result =
            ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1,
                                  tuple_last1, tuple_first1, ::std::make_tuple(T1{42}, T1{42}), TupleNoOp{}, TupleNoOp{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
    }
};

struct test_min_element
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{static_cast<IteratorValueType>(n)};
        auto host_first1 = get_host_pointer(first);
        ::std::for_each(host_first1, host_first1 + n,
                      [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t min_dis = n;
        if (min_dis)
        {
            auto min_it = host_first1 + /*min_idx*/ min_dis / 2;
            *(min_it) = IteratorValueType{/*min_val*/ 0};
            *(host_first1 + n - 1) = IteratorValueType{/*2nd min*/ 0};
        }

        auto tuple_first = oneapi::dpl::make_zip_iterator(first, first);
        auto tuple_last = oneapi::dpl::make_zip_iterator(last, last);

        auto tuple_result =
            ::std::min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first, tuple_last,
                             TuplePredicate<::std::less<IteratorValueType>, 0>{::std::less<IteratorValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto expected_min = ::std::min_element(host_first1, host_first1 + n);

        EXPECT_TRUE((tuple_result - tuple_first) == (expected_min - host_first1),
                    "wrong effect from min_element(tuple)");
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

        auto tuple_first = oneapi::dpl::make_zip_iterator(first, first);
        auto tuple_last = oneapi::dpl::make_zip_iterator(last, last);

        auto comp = [](ValueType const& value) { return value % 10 == 0; };
        ReturnType expected = (n - 1) / 10 + 1;

        auto result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first, tuple_last,
                                    TuplePredicate<decltype(comp), 0>{comp});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        EXPECT_TRUE(result == expected, "wrong effect from count_if(tuple)");
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
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::iota(host_first1, host_first1 + n, value);
        ::std::iota(host_first2, host_first2 + n, value);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        bool is_equal = ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                   TuplePredicate<::std::equal_to<T>, 0>{::std::equal_to<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(is_equal, "wrong effect from equal(tuple) 1");

        host_first2 = get_host_pointer(first2);
        *(host_first2 + n - 1) = T{0};
        is_equal = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1, tuple_first2,
                              TuplePredicate<::std::equal_to<T>, 0>{::std::equal_to<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(!is_equal, "wrong effect from equal(tuple) 2");
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

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        auto f_for_last = [n](T1 x) { return x == n - 1; };
        auto f_for_none = [](T1 x) { return x == -1; };
        auto f_for_first = [](T1 x) { return x % 2 == 0; };

        auto tuple_res1 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                       TuplePredicate<decltype(f_for_last), 0>{f_for_last});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE((tuple_res1 - tuple_first1) == n - 1, "wrong effect from find_if_1 (tuple)");
        auto tuple_res2 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1,
                                       TuplePredicate<decltype(f_for_none), 0>{f_for_none});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(tuple_res2 == tuple_last1, "wrong effect from find_if_2 (tuple)");
        auto tuple_res3 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), tuple_first1, tuple_last1,
                                       TuplePredicate<decltype(f_for_first), 0>{f_for_first});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(tuple_res3 == tuple_first1, "wrong effect from find_if_3 (tuple)");

        // current test doesn't work with zip iterators
        auto tuple_res4 = ::std::find(make_new_policy<new_kernel_name<Policy, 3>>(exec), tuple_first1, tuple_last1,
                                    ::std::make_tuple(T1{-1}, T1{-1}));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(tuple_res4 == tuple_last1, "wrong effect from find (tuple)");
    }
};

struct test_transform_inclusive_scan
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto host_first1 = get_host_pointer(first1);
        ::std::fill(host_first1, host_first1 + n, T1(1));

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);
        auto value = T1(333);

        auto res = ::std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                                 tuple_first2, TupleNoOp{}, TupleNoOp{}, ::std::make_tuple(value, value));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res == tuple_last2, "wrong effect from inclusive_scan(tuple)");
    }
};

struct test_unique
{
    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        auto f = [](Iterator1ValueType a, Iterator1ValueType b) { return a == b; };

        int index = 0;
        ::std::for_each(host_first1, host_first1 + n, [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        int64_t expected_size = (n - 1) / 4 + 1;

        auto tuple_lastnew =
            ::std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                        TuplePredicate<::std::equal_to<Iterator1ValueType>, 0>{::std::equal_to<Iterator1ValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        bool is_correct = (tuple_lastnew - tuple_first1) == expected_size;
        host_first1 = get_host_pointer(first1);
        for (int i = 0; i < ::std::min(tuple_lastnew - tuple_first1, expected_size) && is_correct; ++i)
            if ((*host_first1 + i) != i + 1)
                is_correct = false;

        EXPECT_TRUE(is_correct, "wrong effect from unique(tuple)");
    }
};

struct test_unique_copy
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        auto f = [](Iterator1ValueType a, Iterator1ValueType b) { return a == b; };

        int index = 0;
        ::std::for_each(host_first1, host_first1 + n, [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        ::std::fill(host_first2, host_first2 + n, Iterator1ValueType{-1});
        int64_t expected_size = (n - 1) / 4 + 1;

        auto tuple_last2 =
            ::std::unique_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                             TuplePredicate<::std::equal_to<Iterator1ValueType>, 0>{::std::equal_to<Iterator1ValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        bool is_correct = (tuple_last2 - tuple_first2) == expected_size;
        host_first2 = get_host_pointer(first2);

        for (int i = 0; i < ::std::min(tuple_last2 - tuple_first2, expected_size) && is_correct; ++i)
            if ((*host_first2 + i) != i + 1)
                is_correct = false;

        EXPECT_TRUE(is_correct, "wrong effect from unique_copy(tuple)");
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

        T1 odd = T1{1};
        T2 even = T2{0};
        size_t size1 = n >= 2 ? n / 2 : n;
        size_t size2 = n >= 3 ? n / 3 : n;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        auto host_first3 = get_host_pointer(first3);
        ::std::for_each(host_first1, host_first1 + size1, [&odd](T1& value) {
            value = odd;
            odd += 2;
        });
        ::std::for_each(host_first2, host_first2 + size2, [&even](T2& value) {
            value = even;
            even += 2;
        });
        ::std::fill(host_first3, host_first3 + n, T3{-1});

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(first1 + size1, first1 + size1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(first2 + size2, first2 + size2);
        auto tuple_first3 = oneapi::dpl::make_zip_iterator(first3, first3);

        auto tuple_last3 = ::std::merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                      tuple_last2, tuple_first3, TuplePredicate<::std::less<T2>, 0>{::std::less<T2>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        size_t res_size = tuple_last3 - tuple_first3;
        size_t exp_size = size1 + size2;
        bool is_correct = res_size == exp_size;
        EXPECT_TRUE(is_correct, "wrong result from merge (tuple)");
        host_first3 = get_host_pointer(first3);
        for (size_t i = 0; i < ::std::min(res_size, exp_size) && is_correct; ++i)
            if ((i < size2 * 2 && *(host_first3 + i) != i) ||
                (i >= size2 * 2 && *(host_first3 + i) != *(host_first1 + i - size2)))
                is_correct = false;
        EXPECT_TRUE(is_correct, "wrong effect from merge (tuple)");
    }
};

struct test_stable_sort
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using T =  typename ::std::iterator_traits<Iterator1>::value_type;

        auto value = T(333);
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);
        ::std::iota(host_first1, host_first1 + n, value);
        ::std::copy_n(host_first1, n, host_first2);

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), oneapi::dpl::make_zip_iterator(first1, first2),
                         oneapi::dpl::make_zip_iterator(last1, last2),
                         TuplePredicate<::std::greater<T>, 0>{::std::greater<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first1 = get_host_pointer(first1);
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T>()),
                    "wrong effect from stable_sort (tuple)");
        EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + n, ::std::greater<T>()),
                    "wrong effect from stable_sort (tuple)");
    }
};

struct test_lexicographical_compare
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        using ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);

        auto comp = [](ValueType const& first, ValueType const& second) { return first < second; };

        // init
        ValueType fill_value1{0};
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        ::std::for_each(host_first1, host_first1 + n, [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
        ValueType fill_value2{0};
        ::std::for_each(host_first2, host_first2 + n, [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });
        if (n > 1)
            *(host_first2 + n - 2) = 222;

        bool is_less_exp = n > 1 ? 1 : 0;
        bool is_less_res =
            ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                         tuple_first2, tuple_last2, TuplePredicate<decltype(comp), 0>{comp});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp, "wrong effect from lex_compare (tuple)");
    }
};

struct Assigner{
    template<typename T>
    bool operator()(T x){
        using ::std::get;
        return get<1>(x) != 0;
    }
};

// Make sure that it's possible to use conting iterator inside zip iterator
struct test_counting_zip_transform
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {

        using ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        if (n < 6)
            return;
        auto host_first1 = get_host_pointer(first1);
        auto host_first2 = get_host_pointer(first2);

        ::std::fill(host_first1, host_first1 + n, ValueType{0});
        ::std::fill(host_first2, host_first2 + n, ValueType{0});
        *(host_first1 + (n / 3)) = 10;
        *(host_first1 + (n / 3 * 2)) = 100;

        auto idx = oneapi::dpl::counting_iterator<ValueType>(0);
        auto start = oneapi::dpl::make_zip_iterator(idx, first1);

        auto res = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), start, start + n,
                        oneapi::dpl::make_transform_iterator(
                            first2,
                            [](ValueType& x1) {
                                // It's required to use forward_as_tuple instead of make_tuple
                                // as the latter do not propagate references.
                                return ::std::forward_as_tuple(x1, ::std::ignore);
                            }),
                        Assigner{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_first2 = get_host_pointer(first2);
        EXPECT_TRUE(res.base() - first2 == 2, "Incorrect number of elements");
        EXPECT_TRUE(*host_first2 == n / 3, "Incorrect 1st element");
        EXPECT_TRUE(*(host_first2 + 1) == (n / 3 * 2), "Incorrect 2nd element");
    }
};
#endif

int32_t
main()
{
#if _PSTL_BACKEND_SYCL
    PRINT_DEBUG("test_for_each");
    test1buffer<int32_t, test_for_each>();
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<int32_t, test_transform_reduce_unary>();
    PRINT_DEBUG("test_transform_reduce_binary");
    test2buffers<int32_t, test_transform_reduce_binary>();
    PRINT_DEBUG("test_count_if");
    test1buffer<int32_t, test_count_if>();
    PRINT_DEBUG("test_equal");
    test2buffers<int32_t, test_equal>();
    PRINT_DEBUG("test_inclusive_scan");
    test2buffers<int32_t, test_transform_inclusive_scan>();
    PRINT_DEBUG("test_unique");
    test1buffer<int32_t, test_unique>();
    PRINT_DEBUG("test_unique_copy");
    test2buffers<int32_t, test_unique_copy>();
    PRINT_DEBUG("test_merge");
    test3buffers<int32_t, test_merge>();
    PRINT_DEBUG("test_stable_sort");
    test2buffers<int32_t, test_stable_sort>();
    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<int32_t, test_lexicographical_compare>();
    PRINT_DEBUG("test_counting_zip_transform");
    test2buffers<int32_t, test_counting_zip_transform>();
#endif
    ::std::cout << done() << ::std::endl;
    return 0;
}
