// -*- C++ -*-
//===-- zip_iterator.pass.cpp ---------------------------------------------===//
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
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(memory)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

#include "oneapi/dpl/pstl/utils.h"

#include <cmath>

#if !defined(_PSTL_TEST_FOR_EACH) && \
    !defined(_PSTL_TEST_FOR_EACH_STRUCTURED_BINDING) && \
    !defined(_PSTL_TEST_TRANSFORM_REDUCE_UNARY) && \
    !defined(_PSTL_TEST_TRANSFORM_REDUCE_BINARY) && \
    !defined(_PSTL_TEST_COUNT_IF) && \
    !defined(_PSTL_TEST_EQUAL) && \
    !defined(_PSTL_TEST_INCLUSIVE_SCAN) && \
    !defined(_PSTL_TEST_UNIQUE) && \
    !defined(_PSTL_TEST_UNIQUE_COPY) && \
    !defined(_PSTL_TEST_MERGE) && \
    !defined(_PSTL_TEST_STABLE_SORT) && \
    !defined(_PSTL_TEST_LEXICOGRAPHICAL_COMPIARE) && \
    !defined(_PSTL_TEST_COUNTING_ZIP_TRANSFORM) && \
    !defined(_PSTL_TEST_COUNTING_ZIP_DISCARD)
#define _PSTL_TEST_FOR_EACH
#define _PSTL_TEST_TRANSFORM_REDUCE_UNARY
#define _PSTL_TEST_TRANSFORM_REDUCE_BINARY
#define _PSTL_TEST_COUNT_IF
#define _PSTL_TEST_EQUAL
#define _PSTL_TEST_INCLUSIVE_SCAN
#define _PSTL_TEST_UNIQUE
#define _PSTL_TEST_UNIQUE_COPY
#define _PSTL_TEST_MERGE
#define _PSTL_TEST_STABLE_SORT
#define _PSTL_TEST_LEXICOGRAPHICAL_COMPIARE
#define _PSTL_TEST_COUNTING_ZIP_TRANSFORM
#define _PSTL_TEST_COUNTING_ZIP_DISCARD
#define _PSTL_TEST_FOR_EACH_STRUCTURED_BINDING
#define _PSTL_TEST_EQUAL_STRUCTURED_BINDING
#endif

using namespace TestUtils;

//This macro is required for the tests to work correctly in CI with tbb-backend.
#if TEST_DPCPP_BACKEND_PRESENT
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
    operator()(const T1&, const T2& t2) const
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

DEFINE_TEST(test_for_each)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(6);
        auto f = [](T1& val) { ++val; };
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(std::make_tuple(first1, first1));
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                      TuplePredicate<decltype(f), 0>{f});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1), "wrong effect from for_each(tuple)");
    }
};

#if defined(_PSTL_TEST_FOR_EACH_STRUCTURED_BINDING)

DEFINE_TEST(test_for_each_structured_binding)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each_structured_binding)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(6);
        auto f = [](T1& val) { ++val; };
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                        [f](auto value)
                        {
                            auto [x, y] = value;
                            f(x);
                            f(y);
                        });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 2), "wrong effect from for_each(tuple)");
    }
};
#endif // _PSTL_TEST_FOR_EACH_STRUCTURED_BINDING

DEFINE_TEST(test_transform_reduce_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_unary)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(1);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                ::std::make_tuple(T1{42}, T1{42}), TupleNoOp{}, TupleNoOp{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
    }
};

DEFINE_TEST(test_transform_reduce_binary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_binary)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* first2 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(1);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1,
                                  tuple_last1, tuple_first1, ::std::make_tuple(T1{42}, T1{42}), TupleNoOp{}, TupleNoOp{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
    }
};

DEFINE_TEST(test_min_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_min_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{static_cast<IteratorValueType>(n)};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        auto min_dis = n;
        if (min_dis)
        {
            auto min_it = host_keys.get() + /*min_idx*/ min_dis / 2;
            *(min_it) = IteratorValueType{/*min_val*/ 0};
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd min*/ 0};
        }
        host_keys.update_data();

        auto tuple_first = oneapi::dpl::make_zip_iterator(first, first);
        auto tuple_last = oneapi::dpl::make_zip_iterator(last, last);

        auto tuple_result =
            ::std::min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first, tuple_last,
                             TuplePredicate<::std::less<IteratorValueType>, 0>{::std::less<IteratorValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto expected_min = ::std::min_element(host_keys.get(), host_keys.get() + n);

        EXPECT_TRUE((tuple_result - tuple_first) == (expected_min - host_keys.get()),
                    "wrong effect from min_element(tuple)");
    }
};

DEFINE_TEST(test_count_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_count_if)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

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

DEFINE_TEST(test_equal)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename ::std::iterator_traits<Iterator1>::value_type;

        auto value = T(42);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        ::std::iota(host_vals.get(), host_vals.get() + n, value);
        update_data(host_keys, host_vals);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        bool is_equal = ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                   TuplePredicate<::std::equal_to<T>, 0>{::std::equal_to<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(is_equal, "wrong effect from equal(tuple) 1");

        host_vals.retrieve_data();
        *(host_vals.get() + n - 1) = T{0};
        host_vals.update_data();

        is_equal = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1, tuple_first2,
                              TuplePredicate<::std::equal_to<T>, 0>{::std::equal_to<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(!is_equal, "wrong effect from equal(tuple) 2");
    }
};

#if defined(_PSTL_TEST_EQUAL_STRUCTURED_BINDING)
DEFINE_TEST(test_equal_structured_binding)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal_structured_binding)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename ::std::iterator_traits<Iterator1>::value_type;

        auto value = T(42);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        ::std::iota(host_vals.get(), host_vals.get() + n, value);
        update_data(host_keys, host_vals);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        auto compare = [](auto tuple_first1, auto tuple_first2)
        {
            const auto& [a, b] = tuple_first1;
            const auto& [c, d] = tuple_first2;

            static_assert(::std::is_reference<decltype(a)>::value, "tuple element type is not a reference");
            static_assert(::std::is_reference<decltype(b)>::value, "tuple element type is not a reference");
            static_assert(::std::is_reference<decltype(c)>::value, "tuple element type is not a reference");
            static_assert(::std::is_reference<decltype(d)>::value, "tuple element type is not a reference");

            return (a == c) && (b == d);
        };

        bool is_equal = ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                     compare);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(is_equal, "wrong effect from equal(tuple with use of structured binding) 1");

        host_vals.retrieve_data();
        *(host_vals.get() + n - 1) = T{0};
        host_vals.update_data();

        is_equal = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                compare);
        EXPECT_TRUE(!is_equal, "wrong effect from equal(tuple with use of structured binding) 2");
    }
};
#endif // _PSTL_TEST_EQUAL_STRUCTURED_BINDING

DEFINE_TEST(test_find_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_if)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

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

DEFINE_TEST(test_transform_inclusive_scan)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_inclusive_scan)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(1));
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);
        auto value = T1(333);

        auto res = ::std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1,
                                                   tuple_last1, tuple_first2, TupleNoOp{}, TupleNoOp{},
                                                   ::std::make_tuple(value, value));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res == tuple_last2, "wrong effect from inclusive_scan(tuple)");
    }
};

DEFINE_TEST(test_unique)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        host_keys.update_data();

        const std::int64_t expected_size = (n - 1) / 4 + 1;

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        auto tuple_lastnew =
            ::std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                        TuplePredicate<::std::equal_to<Iterator1ValueType>, 0>{::std::equal_to<Iterator1ValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        bool is_correct = (tuple_lastnew - tuple_first1) == expected_size;
        host_keys.retrieve_data();
        for (int i = 0; i < ::std::min(tuple_lastnew - tuple_first1, expected_size) && is_correct; ++i)
            if ((*host_keys.get() + i) != i + 1)
                is_correct = false;

        EXPECT_TRUE(is_correct, "wrong effect from unique(tuple)");
    }
};

DEFINE_TEST(test_unique_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        ::std::fill(host_vals.get(), host_vals.get() + n, Iterator1ValueType{-1});
        update_data(host_keys, host_vals);

        const std::int64_t expected_size = (n - 1) / 4 + 1;

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        auto tuple_last2 = ::std::unique_copy(
            make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
            TuplePredicate<::std::equal_to<Iterator1ValueType>, 0>{::std::equal_to<Iterator1ValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        bool is_correct = (tuple_last2 - tuple_first2) == expected_size;
        host_vals.retrieve_data();
        for (int i = 0; i < ::std::min(tuple_last2 - tuple_first2, expected_size) && is_correct; ++i)
            if ((*host_vals.get() + i) != i + 1)
                is_correct = false;

        EXPECT_TRUE(is_correct, "wrong effect from unique_copy(tuple)");
    }
};

DEFINE_TEST(test_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_merge)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;
        typedef typename ::std::iterator_traits<Iterator3>::value_type T3;

        T1 odd = T1{1};
        T2 even = T2{0};
        size_t size1 = n >= 2 ? n / 2 : n;
        size_t size2 = n >= 3 ? n / 3 : n;
        ::std::for_each(host_keys.get(), host_keys.get() + size1,
                        [&odd](T1& value)
                        {
                            value = odd;
                            odd += 2;
                        });
        ::std::for_each(host_vals.get(), host_vals.get() + size2,
                        [&even](T2& value)
                        {
                            value = even;
                            even += 2;
                        });
        ::std::fill(host_res.get(), host_res.get() + n, T3{ -1 });
        update_data(host_keys, host_vals, host_res);

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
        TestDataTransfer<UDTKind::eRes, Size> host_res_merge(*this, res_size);

        size_t exp_size = size1 + size2;
        bool is_correct = res_size == exp_size;
        EXPECT_TRUE(is_correct, "wrong result from merge (tuple)");

        retrieve_data(host_keys, host_vals, host_res_merge);

        auto host_first1 = host_keys.get();
        auto host_first3 = host_res_merge.get();

        for (size_t i = 0; i < ::std::min(res_size, exp_size) && is_correct; ++i)
            if ((i < size2 * 2 && *(host_first3 + i) != i) ||
                (i >= size2 * 2 && *(host_first3 + i) != *(host_first1 + i - size2)))
                is_correct = false;
        EXPECT_TRUE(is_correct, "wrong effect from merge (tuple)");
    }
};

DEFINE_TEST(test_stable_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_stable_sort)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T =  typename ::std::iterator_traits<Iterator1>::value_type;

        auto value = T(333);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        ::std::copy_n(host_keys.get(), n, host_vals.get());
        update_data(host_keys, host_vals);

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), oneapi::dpl::make_zip_iterator(first1, first2),
                         oneapi::dpl::make_zip_iterator(last1, last2),
                         TuplePredicate<::std::greater<T>, 0>{::std::greater<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        retrieve_data(host_keys, host_vals);
        EXPECT_TRUE(::std::is_sorted(host_keys.get(), host_keys.get() + n, ::std::greater<T>()),
                    "wrong effect from stable_sort (tuple)");
        EXPECT_TRUE(::std::is_sorted(host_vals.get(), host_vals.get() + n, ::std::greater<T>()),
                    "wrong effect from stable_sort (tuple)");
    }
};

DEFINE_TEST(test_lexicographical_compare)
{
    DEFINE_TEST_CONSTRUCTOR(test_lexicographical_compare)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        // init
        ValueType fill_value1{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
        ValueType fill_value2{0};
        ::std::for_each(host_vals.get(), host_vals.get() + n,
                        [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });
        if (n > 1)
            *(host_vals.get() + n - 2) = 222;
        update_data(host_keys, host_vals);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);

        auto comp = [](ValueType const& first, ValueType const& second) { return first < second; };

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
    bool operator()(T x) const {
        using ::std::get;
        return get<1>(x) != 0;
    }
};

// Make sure that it's possible to use counting iterator inside zip iterator
DEFINE_TEST(test_counting_zip_transform)
{
    DEFINE_TEST_CONSTRUCTOR(test_counting_zip_transform)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        if (n < 6)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        ::std::fill(host_keys.get(), host_keys.get() + n, ValueType{0});
        ::std::fill(host_vals.get(), host_vals.get() + n, ValueType{0});
        *(host_keys.get() + (n / 3)) = 10;
        *(host_keys.get() + (n / 3 * 2)) = 100;
        update_data(host_keys, host_vals);

        auto idx = oneapi::dpl::counting_iterator<ValueType>(0);
        auto start = oneapi::dpl::make_zip_iterator(idx, first1);

        // This usage pattern can be rewritten equivalently and more simply using zip_iterator and discard_iterator,
        // see test_counting_zip_discard
        auto res =
            ::std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), start, start + n,
                           oneapi::dpl::make_transform_iterator(first2,
                                                                [](ValueType& x1)
                                                                {
                                                                    // It's required to use forward_as_tuple instead of make_tuple
                                                                    // as the latter do not propagate references.
                                                                    return ::std::forward_as_tuple(x1, ::std::ignore);
                                                                }),
                           Assigner{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(res.base() - first2 == 2, "Incorrect number of elements");
        EXPECT_TRUE(*host_vals.get() == n / 3, "Incorrect 1st element");
        EXPECT_TRUE(*(host_vals.get() + 1) == (n / 3 * 2), "Incorrect 2nd element");
    }
};

//make sure its possible to use a discard iterator in a zip iterator
DEFINE_TEST(test_counting_zip_discard)
{
    DEFINE_TEST_CONSTRUCTOR(test_counting_zip_discard)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        if (n < 6)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        ::std::fill(host_keys.get(), host_keys.get() + n, ValueType{0});
        ::std::fill(host_vals.get(), host_vals.get() + n, ValueType{0});
        *(host_keys.get() + (n / 3)) = 10;
        *(host_keys.get() + (n / 3 * 2)) = 100;
        update_data(host_keys, host_vals);

        auto idx = oneapi::dpl::counting_iterator<ValueType>(0);
        auto start = oneapi::dpl::make_zip_iterator(idx, first1);
        auto discard = oneapi::dpl::discard_iterator();
        auto out = oneapi::dpl::make_zip_iterator(first2, discard);
        auto res = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), start, start + n, out, Assigner{});

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(res - out == 2, "Incorrect number of elements");
        EXPECT_TRUE(*host_vals.get() == n / 3, "Incorrect 1st element");
        EXPECT_TRUE(*(host_vals.get() + 1) == (n / 3 * 2), "Incorrect 2nd element");
    }
};
#endif

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

#if defined(_PSTL_TEST_FOR_EACH)
    PRINT_DEBUG("test_for_each");
    test1buffer<alloc_type, test_for_each<ValueType>>();
#endif
#if defined(_PSTL_TEST_FOR_EACH_STRUCTURED_BINDING)
    PRINT_DEBUG("test_for_each_structured_binding");
    test1buffer<alloc_type, test_for_each_structured_binding<ValueType>>();
#endif
#if defined(_PSTL_TEST_TRANSFORM_REDUCE_UNARY)
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<alloc_type, test_transform_reduce_unary<ValueType>>();
#endif
#if defined(_PSTL_TEST_TRANSFORM_REDUCE_BINARY)
    PRINT_DEBUG("test_transform_reduce_binary");
    test2buffers<alloc_type, test_transform_reduce_binary<ValueType>>();
#endif
#if defined(_PSTL_TEST_COUNT_IF)
    PRINT_DEBUG("test_count_if");
    test1buffer<alloc_type, test_count_if<ValueType>>();
#endif
#if defined(_PSTL_TEST_EQUAL)
    PRINT_DEBUG("test_equal");
    test2buffers<alloc_type, test_equal<ValueType>>();
#endif
#if defined(_PSTL_TEST_EQUAL_STRUCTURED_BINDING)
    PRINT_DEBUG("test_equal_structured_binding");
    test2buffers<alloc_type, test_equal_structured_binding<ValueType>>();
#endif
#if defined(_PSTL_TEST_INCLUSIVE_SCAN)
    PRINT_DEBUG("test_inclusive_scan");
    test2buffers<alloc_type, test_transform_inclusive_scan<ValueType>>();
#endif
#if defined(_PSTL_TEST_UNIQUE)
    PRINT_DEBUG("test_unique");
    test1buffer<alloc_type, test_unique<ValueType>>();
#endif
#if defined(_PSTL_TEST_UNIQUE_COPY)
    PRINT_DEBUG("test_unique_copy");
    test2buffers<alloc_type, test_unique_copy<ValueType>>();
#endif
#if defined(_PSTL_TEST_MERGE)
    PRINT_DEBUG("test_merge");
    test3buffers<alloc_type, test_merge<ValueType>>(2);
#endif
// sorting with zip iterator does not meet limits of RAM usage on FPGA.
// TODO: try to investigate and reduce RAM consumption
#if defined(_PSTL_TEST_STABLE_SORT) && !ONEDPL_FPGA_DEVICE
    PRINT_DEBUG("test_stable_sort");
    test2buffers<alloc_type, test_stable_sort<ValueType>>();
#endif
#if defined(_PSTL_TEST_LEXICOGRAPHICAL_COMPIARE)
    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<alloc_type, test_lexicographical_compare<ValueType>>();
#endif
#if defined(_PSTL_TEST_COUNTING_ZIP_TRANSFORM)
    PRINT_DEBUG("test_counting_zip_transform");
    test2buffers<alloc_type, test_counting_zip_transform<ValueType>>();
#endif
#if defined(_PSTL_TEST_COUNTING_ZIP_DISCARD)
    PRINT_DEBUG("test_counting_zip_discard");
    test2buffers<alloc_type, test_counting_zip_discard<ValueType>>();
#endif
}
#endif // TEST_DPCPP_BACKEND_PRESENT

::std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    //TODO: There is the over-testing here - each algorithm is run with sycl::buffer as well.
    //So, in case of a couple of 'test_usm_and_buffer' call we get double-testing case with sycl::buffer.

    // Run tests for USM shared memory
    test_usm_and_buffer<sycl::usm::alloc::shared>();
    // Run tests for USM device memory
    test_usm_and_buffer<sycl::usm::alloc::device>();
#endif

    return done(TEST_DPCPP_BACKEND_PRESENT);
}
