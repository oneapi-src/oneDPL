// -*- C++ -*-
//===-- iterators.pass.cpp ------------------------------------------------===//
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

#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <type_traits>

using namespace TestUtils;

//common checks of a random access iterator functionality
template <typename RandomIt>
void test_random_iterator(const RandomIt& it) {
    // check that RandomIt has all necessary publicly accessible member types
    {
        auto t1 = typename RandomIt::difference_type{};
        auto t2 = typename RandomIt::value_type{};
        auto t3 = typename RandomIt::pointer{};
        typename RandomIt::reference ref = *it;
        (void) typename RandomIt::iterator_category{};
    }

    EXPECT_TRUE(  it == it,      "== returned false negative");
    EXPECT_TRUE(!(it == it + 1), "== returned false positive");
    EXPECT_TRUE(  it != it + 1,  "!= returned false negative");
    EXPECT_TRUE(!(it != it),     "!= returned false positive");

    EXPECT_TRUE(*it == *it, "wrong result with operator*");

    RandomIt it1 = it;
    EXPECT_TRUE(it1 == it, "iterator is not copy constructible");
    RandomIt it2 = RandomIt(it);
    EXPECT_TRUE(it2 == it, "iterator is not move constructible");

    ++it1;
    EXPECT_TRUE(it1 == it + 1, "wrong result with prefix operator++");

    using ::std::swap;
    swap(it1, it2);
    EXPECT_TRUE((it1 == it) && (it2 == it + 1), "iterator is not swappable");

    it2 = it;
    EXPECT_TRUE(it2 == it, "iterator is not copy assignable");

    ++it2;
    it2 = RandomIt(it);
    EXPECT_TRUE(it2 == it, "iterator is not move assignable");

    it1 = it;
    EXPECT_TRUE((it1++ == it) && (it1 == it + 1), "wrong result with postfix operator++");

    it1 = it + 1;
    EXPECT_TRUE(--it1 == it, "wrong result with prefix operator--");

    it1 = it + 1;
    EXPECT_TRUE((it1-- == it + 1) && (it1 == it), "wrong result with postfix operator--");

    it1 += 1;
    EXPECT_TRUE(it1 == it + 1, "wrong result with operator+=");

    it1 -= 1;
    EXPECT_TRUE(it1 == it, "wrong result with operator-=");

    EXPECT_TRUE(1 + it == it + 1, "n + iterator != iterator + n");

    EXPECT_TRUE((it + 1) - 1 == it, "wrong result with operator-(difference_type)");

    EXPECT_TRUE((it + 1) - it == 1, "wrong result with iterator subtraction");

    // There is a bug in clang when we pass the same arguments in the function
    if(it[1]!=*(it + 1)){
        ::std::cout<<"wrong result with operator[]"<<::std::endl;
        exit(1);
    }

    EXPECT_TRUE(it < it + 1, "operator< returned false negative");
    EXPECT_TRUE(!(it < it),  "operator< returned false positive");

    EXPECT_TRUE(it + 1 > it, "operator> returned false negative");
    EXPECT_TRUE(!(it > it),  "operator> returned false positive");

    EXPECT_TRUE(it <= it + 1,    "operator<= returned false negative");
    EXPECT_TRUE(it <= it,        "operator<= returned false negative");
    EXPECT_TRUE(!(it + 1 <= it), "operator<= returned false positive");

    EXPECT_TRUE(1 + it >= it,    "operator>= returned false negative");
    EXPECT_TRUE(    it >= it,    "operator>= returned false negative");
    EXPECT_TRUE(!(it >= it + 1), "operator>= returned false positive");
}

struct test_counting_iterator {
    template <typename T, typename IntType>
    void operator()( ::std::vector<T>& in, IntType begin, IntType end, const T& value) {
        EXPECT_TRUE((0 <= begin) && (begin <= end) && (end <= IntType(in.size())),
        "incorrect test_counting_iterator 'begin' and/or 'end' argument values");

        //test that counting_iterator is default constructible
        oneapi::dpl::counting_iterator<IntType> b;

        b = oneapi::dpl::counting_iterator<IntType>(begin);
        auto e = oneapi::dpl::counting_iterator<IntType>(end);

        //checks in using
        ::std::for_each(b, e, [&in, &value](IntType i) { in[i] = value; });

        auto res = ::std::all_of(in.begin(), in.begin() + begin, [&value](const T& a) {return a!=value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator in vector's begin portion");

        res = ::std::all_of(in.begin() + begin, in.begin() + end, [&value](const T& a) {return a==value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator in vector's main portion");

        res = ::std::all_of(in.begin() + end, in.end(), [&value](const T& a) {return a!=value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator in vector's end portion");

        //explicit checks of the counting iterator specific
        // There is a bug in clang when we pass the same arguments in the function
        if(b[0]!=begin){
            ::std::cout<<"wrong result with operator[] for an iterator"<<::std::endl;
            exit(1);
        }
        EXPECT_TRUE(*(b + 1) == begin+1, "wrong result with operator+ for an iterator");
        EXPECT_TRUE(*(b+=1) == begin+1, "wrong result with operator+= for an iterator");
    }
};

struct sort_fun{
    template<typename T1, typename T2>
    bool operator()(T1 a1, T2 a2) const {
        return ::std::get<0>(a1) < ::std::get<0>(a2);
    }
};

template <typename InputIterator>
void test_explicit_move(InputIterator i, InputIterator j) {
    using value_type = typename ::std::iterator_traits<InputIterator>::value_type;
    value_type t(::std::move(*i));
    *i = ::std::move(*j);
    *j = ::std::move(t);
}

struct test_zip_iterator {
    template <typename T1, typename T2>
    void operator()(::std::vector<T1>& in1, ::std::vector<T2>& in2) {
        //test that zip_iterator is default constructible
        oneapi::dpl::zip_iterator<decltype(in1.begin()), decltype(in2.begin())> b;

        b = oneapi::dpl::make_zip_iterator(in1.begin(), in2.begin());
        auto e = oneapi::dpl::make_zip_iterator(in1.end(), in2.end());

        EXPECT_TRUE( (b+1) != e, "size of input sequence insufficient for test" );

        //simple check for-loop.
        {
        ::std::for_each(b, e, [](const ::std::tuple<T1&, T2&>& a) { ::std::get<0>(a) = 1, ::std::get<1>(a) = 1;});
        auto res = ::std::all_of(b, e, [](const ::std::tuple<T1&, T2&>& a) {return ::std::get<0>(a) == 1 && ::std::get<1>(a) == 1;});
        EXPECT_TRUE(res, "wrong result sequence assignment to (1,1) with zip_iterator iterator");
        }

        //check swapping de-referenced iterators (required by sort algorithm)
        {
        using ::std::swap;
        auto t = ::std::make_tuple(T1(3), T2(2));
        *b = t;
        t = *(b + 1);
        EXPECT_TRUE(::std::get<0>(t) == 1 && ::std::get<1>(t) == 1, "wrong result of assignment from zip_iterator");
        swap(*b, *(b + 1));
        EXPECT_TRUE(::std::get<0>(*b) == 1 && ::std::get<1>(*b) == 1, "wrong result swapping zip-iterator");
        EXPECT_TRUE(::std::get<0>(*(b + 1)) == 3 && ::std::get<1>(*(b + 1)) == 2, "wrong result swapping zip-iterator");
        // Test leaves sequence un-sorted.
        }

        //sort sequences by first stream.
        {
        // sanity check if sequence is un-sorted.
        auto res = ::std::is_sorted(b, e, sort_fun());
        EXPECT_TRUE(!res, "input sequence to be sorted is already sorted! Test might lead to false positives.");
        ::std::sort(oneapi::dpl::make_zip_iterator(in1.begin(), in2.begin()),
                  oneapi::dpl::make_zip_iterator(in1.end(), in2.end()),
                  sort_fun());
        res = ::std::is_sorted(b, e, sort_fun());
        EXPECT_TRUE(res, "wrong result sorting sequence using zip-iterator");
            // TODO: Add simple check: comparison with sort_fun().
        }
        test_explicit_move(b, b+1);
        auto iter_base = b.base();
        EXPECT_TRUE(::std::get<0>(iter_base) == in1.begin(), "wrong result from base (get<0>)");
        EXPECT_TRUE(::std::get<1>(iter_base) == in2.begin(), "wrong result from base (get<1>)");

        test_random_iterator(b);
    }
};

template <typename VecIt1, typename VecIt2>
void test_transform_effect(VecIt1 first1, VecIt1 last1, VecIt2 first2) {
    auto triple = [](typename ::std::iterator_traits<VecIt1>::value_type const& val) {
        return typename ::std::iterator_traits<VecIt2>::value_type (3 * val);
    };

    ::std::copy(
        oneapi::dpl::make_transform_iterator(first1, triple),
        oneapi::dpl::make_transform_iterator(last1,  triple),
        first2
    );

    for (typename ::std::iterator_traits<VecIt1>::difference_type i = 0; i < last1 - first1; ++i)
        if ( first2[i] != (typename ::std::iterator_traits<VecIt2>::value_type) triple(first1[i]) ) {
            ::std::cout << "wrong effect with transform iterator" << ::std::endl;
            exit(1);
        }
}

struct test_transform_iterator {
    template <typename T1, typename T2>
    void operator()(::std::vector<T1>& in1, ::std::vector<T2>& in2) {
        ::std::iota(in1.begin(), in1.end(), T1(0));

        test_transform_effect(in1.begin(),  in1.end(),  in2.begin());
        test_transform_effect(in1.cbegin(), in1.cend(), in2.begin());

        auto new_transform_iterator = oneapi::dpl::make_transform_iterator(in2.begin(), [](T2& x) { return x + 1; });
        test_random_iterator(new_transform_iterator);
    }
};

template <typename T, typename IntType>
void test_iterator_by_type(IntType n) {

    const IntType beg = 0;
    const IntType end = n;

    ::std::vector<T> in(n, T(0));
    ::std::vector<IntType> in2(n, IntType(0));

    test_counting_iterator()(in, beg,     end,     /*value*/ T(-1));
    test_counting_iterator()(in, beg+123, end-321, /*value*/ T(42));
    test_random_iterator(oneapi::dpl::counting_iterator<IntType>(beg));

    test_zip_iterator()(in, in2);
    test_transform_iterator()(in, in2);
}

int main() {
    const auto n1 = 1000;
    const auto n2 = 100000;

    test_iterator_by_type<int16_t, int16_t>(n1);
    test_iterator_by_type<int16_t, int64_t>(n2);

    test_iterator_by_type<double, int16_t>(n1);
    test_iterator_by_type<double, int64_t>(n2);

    ::std::cout << done() << ::std::endl;
    return 0;
}
