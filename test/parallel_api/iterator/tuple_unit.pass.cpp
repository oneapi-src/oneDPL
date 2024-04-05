// -*- C++ -*-
//===-- tuple_unit.pass.cpp -----------------------------------------------===//
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

#include _PSTL_TEST_HEADER(tuple)

#include "support/utils.h"

template <typename... T>
using tuplewrapper = oneapi::dpl::__internal::tuple<typename oneapi::dpl::__internal::__lvref_or_val<T>::__type...>;

template <typename... _T>
static oneapi::dpl::__internal::tuple<_T...>
to_onedpl_tuple(const std::tuple<_T...>& __t)
{
    return oneapi::dpl::__internal::tuple<_T...>(__t);
}

template <typename Tuple1, typename Tuple2>
void
test_tuple(Tuple1 t1, Tuple2 t2)
{

    auto onedpl_t1 = to_onedpl_tuple(t1);
    auto onedpl_t2 = to_onedpl_tuple(t2);

    static_assert(std::is_trivially_copyable_v<decltype(onedpl_t1)>, "oneDPL tuple is not trivially copyable");

    // Test binary comparison operators for std::tuple and oneAPI::dpl::__internal::tuple
    EXPECT_EQ((t1 == t2), (onedpl_t1 == onedpl_t2), "equality comparison does not match std::tuple");
    EXPECT_EQ((t1 != t2), (onedpl_t1 != onedpl_t2), "inquality comparison does not match std::tuple");
    EXPECT_EQ((t1 < t2), (onedpl_t1 < onedpl_t2), "less than comparison does not match std::tuple");
    EXPECT_EQ((t1 <= t2), (onedpl_t1 <= onedpl_t2), "less than or equal to comparison does not match std::tuple");
    EXPECT_EQ((t1 > t2), (onedpl_t1 > onedpl_t2), "greater than comparison does not match std::tuple");
    EXPECT_EQ((t1 >= t2), (onedpl_t1 >= onedpl_t2), "greater than or equal to comparison does not match std::tuple");

    EXPECT_EQ((t1 == t2), (t1 == onedpl_t2), "equality comparison does not match std::tuple");
    EXPECT_EQ((t1 != t2), (t1 != onedpl_t2), "inquality comparison does not match std::tuple");
    EXPECT_EQ((t1 < t2), (t1 < onedpl_t2), "less than comparison does not match std::tuple");
    EXPECT_EQ((t1 <= t2), (t1 <= onedpl_t2), "less than or equal to comparison does not match std::tuple");
    EXPECT_EQ((t1 > t2), (t1 > onedpl_t2), "greater than comparison does not match std::tuple");
    EXPECT_EQ((t1 >= t2), (t1 >= onedpl_t2), "greater than or equal to comparison does not match std::tuple");

    EXPECT_EQ((t1 == t2), (onedpl_t1 == t2), "equality comparison does not match std::tuple");
    EXPECT_EQ((t1 != t2), (onedpl_t1 != t2), "inquality comparison does not match std::tuple");
    EXPECT_EQ((t1 < t2), (onedpl_t1 < t2), "less than comparison does not match std::tuple");
    EXPECT_EQ((t1 <= t2), (onedpl_t1 <= t2), "less than or equal to comparison does not match std::tuple");
    EXPECT_EQ((t1 > t2), (onedpl_t1 > t2), "greater than comparison does not match std::tuple");
    EXPECT_EQ((t1 >= t2), (onedpl_t1 >= t2), "greater than or equal to comparison does not match std::tuple");

    auto onedpl_t3 = to_onedpl_tuple(t1);
    auto onedpl_t4 = to_onedpl_tuple(t2);
    auto t3 = t1;
    auto t4 = t2;

    t4 = onedpl_t1;
    EXPECT_TRUE(t1 == t4, "assignment of oneDPL tuple to std::tuple provides incorrect results");

    t3 = onedpl_t2;
    EXPECT_TRUE(t2 == t3, "assignment of oneDPL tuple to std::tuple provides incorrect results");

    onedpl_t3 = t1;
    EXPECT_TRUE(onedpl_t1 == onedpl_t3, "assignment of oneDPL tuple from std::tuple provides incorrect results");

    onedpl_t4 = t2;
    EXPECT_TRUE(onedpl_t2 == onedpl_t4, "assignment of oneDPL tuple from std::tuple provides incorrect results");

    decltype(onedpl_t1) onedpl_t5 = onedpl_t2;
    decltype(onedpl_t1) onedpl_t6 = onedpl_t1;

    swap(onedpl_t5, onedpl_t6);
    EXPECT_TRUE(((onedpl_t1 == onedpl_t5) && (onedpl_t6 == onedpl_t2)),
                "swap of oneDPL tuple provides incorrect results");
}

constexpr void
test_compile_time()
{
    constexpr std::tuple<int, int, uint64_t> std_tup1{1, 2, 3};
    constexpr std::tuple<int, uint32_t, int> std_tup2{99, 2, 3};
    constexpr oneapi::dpl::tuple<int, int, uint64_t> dpl_tup1{1, 2, 3};
    constexpr oneapi::dpl::tuple<int, uint32_t, int> dpl_tup2{99, 2, 3};

    static_assert((std_tup1 == std_tup2) == (dpl_tup1 == dpl_tup2),
                  "compile time equality comparison does not match std::tuple");
    static_assert((std_tup1 < std_tup2) == (dpl_tup1 < dpl_tup2),
                  "compile time less than comparison does not match std::tuple");
    static_assert((std_tup1 <= std_tup2) == (dpl_tup1 <= dpl_tup2),
                  "compile time lt-equal comparison does not match std::tuple");
    static_assert((std_tup1 > std_tup2) == (dpl_tup1 > dpl_tup2),
                  "compile time greater comparison does not match std::tuple");
    static_assert((std_tup1 >= std_tup2) == (dpl_tup1 >= dpl_tup2),
                  "compile time gt-equal comparison does not match std::tuple");
    static_assert((std_tup1 != std_tup2) == (dpl_tup1 != dpl_tup2),
                  "compile time non-equality comparison does not match std::tuple");

    static_assert((std_tup1 == std_tup2) == (std_tup1 == dpl_tup2),
                  "compile time equality comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_tup1 < std_tup2) == (std_tup1 < dpl_tup2),
                  "compile time less than comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_tup1 <= std_tup2) == (std_tup1 <= dpl_tup2),
                  "compile time lt-equal comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_tup1 > std_tup2) == (std_tup1 > dpl_tup2),
                  "compile time greater comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_tup1 >= std_tup2) == (std_tup1 >= dpl_tup2),
                  "compile time gt-equal comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_tup1 != std_tup2) == (std_tup1 != dpl_tup2),
                  "compile time non-equality comparison does not match std::tuple with std::tuple as lhs");

    static_assert((std_tup1 == std_tup2) == (dpl_tup1 == std_tup2),
                  "compile time equality comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_tup1 < std_tup2) == (dpl_tup1 < std_tup2),
                  "compile time less than comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_tup1 <= std_tup2) == (dpl_tup1 <= std_tup2),
                  "compile time lt-equal comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_tup1 > std_tup2) == (dpl_tup1 > std_tup2),
                  "compile time greater comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_tup1 >= std_tup2) == (dpl_tup1 >= std_tup2),
                  "compile time gt-equal comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_tup1 != std_tup2) == (dpl_tup1 != std_tup2),
                  "compile time non-equality comparison does not match std::tuple with std::tuple as rhs");

    constexpr std::tuple<> std_empty1{};
    constexpr std::tuple<> std_empty2{};
    constexpr oneapi::dpl::tuple<> dpl_empty1{};
    constexpr oneapi::dpl::tuple<> dpl_empty2{};

    static_assert((std_empty1 == std_empty2) == (dpl_empty1 == dpl_empty2),
                  "empty compile time equality comparison does not match std::tuple");
    static_assert((std_empty1 < std_empty2) == (dpl_empty1 < dpl_empty2),
                  "empty compile time less than comparison does not match std::tuple");
    static_assert((std_empty1 <= std_empty2) == (dpl_empty1 <= dpl_empty2),
                  "empty compile time lt-equal comparison does not match std::tuple");
    static_assert((std_empty1 > std_empty2) == (dpl_empty1 > dpl_empty2),
                  "empty compile time greater comparison does not match std::tuple");
    static_assert((std_empty1 >= std_empty2) == (dpl_empty1 >= dpl_empty2),
                  "empty compile time gt-equal comparison does not match std::tuple");
    static_assert((std_empty1 != std_empty2) == (dpl_empty1 != dpl_empty2),
                  "empty compile time non-equality comparison does not match std::tuple");

    static_assert((std_empty1 == std_empty2) == (std_empty1 == dpl_empty2),
                  "empty compile time equality comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_empty1 < std_empty2) == (std_empty1 < dpl_empty2),
                  "empty compile time less than comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_empty1 <= std_empty2) == (std_empty1 <= dpl_empty2),
                  "empty compile time lt-equal comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_empty1 > std_empty2) == (std_empty1 > dpl_empty2),
                  "empty compile time greater comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_empty1 >= std_empty2) == (std_empty1 >= dpl_empty2),
                  "empty compile time gt-equal comparison does not match std::tuple with std::tuple as lhs");
    static_assert((std_empty1 != std_empty2) == (std_empty1 != dpl_empty2),
                  "empty compile time non-equality comparison does not match std::tuple with std::tuple as lhs");

    static_assert((std_empty1 == std_empty2) == (dpl_empty1 == std_empty2),
                  "empty compile time equality comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_empty1 < std_empty2) == (dpl_empty1 < std_empty2),
                  "empty compile time less than comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_empty1 <= std_empty2) == (dpl_empty1 <= std_empty2),
                  "empty compile time lt-equal comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_empty1 > std_empty2) == (dpl_empty1 > std_empty2),
                  "empty compile time greater comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_empty1 >= std_empty2) == (dpl_empty1 >= std_empty2),
                  "empty compile time gt-equal comparison does not match std::tuple with std::tuple as rhs");
    static_assert((std_empty1 != std_empty2) == (dpl_empty1 != std_empty2),
                  "empty compile time non-equality comparison does not match std::tuple with std::tuple as rhs");
}

int
main()
{
    test_compile_time();

    test_tuple(std::tuple<int, int, int>{1, 2, 3}, std::tuple<int, int, int>{1, 2, 3});
    test_tuple(std::tuple<int, int, int>{1, 2, 3}, std::tuple<int, int, int>{1, 2, 4});
    test_tuple(std::tuple<int, int, int>{1, 2, 3}, std::tuple<int, int, int>{0, 2, 4});
    test_tuple(std::tuple<int, int, uint64_t>{1, 2, 3}, std::tuple<uint32_t, int, int>{1, 2, 3});
    test_tuple(std::tuple<int, int, uint64_t>{1, 2, 3}, std::tuple<uint32_t, int, int>{0, 2, 4});
    test_tuple(std::tuple<>{}, std::tuple<>{});

    return TestUtils::done();
}
