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

#include "sycl_iterator_test.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <typename _Iterator1, typename _Iterator2, typename = void>
struct is_equal_exist : std::false_type
{
};

template <typename _Iterator1, typename _Iterator2>
struct is_equal_exist<_Iterator1, _Iterator2,
                      std::void_t<decltype(std::declval<::std::decay_t<_Iterator1>>().operator==(
                          std::declval<::std::decay_t<_Iterator2>>()))>> : std::true_type
{
};
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    constexpr std::size_t count = 10;

    sycl::buffer<float> buf{count};

    auto const_it = oneapi::dpl::cbegin(buf);
    auto it = oneapi::dpl::begin(buf);

    EXPECT_TRUE(const_it == it, "Wrong compare result of oneapi::dpl::cbegin(buf) and oneapi::dpl::begin(buf)");
    EXPECT_TRUE(it == const_it, "Wrong compare result of oneapi::dpl::begin(buf) and oneapi::dpl::cbegin(buf)");

    static_assert(is_equal_exist<decltype(it), decltype(const_it)>::value, "We should be able to call operator==(sycl_iterator, sycl_const_iterator)");
    static_assert(is_equal_exist<decltype(const_it), decltype(it)>::value, "We should be able to call operator==(sycl_const_iterator, sycl_iterator)");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
