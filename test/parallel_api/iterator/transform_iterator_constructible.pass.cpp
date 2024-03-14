// -*- C++ -*-
//===-- transform_iterator_constructible.pass.cpp -------------------------===//
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
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

using namespace TestUtils;

struct noop
{
    int
    operator()(int a) const
    {
        return a;
    }
};

struct noop_nodefault
{
    noop_nodefault() = delete;
    noop_nodefault(int) {}
    int
    operator()(int a) const
    {
        return a;
    }
};

void
test_default_constructible()
{
    auto transformation = [](int) { return 0; };

    int* ptr = nullptr;
    oneapi::dpl::transform_iterator<int*, decltype(transformation)> trans1{ptr, transformation};
    //default constructibility of lambdas depends on c++ standard, we want transform iterator to match its template args
    EXPECT_TRUE((std::is_default_constructible_v<decltype(transformation)> ==
                 ::std::is_default_constructible_v<decltype(trans1)>),
                "transform_iterator with lambda does not match default constructibility status of the lambda itself");

    //both types are default constructible
    oneapi::dpl::transform_iterator<int*, noop> trans2{ptr, noop{}};
    EXPECT_TRUE(::std::is_default_constructible_v<decltype(trans2)>,
                "transform_iterator with default constructible lambda is seen to be non-default constructible");

    //functor is not default constructible
    oneapi::dpl::transform_iterator<int*, noop_nodefault> trans3{ptr, noop_nodefault{1}};
    EXPECT_TRUE(!::std::is_default_constructible_v<decltype(trans3)>,
                "transform_iterator with non-default constructible lambda is seen to be default constructible");

    oneapi::dpl::transform_iterator<decltype(trans3), noop> trans4{trans3, noop{}};
    EXPECT_TRUE(
        !::std::is_default_constructible_v<decltype(trans4)>,
        "transform_iterator with non-default constructible iterator source is seen to be default constructible");
}

std::int32_t
main()
{
    test_default_constructible();

    return TestUtils::done();
}
