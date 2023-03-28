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

#include <vector>

#if TEST_DPCPP_BACKEND_PRESENT

namespace oneapi::dpl::__internal
{
// Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
void
test_iterators_possibly_equal()
{
    // Check some internals from oneapi::dpl::__internal
    using namespace oneapi::dpl::__internal;

    constexpr size_t count = 0;
    sycl::buffer<int> buf1(count);
    sycl::buffer<int> buf2(count);

    auto it1 = oneapi::dpl::begin(buf1);
    auto it2 = oneapi::dpl::begin(buf2);
    auto& it1Ref = it1;
    auto& it2Ref = it2;

    EXPECT_TRUE(__iterators_possibly_equal(it1, it1), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__iterators_possibly_equal(it1, it1Ref), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__iterators_possibly_equal(it1Ref, it1), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__iterators_possibly_equal(it1Ref, it1Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1, it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1Ref, it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1, it2Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1Ref, it2Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), it2Ref),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), oneapi::dpl::begin(buf2)),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1, oneapi::dpl::begin(buf2)), "wrong __iterators_possibly_equal result");

    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), nullptr),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(nullptr, oneapi::dpl::begin(buf2)),
                 "wrong __iterators_possibly_equal result");

    {
        float floatData = .0;

        ::std::vector<int> dataVec{1, 2, 3};
        const auto intConstData = dataVec.data();
        auto intData = dataVec.data();

        // check pointer + pointer
        EXPECT_TRUE(__iterators_possibly_equal(intData, intData), "wrong __iterators_possibly_equal result");
        // check const pointer + pointer
        EXPECT_TRUE(__iterators_possibly_equal(intConstData, intData), "wrong __iterators_possibly_equal result");
        // check pointer + const pointer
        EXPECT_TRUE(__iterators_possibly_equal(intData, intConstData), "wrong __iterators_possibly_equal result");
        // check pointer + pointer to other type
        EXPECT_FALSE(__iterators_possibly_equal(intData, &floatData), "wrong __iterators_possibly_equal result");
    }

    {
        int srcIntData = 0;
        const auto& intConstData = srcIntData;
        auto& intData = srcIntData;
        const float floatData = .0;

        // Check pointer to const data + pointer to data
        EXPECT_TRUE(__iterators_possibly_equal(&intConstData, &intData), "wrong __iterators_possibly_equal result");
        // Check pointer to data + pointer to const data
        EXPECT_TRUE(__iterators_possibly_equal(&intData, &intConstData), "wrong __iterators_possibly_equal result");
        // Check pointer to const data + pointer to const data
        EXPECT_TRUE(__iterators_possibly_equal(&intConstData, &intConstData),
                    "wrong __iterators_possibly_equal result");
        // check pointer + pointer to other const type
        EXPECT_FALSE(__iterators_possibly_equal(intData, &floatData), "wrong __iterators_possibly_equal result");
    }
}
};

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    // Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
    oneapi::dpl::__internal::test_iterators_possibly_equal();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
