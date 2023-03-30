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

    // sub - buffer vs it's "root" buffer (expect true)
    sycl::buffer<int, 1> buf11{buf1, sycl::range<1>{0}, sycl::range<1>{0}};
    EXPECT_TRUE(__iterators_possibly_equal(oneapi::dpl::end(buf1), oneapi::dpl::begin(buf11)),
                "wrong __iterators_possibly_equal result");

    // sub - buffer vs sub - buffer which share a "root" buffer(expect true)
    sycl::buffer<int, 1> buf12{buf1, sycl::range<1>{0}, sycl::range<1>{0}};
    EXPECT_TRUE(__iterators_possibly_equal(oneapi::dpl::begin(buf11), oneapi::dpl::end(buf12)),
                "wrong __iterators_possibly_equal result");

    // two sycl_iterators pointing to different elements in the same "root" buffer(expect false)
    auto it1next = it1 + 1;
    EXPECT_FALSE(__iterators_possibly_equal(it1, it1next), "wrong __iterators_possibly_equal result");

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

void
test_sycl_const_iterator_assignment()
{
    constexpr std::size_t count = 10;

    sycl::buffer<float> buf{count};

    using TSyclConstIterator = decltype(oneapi::dpl::cbegin(buf));
    using TSyclIterator = decltype(oneapi::dpl::begin(buf));

    static_assert(::std::is_same_v<TSyclConstIterator, decltype(oneapi::dpl::cend(buf))>, "");
    static_assert(::std::is_same_v<TSyclIterator, decltype(oneapi::dpl::end(buf))>, "");

    TSyclIterator it = oneapi::dpl::begin(buf);
    TSyclConstIterator it_const = it;
    it_const = it_const;

    //TSyclIterator it1 = it_const;
    //it1;

    TSyclConstIterator it_const1(it_const);
    EXPECT_TRUE(it_const1 == it_const, "Wrong compare result of two iterators");
}

void
test_sycl_const_iterator_equal()
{
    constexpr std::size_t count = 10;

    sycl::buffer<float> buf{count};

    const auto it_cbegin = oneapi::dpl::cbegin(buf);
    const auto it_begin = oneapi::dpl::begin(buf);
    const auto it_cend = oneapi::dpl::cend(buf);
    const auto it_end = oneapi::dpl::end(buf);

    EXPECT_TRUE(it_cbegin == it_begin, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_begin == it_cbegin, "Wrong compare result of two iterators");

    EXPECT_TRUE(it_cbegin != it_end, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_end != it_cbegin, "Wrong compare result of two iterators");

    EXPECT_TRUE(it_cend == it_end, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_cbegin == it_cend, "Wrong compare result of two iterators");
}

void
test_sycl_const_iterator_not_equal()
{
    constexpr std::size_t count = 10;

    sycl::buffer<float> buf{count};

    const auto it_cbegin = oneapi::dpl::cbegin(buf);
    const auto it_begin = oneapi::dpl::begin(buf);
    const auto it_cend = oneapi::dpl::cend(buf);
    const auto it_end = oneapi::dpl::end(buf);

    EXPECT_FALSE(it_cbegin != it_begin, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_begin != it_cbegin, "Wrong compare result of two iterators");

    EXPECT_FALSE(it_cbegin == it_end, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_end == it_cbegin, "Wrong compare result of two iterators");

    EXPECT_FALSE(it_cend != it_end, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_cbegin != it_cend, "Wrong compare result of two iterators");
}

void
test_sycl_const_iterator_less()
{
    constexpr std::size_t count = 10;

    sycl::buffer<float> buf{count};

    const auto it_cbegin = oneapi::dpl::cbegin(buf);
    const auto it_cbegin_1 = oneapi::dpl::cbegin(buf) + 1;
    const auto it_begin = oneapi::dpl::begin(buf);
    const auto it_begin_1 = oneapi::dpl::begin(buf) + 1;
    const auto it_cend = oneapi::dpl::cend(buf);
    const auto it_end = oneapi::dpl::end(buf);

    EXPECT_TRUE(it_begin < it_begin_1, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_begin < it_cend, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_begin < it_end, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_cbegin < it_cend, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_cbegin < it_end, "Wrong compare result of two iterators");

    EXPECT_FALSE(it_begin_1 < it_begin, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_cend < it_begin, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_end < it_begin, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_cend < it_cbegin, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_end < it_cbegin, "Wrong compare result of two iterators");

    EXPECT_TRUE(it_cbegin < it_cbegin_1, "Wrong compare result of two iterators");
    EXPECT_TRUE(it_cbegin < it_end, "Wrong compare result of two iterators");
    EXPECT_FALSE(it_end < it_cbegin, "Wrong compare result of two iterators");
}

void
test_sycl_const_iterator_minus()
{
    constexpr std::size_t count = 10;

    sycl::buffer<float> buf{count};

    const auto it_cbegin = oneapi::dpl::cbegin(buf);
    const auto it_cbegin_1 = oneapi::dpl::cbegin(buf) + 1;
    const auto it_begin = oneapi::dpl::begin(buf);
    const auto it_begin_1 = oneapi::dpl::begin(buf) + 1;
    const auto it_cend = oneapi::dpl::cend(buf);
    const auto it_end = oneapi::dpl::end(buf);

    EXPECT_TRUE(0 == (it_cbegin - it_cbegin), "Wrong diff result of two iterators");
    EXPECT_TRUE(1 == (it_cbegin_1 - it_cbegin), "Wrong diff result of two iterators");
    EXPECT_TRUE(0 == (it_begin - it_begin), "Wrong diff result of two iterators");
    EXPECT_TRUE(1 == (it_begin_1 - it_begin), "Wrong diff result of two iterators");

    EXPECT_TRUE(count == (it_end - it_cbegin), "Wrong diff result of two iterators");
    EXPECT_TRUE(count == (it_cend - it_begin), "Wrong diff result of two iterators");
}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    // Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
    oneapi::dpl::__internal::test_iterators_possibly_equal();

    test_sycl_const_iterator_assignment();
    test_sycl_const_iterator_equal();
    test_sycl_const_iterator_not_equal();
    test_sycl_const_iterator_less();
    test_sycl_const_iterator_minus();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
