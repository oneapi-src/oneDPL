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

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#include "has_type_member.h"

struct S
{
};
union U
{
    int i;
    float f;
};

template <typename T, typename Expected>
void
check()
{
    ASSERT_SAME_TYPE(Expected, typename dpl::underlying_type<T>::type);
    ASSERT_SAME_TYPE(Expected, typename dpl::underlying_type_t<T>);
}

enum E
{
    V = INT_MIN
};

enum G : char
{
};
enum class H
{
    red,
    green = 20,
    blue
};
enum class I : long
{
    red,
    green = 20,
    blue
};
enum struct J
{
    red,
    green = 20,
    blue
};
enum struct K : short
{
    red,
    green = 20,
    blue
};

template <typename T>
using has_underlying_type_member = has_type_member<dpl::underlying_type<T>>;

void
kernel_test1(sycl::queue& deviceQueue)
{

    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            //  Basic tests
            check<E, int>();

            //  Class enums and enums with specified underlying type
            check<G, char>();
            check<H, int>();
            check<I, long>();
            check<J, int>();
            check<K, short>();

            //  SFINAE-able underlying_type
            static_assert(has_underlying_type_member<E>::value);
            static_assert(has_underlying_type_member<G>::value);

            static_assert(!has_underlying_type_member<void>::value);
            static_assert(!has_underlying_type_member<int>::value);
            static_assert(!has_underlying_type_member<int[]>::value);
            static_assert(!has_underlying_type_member<S>::value);
            static_assert(!has_underlying_type_member<void (S::*)(int)>::value);
            static_assert(!has_underlying_type_member<void (S::*)(int, ...)>::value);
            static_assert(!has_underlying_type_member<U>::value);
            static_assert(!has_underlying_type_member<void(int)>::value);
            static_assert(!has_underlying_type_member<void(int, ...)>::value);
            static_assert(!has_underlying_type_member<int&>::value);
            static_assert(!has_underlying_type_member<int&&>::value);
            static_assert(!has_underlying_type_member<int*>::value);
            static_assert(!has_underlying_type_member<dpl::nullptr_t>::value);
        });
    });
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() { static_assert(!has_underlying_type_member<double>::value); });
    });
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);

    const auto device = deviceQueue.get_device();
    if (TestUtils::has_type_support<double>(device))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
