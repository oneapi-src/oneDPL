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

// type_traits

// template <class T, class... Args>
//   struct is_nothrow_constructible;

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class T>
void
test_is_nothrow_constructible()
{
    static_assert(dpl::is_nothrow_constructible<T>::value);
    static_assert(dpl::is_nothrow_constructible_v<T>);
}

template <class T, class A0>
void
test_is_nothrow_constructible()
{
    static_assert(dpl::is_nothrow_constructible<T, A0>::value);
    static_assert(dpl::is_nothrow_constructible_v<T, A0>);
}

template <class T>
void
test_is_not_nothrow_constructible()
{
    static_assert(!dpl::is_nothrow_constructible<T>::value);
    static_assert(!dpl::is_nothrow_constructible_v<T>);
}

template <class T, class A0>
void
test_is_not_nothrow_constructible()
{
    static_assert(!dpl::is_nothrow_constructible<T, A0>::value);
    static_assert(!dpl::is_nothrow_constructible_v<T, A0>);
}

template <class T, class A0, class A1>
void
test_is_not_nothrow_constructible()
{
    static_assert(!dpl::is_nothrow_constructible<T, A0, A1>::value);
    static_assert(!dpl::is_nothrow_constructible_v<T, A0, A1>);
}

class Empty
{
};

union Union {
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A(const A&);
};

struct C
{
    C(C&); // not const
    void
    operator=(C&); // not const
};

struct Tuple
{
    Tuple(Empty&&) noexcept {}
};

bool
kernel_test()
{
    test_is_nothrow_constructible<int>();
    test_is_nothrow_constructible<int, const int&>();
    test_is_nothrow_constructible<Empty>();
    test_is_nothrow_constructible<Empty, const Empty&>();

    test_is_not_nothrow_constructible<A, int>();
    test_is_not_nothrow_constructible<A, int, float>();
    test_is_not_nothrow_constructible<A>();
    test_is_not_nothrow_constructible<C>();
    test_is_nothrow_constructible<Tuple&&, Empty>(); // See bug #19616.

    static_assert(!dpl::is_constructible<Tuple&, Empty>::value);
    test_is_not_nothrow_constructible<Tuple&, Empty>(); // See bug #19616.
    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    EXPECT_TRUE(ret, "")
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
