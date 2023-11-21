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

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class T, class U>
void
test_add_rvalue_reference()
{
    ASSERT_SAME_TYPE(U, typename dpl::add_rvalue_reference<T>::type);
    ASSERT_SAME_TYPE(U, dpl::add_rvalue_reference_t<T>);
}

template <class F>
void
test_function0()
{
    ASSERT_SAME_TYPE(F&&, typename dpl::add_rvalue_reference<F>::type);
    ASSERT_SAME_TYPE(F&&, dpl::add_rvalue_reference_t<F>);
}

template <class F>
void
test_function1()
{
    ASSERT_SAME_TYPE(F, typename dpl::add_rvalue_reference<F>::type);
    ASSERT_SAME_TYPE(F, dpl::add_rvalue_reference_t<F>);
}

struct Foo
{
};

bool
kernel_test()
{
    test_add_rvalue_reference<void, void>();
    test_add_rvalue_reference<int, int&&>();
    test_add_rvalue_reference<int[3], int(&&)[3]>();
    test_add_rvalue_reference<int&, int&>();
    test_add_rvalue_reference<const int&, const int&>();
    test_add_rvalue_reference<int*, int*&&>();
    test_add_rvalue_reference<const int*, const int*&&>();
    test_add_rvalue_reference<Foo, Foo&&>();

    //  LWG 2101 specifically talks about add_rvalue_reference and functions.
    //  The term of art is "a referenceable type", which a cv- or ref-qualified function is not.
    test_function0<void()>();
    test_function1<void() const>();
    test_function1<void()&>();
    test_function1<void() &&>();
    test_function1<void() const&>();
    test_function1<void() const&&>();

    //  But a cv- or ref-qualified member function *is* "a referenceable type"
    test_function0<void (Foo::*)()>();
    test_function0<void (Foo::*)() const>();
    test_function0<void (Foo::*)()&>();
    test_function0<void (Foo::*)() &&>();
    test_function0<void (Foo::*)() const&>();
    test_function0<void (Foo::*)() const&&>();

    return true;
}

class KernelTest;

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
            cgh.single_task<class KernelNot1Test>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::add_rvalue_reference");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
