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
template <class T>
void
test_is_copy_assignable()
{
    static_assert(dpl::is_copy_assignable<T>::value);
    static_assert(dpl::is_copy_assignable_v<T>);
}

template <class T>
void
test_is_not_copy_assignable()
{
    static_assert(!dpl::is_copy_assignable<T>::value);
    static_assert(!dpl::is_copy_assignable_v<T>);
}

class Empty
{
};

class NotEmpty
{
  public:
    virtual ~NotEmpty();
};

union Union {
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A();
};

class B
{
    B&
    operator=(const B&);
};

struct C
{
    void
    operator=(C&); // not const
};

bool
kernel_test()
{
    test_is_copy_assignable<int>();
    test_is_copy_assignable<int&>();
    test_is_copy_assignable<A>();
    test_is_copy_assignable<bit_zero>();
    test_is_copy_assignable<Union>();
    test_is_copy_assignable<NotEmpty>();
    test_is_copy_assignable<Empty>();

    test_is_not_copy_assignable<const int>();
    test_is_not_copy_assignable<int[]>();
    test_is_not_copy_assignable<int[3]>();
    test_is_not_copy_assignable<B>();
    test_is_not_copy_assignable<void>();
    test_is_not_copy_assignable<C>();
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

    EXPECT_TRUE(ret, "Wrong result of dpl::is_copy_assignable check");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
