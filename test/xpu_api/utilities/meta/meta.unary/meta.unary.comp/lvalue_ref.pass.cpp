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

template <class T>
void
test_lvalue_ref()
{
    static_assert(dpl::is_reference<T>::value);
    static_assert(!dpl::is_arithmetic<T>::value);
    static_assert(!dpl::is_fundamental<T>::value);
    static_assert(!dpl::is_object<T>::value);
    static_assert(!dpl::is_scalar<T>::value);
    static_assert(dpl::is_compound<T>::value);
    static_assert(!dpl::is_member_pointer<T>::value);
}

bool
kernel_test()
{
    test_lvalue_ref<int&>();
    test_lvalue_ref<const int&>();
    return true;
}

int
main()
{
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

    EXPECT_TRUE(ret, "Wrong result of lvalue ref check");

    return 0;
}
