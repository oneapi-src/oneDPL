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

// union

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class T>
void
test_union_imp()
{
    static_assert(!dpl::is_reference<T>::value);
    static_assert(!dpl::is_arithmetic<T>::value);
    static_assert(!dpl::is_fundamental<T>::value);
    static_assert(dpl::is_object<T>::value);
    static_assert(!dpl::is_scalar<T>::value);
    static_assert(dpl::is_compound<T>::value);
    static_assert(!dpl::is_member_pointer<T>::value);
}

template <class T>
void
test_union()
{
    test_union_imp<T>();
    test_union_imp<const T>();
    test_union_imp<volatile T>();
    test_union_imp<const volatile T>();
}

union Union {
    int _;
    float __;
};

bool
kernel_test()
{
    test_union<Union>();
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

    EXPECT_TRUE(ret, "Wrong result of union check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return 0;
}
