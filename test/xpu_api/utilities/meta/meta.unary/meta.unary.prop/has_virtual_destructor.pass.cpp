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

// has_virtual_destructor

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class T>
void
test_has_not_virtual_destructor(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::has_virtual_destructor<T>::value);
            static_assert(!dpl::has_virtual_destructor<const T>::value);
            static_assert(!dpl::has_virtual_destructor<volatile T>::value);
            static_assert(!dpl::has_virtual_destructor<const volatile T>::value);

            static_assert(!dpl::has_virtual_destructor_v<T>);
            static_assert(!dpl::has_virtual_destructor_v<const T>);
            static_assert(!dpl::has_virtual_destructor_v<volatile T>);
            static_assert(!dpl::has_virtual_destructor_v<const volatile T>);
        });
    });
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
    ~A();
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_has_not_virtual_destructor<void>(deviceQueue);
    test_has_not_virtual_destructor<A>(deviceQueue);
    test_has_not_virtual_destructor<int&>(deviceQueue);
    test_has_not_virtual_destructor<Union>(deviceQueue);
    test_has_not_virtual_destructor<Empty>(deviceQueue);
    test_has_not_virtual_destructor<int>(deviceQueue);
    test_has_not_virtual_destructor<int*>(deviceQueue);
    test_has_not_virtual_destructor<const int*>(deviceQueue);
    test_has_not_virtual_destructor<char[3]>(deviceQueue);
    test_has_not_virtual_destructor<char[]>(deviceQueue);
    test_has_not_virtual_destructor<bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_has_not_virtual_destructor<double>(deviceQueue);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
