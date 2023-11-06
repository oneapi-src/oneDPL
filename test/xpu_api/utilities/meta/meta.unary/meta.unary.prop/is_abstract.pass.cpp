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
#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class T>
void
test_is_not_abstract(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_abstract<T>::value);
            static_assert(!dpl::is_abstract<const T>::value);
            static_assert(!dpl::is_abstract<volatile T>::value);
            static_assert(!dpl::is_abstract<const volatile T>::value);

            static_assert(!dpl::is_abstract_v<T>);
            static_assert(!dpl::is_abstract_v<const T>);
            static_assert(!dpl::is_abstract_v<volatile T>);
            static_assert(!dpl::is_abstract_v<const volatile T>);
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

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_abstract<void>(deviceQueue);
    test_is_not_abstract<int&>(deviceQueue);
    test_is_not_abstract<int>(deviceQueue);
    test_is_not_abstract<int*>(deviceQueue);
    test_is_not_abstract<const int*>(deviceQueue);
    test_is_not_abstract<char[3]>(deviceQueue);
    test_is_not_abstract<char[]>(deviceQueue);
    test_is_not_abstract<Union>(deviceQueue);
    test_is_not_abstract<Empty>(deviceQueue);
    test_is_not_abstract<bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_abstract<double>(deviceQueue);
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
