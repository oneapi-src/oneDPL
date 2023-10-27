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

// is_polymorphic

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class T>
void
test_is_not_polymorphic(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_polymorphic<T>::value);
            static_assert(!dpl::is_polymorphic<const T>::value);
            static_assert(!dpl::is_polymorphic<volatile T>::value);
            static_assert(!dpl::is_polymorphic<const volatile T>::value);
            static_assert(!dpl::is_polymorphic_v<T>);
            static_assert(!dpl::is_polymorphic_v<const T>);
            static_assert(!dpl::is_polymorphic_v<volatile T>);
            static_assert(!dpl::is_polymorphic_v<const volatile T>);
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

class Final final
{
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_not_polymorphic<void>(deviceQueue);
    test_is_not_polymorphic<int&>(deviceQueue);
    test_is_not_polymorphic<int>(deviceQueue);
    test_is_not_polymorphic<int*>(deviceQueue);
    test_is_not_polymorphic<const int*>(deviceQueue);
    test_is_not_polymorphic<char[3]>(deviceQueue);
    test_is_not_polymorphic<char[]>(deviceQueue);
    test_is_not_polymorphic<Union>(deviceQueue);
    test_is_not_polymorphic<Empty>(deviceQueue);
    test_is_not_polymorphic<bit_zero>(deviceQueue);
    test_is_not_polymorphic<Final>(deviceQueue);

    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_polymorphic<double>(deviceQueue);
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
