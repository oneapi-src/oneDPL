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
#include <oneapi/dpl/cstddef>           // for dpl::nullptr_t

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <class KernelTest, class T>
void
test_is_compound(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(dpl::is_compound<T>::value);
            static_assert(dpl::is_compound<const T>::value);
            static_assert(dpl::is_compound<volatile T>::value);
            static_assert(dpl::is_compound<const volatile T>::value);

            static_assert(dpl::is_compound_v<T>);
            static_assert(dpl::is_compound_v<const T>);
            static_assert(dpl::is_compound_v<volatile T>);
            static_assert(dpl::is_compound_v<const volatile T>);
        });
    });
}

template <class KernelTest, class T>
void
test_is_not_compound(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
            static_assert(!dpl::is_compound<T>::value);
            static_assert(!dpl::is_compound<const T>::value);
            static_assert(!dpl::is_compound<volatile T>::value);
            static_assert(!dpl::is_compound<const volatile T>::value);

            static_assert(!dpl::is_compound_v<T>);
            static_assert(!dpl::is_compound_v<const T>);
            static_assert(!dpl::is_compound_v<volatile T>);
            static_assert(!dpl::is_compound_v<const volatile T>);
        });
    });
}

class incomplete_type;

class Empty
{
};

union Union {
};

struct bit_zero
{
    int : 0;
};

enum Enum
{
    zero,
    one
};

typedef void (*FunctionPtr)();

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;
class KernelTest17;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();;
    test_is_compound<KernelTest1, char[3]>(deviceQueue);
    test_is_compound<KernelTest2, char[]>(deviceQueue);
    test_is_compound<KernelTest3, void*>(deviceQueue);
    test_is_compound<KernelTest4, FunctionPtr>(deviceQueue);
    test_is_compound<KernelTest5, int&>(deviceQueue);
    test_is_compound<KernelTest6, int&&>(deviceQueue);
    test_is_compound<KernelTest7, Union>(deviceQueue);
    test_is_compound<KernelTest8, Empty>(deviceQueue);
    test_is_compound<KernelTest9, incomplete_type>(deviceQueue);
    test_is_compound<KernelTest10, bit_zero>(deviceQueue);
    test_is_compound<KernelTest11, int*>(deviceQueue);
    test_is_compound<KernelTest12, const int*>(deviceQueue);
    test_is_compound<KernelTest13, Enum>(deviceQueue);

    test_is_not_compound<KernelTest14, dpl::nullptr_t>(deviceQueue);
    test_is_not_compound<KernelTest15, void>(deviceQueue);
    test_is_not_compound<KernelTest16, int>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_not_compound<KernelTest17, double>(deviceQueue);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return 0;
}
