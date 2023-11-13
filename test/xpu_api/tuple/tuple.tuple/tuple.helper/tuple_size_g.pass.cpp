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

#include <oneapi/dpl/tuple>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                static_assert(dpl::tuple_size<dpl::tuple<>>::value == 0);
                static_assert(dpl::tuple_size<dpl::tuple<int>>::value == 1);
                static_assert(dpl::tuple_size<dpl::tuple<void>>::value == 1);
                typedef dpl::tuple<int, const int&, void> test_tuple1;
                static_assert(dpl::tuple_size<test_tuple1>::value == 3);
                static_assert(dpl::tuple_size<dpl::tuple<dpl::tuple<void>>>::value == 1);

                static_assert(dpl::tuple_size<const dpl::tuple<>>::value == 0);
                static_assert(dpl::tuple_size<const dpl::tuple<int>>::value == 1);
                static_assert(dpl::tuple_size<const dpl::tuple<void>>::value == 1);
                static_assert(dpl::tuple_size<const test_tuple1>::value == 3);
                static_assert(dpl::tuple_size<const dpl::tuple<dpl::tuple<void>>>::value == 1);

                static_assert(dpl::tuple_size<volatile dpl::tuple<>>::value == 0);
                static_assert(dpl::tuple_size<volatile dpl::tuple<int>>::value == 1);
                static_assert(dpl::tuple_size<volatile dpl::tuple<void>>::value == 1);
                static_assert(dpl::tuple_size<volatile test_tuple1>::value == 3);
                static_assert(dpl::tuple_size<volatile dpl::tuple<dpl::tuple<void>>>::value == 1);

                static_assert(dpl::tuple_size<const volatile dpl::tuple<>>::value == 0);
                static_assert(dpl::tuple_size<const volatile dpl::tuple<int>>::value == 1);
                static_assert(dpl::tuple_size<const volatile dpl::tuple<void>>::value == 1);
                static_assert(dpl::tuple_size<const volatile test_tuple1>::value == 3);
                static_assert(dpl::tuple_size<const volatile dpl::tuple<dpl::tuple<void>>>::value == 1);
            });
        });
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
