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

#include <oneapi/dpl/utility>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
struct NoexceptMoveAssignClass                          // KSATODO (empty)
{
};

struct NoexceptMoveConsClass                            // KSATODO (empty)
{
};

// KSATODO struct NoexceptMoveConsNoexceptMoveAssignClass
struct NoexceptMoveConsNoexceptMoveAssignClass
{
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::pair<int, int> tt1;
                typedef dpl::pair<int, float> tt2;
                typedef dpl::pair<short, NoexceptMoveAssignClass> tt3;
                typedef dpl::pair<short, NoexceptMoveConsClass> tt4;
                typedef dpl::pair<NoexceptMoveConsClass, NoexceptMoveConsClass> tt5;
                typedef dpl::pair<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass> tt6;
                typedef dpl::pair<NoexceptMoveConsNoexceptMoveAssignClass, float> tt7;
                typedef dpl::pair<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass> tt8;

                static_assert(noexcept(dpl::declval<tt1&>().swap(dpl::declval<tt1&>())));
                static_assert(noexcept(dpl::declval<tt2&>().swap(dpl::declval<tt2&>())));
                static_assert(noexcept(dpl::declval<tt3&>().swap(dpl::declval<tt3&>())));
                static_assert(noexcept(dpl::declval<tt4&>().swap(dpl::declval<tt4&>())));
                static_assert(noexcept(dpl::declval<tt5&>().swap(dpl::declval<tt5&>())));
                static_assert(noexcept(dpl::declval<tt6&>().swap(dpl::declval<tt6&>())));
                static_assert(noexcept(dpl::declval<tt7&>().swap(dpl::declval<tt7&>())));
                static_assert(noexcept(dpl::declval<tt8&>().swap(dpl::declval<tt8&>())));
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
