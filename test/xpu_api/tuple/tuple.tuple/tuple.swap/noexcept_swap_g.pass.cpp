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
#include <oneapi/dpl/utility>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
struct NoexceptMoveAssignClass
{
};

struct NonNoexceptMoveAssignClass
{
    NonNoexceptMoveAssignClass&
    operator=(NonNoexceptMoveAssignClass&&) noexcept(false);
};

struct NoexceptMoveConsClass
{
};

struct NonNoexceptMoveConsClass
{
    NonNoexceptMoveConsClass&
    operator=(NonNoexceptMoveConsClass&&) noexcept(false);
};

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
                typedef dpl::tuple<int> tt1;
                typedef dpl::tuple<int, float> tt2;
                typedef dpl::tuple<short, float, int> tt3;
                typedef dpl::tuple<short, NoexceptMoveAssignClass, float> tt4;
                typedef dpl::tuple<short, NoexceptMoveConsClass, float> tt5;
                typedef dpl::tuple<NoexceptMoveConsClass> tt6;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass> tt7;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass, float> tt8;
                typedef dpl::tuple<float, NoexceptMoveConsNoexceptMoveAssignClass, short> tt9;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass,
                                   char>
                    tt10;
                typedef dpl::tuple<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass,
                                   NoexceptMoveConsNoexceptMoveAssignClass>
                    tt11;

                static_assert(noexcept(dpl::declval<tt1&>().swap(dpl::declval<tt1&>())));
                static_assert(noexcept(dpl::declval<tt2&>().swap(dpl::declval<tt2&>())));
                static_assert(noexcept(dpl::declval<tt3&>().swap(dpl::declval<tt3&>())));
                static_assert(noexcept(dpl::declval<tt4&>().swap(dpl::declval<tt4&>())));
                static_assert(noexcept(dpl::declval<tt5&>().swap(dpl::declval<tt5&>())));
                static_assert(noexcept(dpl::declval<tt6&>().swap(dpl::declval<tt6&>())));
                static_assert(noexcept(dpl::declval<tt7&>().swap(dpl::declval<tt7&>())));
                static_assert(noexcept(dpl::declval<tt8&>().swap(dpl::declval<tt8&>())));
                static_assert(noexcept(dpl::declval<tt9&>().swap(dpl::declval<tt9&>())));
                static_assert(noexcept(dpl::declval<tt10&>().swap(dpl::declval<tt10&>())));
                static_assert(noexcept(dpl::declval<tt11&>().swap(dpl::declval<tt11&>())));
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
