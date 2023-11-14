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
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
struct A
{
    A(const A&) = delete;
    A& operator=(const A&) = delete;

    A() = default;
    A(A&&) = default;
};

struct AMoveNonNoexcept
{
    AMoveNonNoexcept(const AMoveNonNoexcept&) = delete;
    AMoveNonNoexcept&
    operator=(const AMoveNonNoexcept&) = delete;

    AMoveNonNoexcept() = default;
    AMoveNonNoexcept(AMoveNonNoexcept&&) noexcept(false) = default;
};

struct legacy
{
    legacy() = default;
    legacy(const legacy&) = delete;
};

struct legacyMoveNonNoexcept
{
    legacyMoveNonNoexcept() = default;
    legacyMoveNonNoexcept(const legacyMoveNonNoexcept&) noexcept(false) = default;
};

void
kernel_test()
{
    int i = 0;
    const int ci = 0;


    static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(i)), int&&>);
    static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(ci)), const int&&>);

    {
        A a;
        const A ca;
        legacy l;

        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(a)), A&&>);
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(ca)), const A&&>);
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(l)), legacy&&>);
    }

    {
        AMoveNonNoexcept a;
        const AMoveNonNoexcept ca;
        legacyMoveNonNoexcept l;

        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(a)), AMoveNonNoexcept&&>);
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(ca)), const AMoveNonNoexcept&&>);
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(l)), const legacyMoveNonNoexcept&>);
    }

    constexpr int i1 = 23;
    constexpr int i2 = dpl::move_if_noexcept(i1);
    static_assert(i2 == 23);
}

class KernelTest;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() { kernel_test(); });
        });
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
