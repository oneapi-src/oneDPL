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
struct Poison
{
    Poison(Poison&&) = delete;
};

struct ThrowingCopy
{
    ThrowingCopy(const ThrowingCopy&);
    ThrowingCopy&
    operator=(const ThrowingCopy&);
};

bool
kernel_test()
{
    static_assert(!dpl::is_copy_constructible<Poison>::value);
    static_assert(!dpl::is_move_constructible<Poison>::value);
    static_assert(!dpl::is_copy_assignable<Poison>::value);
    static_assert(!dpl::is_move_assignable<Poison>::value);
    static_assert(!dpl::is_copy_constructible<dpl::pair<int, Poison>>::value);
    static_assert(!dpl::is_move_constructible<dpl::pair<int, Poison>>::value);
    static_assert(!dpl::is_copy_assignable<dpl::pair<int, Poison>>::value);
    static_assert(!dpl::is_move_assignable<dpl::pair<int, Poison>>::value);
    static_assert(!dpl::is_constructible<dpl::pair<int, Poison>&, dpl::pair<char, Poison>&>::value);
    static_assert(!dpl::is_assignable<dpl::pair<int, Poison>&, dpl::pair<char, Poison>&>::value);
    static_assert(!dpl::is_constructible<dpl::pair<int, Poison>&, dpl::pair<char, Poison>>::value);
    static_assert(!dpl::is_assignable<dpl::pair<int, Poison>&, dpl::pair<char, Poison>>::value);
    static_assert(!dpl::is_copy_constructible<dpl::pair<ThrowingCopy, std::unique_ptr<int>>>::value);
    static_assert(dpl::is_move_constructible<dpl::pair<ThrowingCopy, std::unique_ptr<int>>>::value);
    static_assert(!std::is_nothrow_move_constructible<dpl::pair<ThrowingCopy, std::unique_ptr<int>>>::value);
    static_assert(!dpl::is_copy_assignable<dpl::pair<ThrowingCopy, std::unique_ptr<int>>>::value);
    static_assert(dpl::is_move_assignable<dpl::pair<ThrowingCopy, std::unique_ptr<int>>>::value);
    static_assert(!std::is_nothrow_move_assignable<dpl::pair<ThrowingCopy, std::unique_ptr<int>>>::value);
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

    EXPECT_TRUE(ret, "Wrong result of dpl::pair traits check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
