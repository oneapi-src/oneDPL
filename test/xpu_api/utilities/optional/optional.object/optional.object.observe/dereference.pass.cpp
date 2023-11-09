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

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::optional;

struct X
{
    constexpr int
    test() const&
    {
        return 3;
    }
    constexpr int
    test() &
    {
        return 4;
    }
    constexpr int
    test() const&&
    {
        return 5;
    }
    constexpr int
    test() &&
    {
        return 6;
    }
};

bool
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                optional<X> opt(X{});
                const optional<X>& const_opt = opt;
                
                ret_access[0] &= ((*opt).test() == 4);
                ret_access[0] &= ((*const_opt).test() == 3);
                ret_access[0] &= ((*std::move(opt)).test() == 6);
                ret_access[0] &= ((*std::move(const_opt)).test() == 5);

                constexpr optional<X> opt1(X{});
                static_assert((*opt1).test() == 3);
                static_assert((*std::move(opt1)).test() == 5);
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional dereference in kernel_test");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
