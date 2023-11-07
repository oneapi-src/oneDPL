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
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::optional;

bool
kernel_test()
{
    sycl::queue q;
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::accesdpl::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    typedef int T;
                    constexpr optional<T> opt(T(5));
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 5);

                    struct test_constexpr_ctor : public optional<T>
                    {
                        constexpr test_constexpr_ctor(T&&) {}
                    };
                }
                {
                    typedef double T;
                    constexpr optional<T> opt(T(3));
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 3);

                    struct test_constexpr_ctor : public optional<T>
                    {
                        constexpr test_constexpr_ctor(T&&) {}
                    };
                }
                {
                    const int x = 42;
                    optional<const int> o(dpl::move(x));
                    ret_access[0] &= (*o == 42);
                }
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
    EXPECT_TRUE(ret, "Wrong result of dpl::optional passed as r-value check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
