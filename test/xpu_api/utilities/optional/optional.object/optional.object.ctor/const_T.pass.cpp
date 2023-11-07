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
    int i_;
    X(int i) : i_(i) {}
    X(const X& x) : i_(x.i_) {}
    ~X() { i_ = 0; }
    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

bool
kernel_test()
{
    sycl::queue q;
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    typedef int T;
                    constexpr T t(5);
                    constexpr optional<T> opt(t);
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 5);
                }
                {
                    typedef float T;
                    constexpr T t(3);
                    constexpr optional<T> opt(t);
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == 3);
                }
                {
                    const int x = 42;
                    optional<const int> o(x);
                    ret_access[0] &= (*o == x);
                }
                {
                    typedef X T;
                    const T t(3);
                    optional<T> opt = t;
                    ret_access[0] &= (static_cast<bool>(opt) == true);
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
    EXPECT_TRUE(ret, "Wrong result of constexpr dpl::optional and operator '==' check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
