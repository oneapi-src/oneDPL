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

// <optional>

// template <class U>
//   optional(const optional<U>& rhs);

#include "support/test_config.h"

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::optional;

template <class KernelTest, class T, class U>
bool
kernel_test(const optional<U>& rhs)
{
    sycl::queue q;
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<U>, 1> buffer2(&rhs, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::accesdpl::mode::write>(cgh);
            auto rhs_access = buffer2.template get_access<sycl::accesdpl::mode::write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                bool rhs_engaged = static_cast<bool>(rhs_access[0]);
                optional<T> lhs = rhs_access[0];
                ret_access[0] &= (static_cast<bool>(lhs) == rhs_engaged);
                if (rhs_engaged)
                    ret_access[0] &= (*lhs == *rhs_access[0]);
            });
        });
    }
    return ret;
}

class X
{
    int i_;

  public:
    X(int i) : i_(i) {}
    X(const X& x) : i_(x.i_) {}
    ~X() { i_ = 0; }
    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

class Y
{
    int i_;

  public:
    Y(int i) : i_(i) {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_;
    }
};

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;

bool
test()
{
    bool ret = true;
    {
        typedef short U;
        typedef int T;
        optional<U> rhs;
        ret &= kernel_test<KernelTest1, T>(rhs);
    }
    {
        typedef short U;
        typedef int T;
        optional<U> rhs(U{3});
        ret &= kernel_test<KernelTest2, T>(rhs);
    }
    {
        typedef X T;
        typedef int U;
        optional<U> rhs;
        ret &= kernel_test<KernelTest3, T>(rhs);
    }
    {
        typedef X T;
        typedef int U;
        optional<U> rhs(U{3});
        ret &= kernel_test<KernelTest4, T>(rhs);
    }
    {
        typedef Y T;
        typedef int U;
        optional<U> rhs;
        ret &= kernel_test<KernelTest5, T>(rhs);
    }
    {
        typedef Y T;
        typedef int U;
        optional<U> rhs(U{3});
        ret &= kernel_test<KernelTest6, T>(rhs);
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = test();
    EXPECT_TRUE(ret, "Wrong result of dpl::optional and operator '==' check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
