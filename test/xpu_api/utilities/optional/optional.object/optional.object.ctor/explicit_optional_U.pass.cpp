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
#include <oneapi/dpl/utility>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::optional;

template <class KernelTest, class T, class U>
bool
test(optional<U>&& rhs, bool is_going_to_throw = false)
{
    sycl::queue q;
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<optional<U>, 1> buffer2(&rhs, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto rhs_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest>([=]() {
                static_assert(!(dpl::is_convertible<optional<U>&&, optional<T>>::value));
                bool rhs_engaged = static_cast<bool>(rhs_access[0]);
                optional<T> lhs(dpl::move(rhs_access[0]));
                ret_access[0] &= (static_cast<bool>(lhs) == rhs_engaged);
            });
        });
    }
    return ret;
}

class X
{
    int i_;

  public:
    explicit X(int i) : i_(i) {}
    X(X&& x) : i_(dpl::exchange(x.i_, 0)) {}
    ~X() { i_ = 0; }
    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_;
    }
};

class KernelTest1;
class KernelTest2;

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    bool ret = true;
    {
        optional<int> rhs;
        ret &= test<KernelTest1, X>(dpl::move(rhs));
    }
    {
        optional<int> rhs(3);
        ret &= test<KernelTest2, X>(dpl::move(rhs));
    }
    EXPECT_TRUE(ret, "Wrong result of dpl::optional moving constructor check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
