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

struct PODType
{
    int value;
    int value2;
};

class X
{
  public:
    bool dtor_called = false;
    X() = default;
    ~X() { dtor_called = true; }
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
            auto ret_access = buffer1.get_access<sycl::accesdpl::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    typedef int T;
                    static_assert(dpl::is_trivially_destructible<T>::value);
                    static_assert(dpl::is_trivially_destructible<optional<T>>::value);
                }
                {
                    typedef float T;
                    static_assert(dpl::is_trivially_destructible<T>::value);
                    static_assert(dpl::is_trivially_destructible<optional<T>>::value);
                }
                {
                    typedef PODType T;
                    static_assert(dpl::is_trivially_destructible<T>::value);
                    static_assert(dpl::is_trivially_destructible<optional<T>>::value);
                }
                {
                    typedef X T;
                    static_assert(!dpl::is_trivially_destructible<T>::value);
                    static_assert(!dpl::is_trivially_destructible<optional<T>>::value);
                    {
                        X x;
                        optional<X> opt{x};
                        ret_access[0] &= (opt->dtor_called == false);
                    }
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
    EXPECT_TRUE(ret, "Wrong result of dpl::is_trivially_destructible and dpl::optional check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
