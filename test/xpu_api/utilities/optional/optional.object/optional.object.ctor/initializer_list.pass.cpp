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

// template <class U, class... Args>
//     constexpr
//     explicit optional(in_place_t, initializer_list<U> il, Args&&... args);

#include "support/test_config.h"

#include <oneapi/dpl/optional>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>
#include <oneapi/dpl/array>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using dpl::in_place;
using dpl::in_place_t;
using dpl::optional;

class X
{
    int i_;
    int j_ = 0;

  public:
    X() : i_(0) {}
    X(int i) : i_(i) {}
    X(int i, int j) : i_(i), j_(j) {}

    ~X() {}

    friend bool
    operator==(const X& x, const X& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
    }
};

class Y
{
    int i_;
    int j_ = 0;

  public:
    constexpr Y() : i_(0) {}
    constexpr Y(int i) : i_(i) {}
    constexpr Y(std::initializer_list<int> il) : i_(il.begin()[0]), j_(il.begin()[1]) {}

    friend constexpr bool
    operator==(const Y& x, const Y& y)
    {
        return x.i_ == y.i_ && x.j_ == y.j_;
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
            auto ret_access = buffer1.get_access<sycl::accesdpl::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    static_assert(!dpl::is_constructible<X, std::initializer_list<int>&>::value);
                    static_assert(!dpl::is_constructible<optional<X>, std::initializer_list<int>&>::value);
                }
                {
                    static_assert(dpl::is_constructible<optional<Y>, std::initializer_list<int>&>::value);
                    constexpr optional<Y> opt(in_place, {3, 1});
                    static_assert(static_cast<bool>(opt) == true);
                    static_assert(*opt == Y{3, 1});

                    struct test_constexpr_ctor : public optional<Y>
                    {
                        constexpr test_constexpr_ctor(in_place_t, std::initializer_list<int> i)
                            : optional<Y>(in_place, i)
                        {
                        }
                    };
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
    EXPECT_TRUE(ret, "Wrong result of constexpr dpl::optional and initialization list check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
