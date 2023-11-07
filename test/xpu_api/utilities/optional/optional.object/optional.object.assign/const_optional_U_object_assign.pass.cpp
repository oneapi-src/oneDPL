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

struct Y1
{
    Y1() = default;
    Y1(const int&) {}
    Y1&
    operator=(const Y1&) = delete;
};

struct Y2
{
    Y2() = default;
    Y2(const int&) = delete;
    Y2&
    operator=(const int&)
    {
        return *this;
    }
};

template <class T>
struct AssignableFrom
{
    int type_constructed = 0;
    int type_assigned = 0;
    int int_constructed = 0;
    int int_assigned = 0;

    AssignableFrom() = default;

    explicit AssignableFrom(T) { ++type_constructed; }
    AssignableFrom& operator=(T)
    {
        ++type_assigned;
        return *this;
    }

    AssignableFrom(int) { ++int_constructed; }
    AssignableFrom&
    operator=(int)
    {
        ++int_assigned;
        return *this;
    }

  private:
    AssignableFrom(AssignableFrom const&) = delete;
    AssignableFrom&
    operator=(AssignableFrom const&) = delete;
};

bool
test_ambigious_assign()
{
    sycl::queue q;
    bool ret = true;
    sycl::range<1> numOfItems1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](sycl::handler& cgh) {
            sycl::stream out(1024, 256, cgh);
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                using OptInt = dpl::optional<int>;
                {
                    using T = AssignableFrom<OptInt const&>;
                    const OptInt a(42);
                    {
                        dpl::optional<T> t;
                        t = a;
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        dpl::optional<T> t(42);
                        t = a;
                        ret_access[0] &= (t->type_constructed == 0);
                        ret_access[0] &= (t->type_assigned == 1);
                        ret_access[0] &= (t->int_constructed == 1);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        dpl::optional<T> t(42);
                        t = dpl::move(a);
                        ret_access[0] &= (t->type_constructed == 0);
                        ret_access[0] &= (t->type_assigned == 1);
                        ret_access[0] &= (t->int_constructed == 1);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                }
                {
                    using T = AssignableFrom<OptInt&>;
                    OptInt a(42);
                    {
                        dpl::optional<T> t;
                        t = a;
                        ret_access[0] &= (t->type_constructed == 1);
                        ret_access[0] &= (t->type_assigned == 0);
                        ret_access[0] &= (t->int_constructed == 0);
                        ret_access[0] &= (t->int_assigned == 0);
                    }
                    {
                        using Opt = dpl::optional<T>;
                        static_assert(!dpl::is_assignable_v<Opt&, OptInt const&>);
                    }
                }
            });
        });
    }
    return ret;
}

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
            cgh.single_task<class KernelTest2>([=]() {
                {
                    optional<int> opt;
                    constexpr optional<short> opt2;
                    opt = opt2;
                    static_assert(static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt;
                    constexpr optional<short> opt2(short{2});
                    opt = opt2;
                    static_assert(static_cast<bool>(opt2) == true);
                    static_assert(*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
                }
                {
                    optional<int> opt(3);
                    constexpr optional<short> opt2;
                    opt = opt2;
                    static_assert(static_cast<bool>(opt2) == false);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                }
                {
                    optional<int> opt(3);
                    constexpr optional<short> opt2(short{2});
                    opt = opt2;
                    static_assert(static_cast<bool>(opt2) == true);
                    static_assert(*opt2 == 2);
                    ret_access[0] &= (static_cast<bool>(opt) == static_cast<bool>(opt2));
                    ret_access[0] &= (*opt == *opt2);
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
    auto ret = test_ambigious_assign();
    ret &= kernel_test();
    EXPECT_TRUE(ret, "Wrong result of constexpr dpl::optional copy check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
