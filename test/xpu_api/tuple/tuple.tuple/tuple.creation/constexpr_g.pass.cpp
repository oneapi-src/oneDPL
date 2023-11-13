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

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
// make_tuple
void
test_make_tuple()
{
    {
        typedef dpl::tuple<int, float> tuple_type;
        [[maybe_unuased]] constexpr tuple_type p1 = dpl::make_tuple(22, 22.222f);
    }

    {
        constexpr auto p1 = dpl::make_tuple(22, 22.222f, 77799);
        static_assert(dpl::get<0>(p1) == 22);
    }
}

// forward_as_tuple
void
test_forward_as_tuple()
{
    {
        static const int i(22);
        static const float f(22.222f);
        typedef dpl::tuple<const int&, const float&&> tuple_type;
        [[maybe_unuased]] constexpr tuple_type p1 = dpl::forward_as_tuple(i, dpl::move(f));
    }

    {
        static const int i(22);
        static const float f(22.222f);
        static const int ii(77799);

        typedef dpl::tuple<const int&, const float&, const int&&> tuple_type;
        [[maybe_unuased]] constexpr tuple_type p1 = dpl::forward_as_tuple(i, f, dpl::move(ii));
    }
}

// tie
void
test_tie()
{
    {
        static const int i(22);
        static const float f(22.222f);
        typedef dpl::tuple<const int&, const float&> tuple_type;
        [[maybe_unuased]] constexpr tuple_type p1 = dpl::tie(i, f);
    }

    {
        static const int i(22);
        static const float f(22.222f);
        static const int ii(77799);

        typedef dpl::tuple<const int&, const float&, const int&> tuple_type;
        [[maybe_unuased]] constexpr tuple_type p1 = dpl::tie(i, f, ii);
    }
}

// get
void
test_get()
{
    {
        typedef dpl::tuple<int, float> tuple_type;
        constexpr tuple_type t1{55, 77.77f};
        [[maybe_unuased]] constexpr auto var = dpl::get<1>(t1);
    }

    {
        typedef dpl::tuple<int, float, int> tuple_type;
        constexpr tuple_type t1{55, 77.77f, 99};
        [[maybe_unuased]] constexpr auto var = dpl::get<2>(t1);
    }
}

// tuple_cat
void
test_tuple_cat()
{
    typedef dpl::tuple<int, float> tuple_type1;
    typedef dpl::tuple<int, int, float> tuple_type2;

    constexpr tuple_type1 t1{55, 77.77f};
    constexpr tuple_type2 t2{55, 99, 77.77f};
    [[maybe_unuased]] constexpr auto cat1 = dpl::tuple_cat(t1, t2);
}

namespace
{
template <class T>
constexpr int zero_from_anything(T)
{
    return 0;
}
} // namespace

// ignore
void
test_ignore()
{
    [[maybe_unuased]] constexpr auto ign1 = dpl::ignore;
    [[maybe_unuased]] constexpr auto ign2 = dpl::make_tuple(dpl::ignore);
    [[maybe_unuased]] constexpr int ign3 = zero_from_anything(dpl::ignore);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test_make_tuple();
                test_forward_as_tuple();
                test_tie();
                test_get();
                test_tuple_cat();
                test_ignore();
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
