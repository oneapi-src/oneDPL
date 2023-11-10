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

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <typename Tuple>
constexpr bool
test_constexpr_default_ctor()
{
    constexpr Tuple tpl;
    constexpr std::tuple_element_t<0, Tuple> desired_value{};

    static_assert(dpl::get<0>(tpl) == desired_value);
    return true;
}

template <typename TTestTuple, typename TValueTuple>
constexpr void
test_constexpr_single_val_ctor()
{
    constexpr TValueTuple rhs;
    constexpr TTestTuple lhs{rhs};

    static_assert(dpl::get<0>(lhs) == dpl::get<0>(rhs));
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::tuple<int, int, int> tuple_type;

                // 01: default ctor
                test_constexpr_default_ctor<tuple_type>();

                // 02: default copy ctor
                test_constexpr_single_val_ctor<tuple_type, tuple_type>();

                // 03: element move ctor, single element
                const int i1(415);
                constexpr tuple_type t2{44, 55, dpl::move(i1)};

                // 04: element move ctor, three element
                const int i2(510);
                const int i3(408);
                const int i4(650);
                constexpr tuple_type t4{dpl::move(i2), dpl::move(i3), dpl::move(i4)};

                // 05: value-type conversion constructor
                const int i5(310);
                const int i6(310);
                const int i7(310);
                constexpr tuple_type t8(i5, i6, i7);

                // 06: different-tuple-type conversion constructor
                test_constexpr_single_val_ctor<tuple_type, dpl::tuple<short, short, short>>();
                test_constexpr_single_val_ctor<dpl::tuple<short, short, short>, tuple_type>();
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
