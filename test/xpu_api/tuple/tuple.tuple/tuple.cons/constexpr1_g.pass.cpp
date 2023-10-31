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

#include "testsuite_common_types.h"

#if TEST_DPCPP_BACKEND_PRESENT
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef dpl::tuple<int, int> tuple_type;

                // 01: default ctor
                constexpr_default_constructible test1;
                test1.operator()<tuple_type>();

                // 02: default copy ctor
                constexpr_single_value_constructible test2;
                test2.operator()<tuple_type, tuple_type>();

                // 03: element move ctor, single element
                const int i1(415);
                constexpr tuple_type t2{44, dpl::move(i1)};

                // 04: element move ctor, two element
                const int i2(510);
                const int i3(408);
                constexpr tuple_type t4{dpl::move(i2), dpl::move(i3)};

                // 05: value-type conversion constructor
                const int i4(650);
                const int i5(310);
                constexpr tuple_type t8(i4, i5);

                // 06: pair conversion ctor
                test2.operator()<tuple_type, dpl::pair<int, int>>();
                test2.operator()<dpl::tuple<short, short>, dpl::pair<int, int>>();
                test2.operator()<tuple_type, dpl::pair<short, short>>();

                // 07: different-tuple-type conversion constructor
                test2.operator()<tuple_type, dpl::tuple<short, short>>();
                test2.operator()<dpl::tuple<short, short>, tuple_type>();
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
