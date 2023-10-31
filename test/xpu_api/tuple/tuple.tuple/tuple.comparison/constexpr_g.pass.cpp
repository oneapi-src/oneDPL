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

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                constexpr_comparison_operators test;
                test.operator()<dpl::tuple<int, int>>();
            });
        });
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
