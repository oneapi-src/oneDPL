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

#if TEST_DPCPP_BACKEND_PRESENT
#    include "oneapi/dpl/async"
#    include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#endif // TEST_DPCPP_BACKEND_PRESENT

#include <memory>

#include "support/utils.h"


struct UserEvent
{
    void wait_and_throw()
    {
    }
};

auto
create_future(UserEvent e, std::shared_ptr<int> ptr)
{
    return oneapi::dpl::__par_backend_hetero::__future(e, ptr);
}

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    long counter_state = 0;

    // Case 1: we save future in result
    {
        auto ptr = std::make_shared<int>();
        {
            auto res = create_future(UserEvent{}, ptr);
            counter_state = ptr.use_count();
            EXPECT_TRUE(counter_state == 2, "wrong counter state #1");
        }

        EXPECT_TRUE(ptr.use_count() == 1, "wrong counter state #2");
    }

    // Case 2: we don't save the result
    {
        auto ptr = std::make_shared<int>();
        {
            /*auto res = */create_future(UserEvent{}, ptr);
            EXPECT_TRUE(ptr.use_count() != counter_state, "wrong counter state #3");
        }

        EXPECT_TRUE(ptr.use_count() == 1, "wrong counter state #4");
    }
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
