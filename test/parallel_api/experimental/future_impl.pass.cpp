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

// Required to comment next line for reproducing error case
#define CALL_WAIT_AND_THROW_IN_DESTRUCTOR 1

struct UserEvent
{
    inline static long wait_counter = 0;

    ~UserEvent()
    {
#if CALL_WAIT_AND_THROW_IN_DESTRUCTOR
        wait_and_throw();
#endif // CALL_WAIT_AND_THROW_IN_DESTRUCTOR
    }

    void wait_and_throw()
    {
        ++wait_counter;
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

    EXPECT_TRUE(UserEvent::wait_counter == 0, "wrong wait counter state #1");

    // Case 1: we save future in result
    {
        auto ptr = std::make_shared<int>();
        long wait_counter_state = 0;

        {
            UserEvent ev;
            auto res = create_future(ev, ptr);
            wait_counter_state = UserEvent::wait_counter;

            counter_state = ptr.use_count();
            EXPECT_TRUE(counter_state == 2, "wrong counter state #1");
        }

        EXPECT_TRUE(counter_state == 2, "wrong counter state #1");

        EXPECT_TRUE(UserEvent::wait_counter != wait_counter_state, "wrong counter state #2");
    }

    // Case 2: we don't save the result
    {
        auto ptr = std::make_shared<int>();
        long wait_counter_state = 0;
        {
            UserEvent ev;
            /*auto res = */create_future(UserEvent{}, ptr);
            wait_counter_state = UserEvent::wait_counter;

            EXPECT_TRUE(ptr.use_count() != counter_state, "wrong counter state #3");

            // This means that managed resources already was destroyed because the instance of __future class wasn't saved.
            // In the case of sycl__event this means that we haven't sycl::event::wait() call
            // and destroyed some managed resources for example sycl::buffer,
            // but our async algorithm can continue working after that.
            EXPECT_TRUE(ptr.use_count() == 1, "wrong counter state #3");
        }

        EXPECT_TRUE(ptr.use_count() == 1, "wrong counter state #4");

        EXPECT_TRUE(UserEvent::wait_counter != wait_counter_state, "wrong counter state #3");
    }
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
