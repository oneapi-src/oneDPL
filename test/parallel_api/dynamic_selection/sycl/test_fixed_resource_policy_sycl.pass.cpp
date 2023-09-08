// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "support/test_dynamic_selection_utils.h"
#include "support/test_config.h"
#include "support/utils.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE
#    include "support/sycl_sanity.h"
#endif

int
main()
{
    bool processed = false;
#if TEST_DYNAMIC_SELECTION_AVAILABLE
    using policy_t = oneapi::dpl::experimental::fixed_resource_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u;
    build_universe(u);
    std::cout << "UNIVERSE SIZE " << u.size() << std::endl;
    if (!u.empty())
    {
        auto f = [u](int i, int offset = 0) { return u[offset]; };

        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        EXPECT_EQ(0, test_initialization<policy_t, sycl::queue>(u), "");
        EXPECT_EQ(0, test_select<policy_t, decltype(u), decltype(f)&, false>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 1), "");
        EXPECT_EQ(0, test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 1), "");
        EXPECT_EQ(0, test_submit_and_wait<just_call_submit, policy_t>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait<just_call_submit, policy_t>(u, f, 1), "");
        EXPECT_EQ(0, test_submit_and_wait<call_select_before_submit, policy_t>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 1), "");
        EXPECT_EQ(0, test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 1), "");
        EXPECT_EQ(0, test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f), "");
        EXPECT_EQ(0, test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 1), "");

        processed = true;
    }
#endif    

    return TestUtils::done(processed);
}
