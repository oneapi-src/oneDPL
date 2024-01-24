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

int
main()
{
#if TEST_DYNAMIC_SELECTION_AVAILABLE
    using policy_t = oneapi::dpl::experimental::round_robin_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u;
    build_universe(u);
    if (u.empty())
    {
        std::cout << "PASS\n";
        return 0;
    }

    auto n = u.size();
    std::cout << "UNIVERSE SIZE " << n << std::endl;

    auto f = [u, n](int i) { return u[(i - 1) % n]; };

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;
    if (test_initialization<policy_t, sycl::queue>(u) ||
        test_select<policy_t, decltype(u), decltype(f)&, false>(u, f) ||
        test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f) ||
        test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f) ||
        test_submit_and_wait<just_call_submit, policy_t>(u, f) ||
        test_submit_and_wait<call_select_before_submit, policy_t>(u, f) ||
        test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f) ||
        test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f))
    {
        std::cout << "FAIL\n";
        return 1;
    }
    else
    {
        std::cout << "PASS\n";
        return 0;
    }
#else
    std::cout << "SKIPPED\n";
    return 0;
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
}
