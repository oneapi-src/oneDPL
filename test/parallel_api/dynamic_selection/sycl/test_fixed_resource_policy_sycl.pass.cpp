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
#include "support/sycl_sanity.h"
#include "support/test_dynamic_selection_utils.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"

int main() {
  using policy_t = oneapi::dpl::experimental::fixed_resource_policy<oneapi::dpl::experimental::sycl_backend>;
  std::vector<sycl::queue> u;
  build_universe(u);
  std::cout << "UNIVERSE SIZE " << u.size() << std::endl;
  if (u.empty()) {
    std::cout << "PASS\n";
    return 0;
  }
  auto f = [u](int i, int offset) { return u[offset]; };

  constexpr bool just_call_submit = false;
  constexpr bool call_select_before_submit = true;
  if ( test_initialization<policy_t, sycl::queue>(u)
       || test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f)
       || test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 1)
       || test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 2)
       || test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f)
       || test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 1)
       || test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 2)
       || test_submit_and_wait<just_call_submit, policy_t>(u, f)
       || test_submit_and_wait<just_call_submit, policy_t>(u, f, 1)
       || test_submit_and_wait<just_call_submit, policy_t>(u, f, 2)
       || test_submit_and_wait<call_select_before_submit, policy_t>(u, f)
       || test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 1)
       || test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 2)
       || test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f)
       || test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 1)
       || test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 2)
       || test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f)
       || test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 1)
       || test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 2)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


