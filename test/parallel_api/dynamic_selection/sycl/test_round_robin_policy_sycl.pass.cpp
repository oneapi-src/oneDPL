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
#include "support/test_ds_utils.h"
#include "support/sycl_sanity.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"

int main() {
  using policy_t = oneapi::dpl::experimental::round_robin_policy;
  std::vector<sycl::queue> u;
  build_universe(u);
  if (u.empty()) {
    std::cout << "PASS\n";
    return 0;
  }
  sycl::queue test_resource = u[0];

  auto n = u.size();
  std::cout << "UNIVERSE SIZE " << n << std::endl;

  auto f = [test_resource, u, n](int i) { return u[(i-1)%n]; };

  if (test_cout<policy_t>()
      || test_properties<policy_t>(u, test_resource)
      || test_submit_and_wait<policy_t>(u, f)
      || test_submit_and_wait_on_policy<policy_t>(u, f)
      || test_submit_and_wait_on_sync<policy_t>(u, f)
      || test_submit_and_get_submission_group<policy_t>(u, f)
      || test_submit_and_get_submission_group_single_element<policy_t>(u, f)
      || test_submit_and_get_submission_group_empty<policy_t>(u, f)
      || test_select<policy_t>(u, f)
      || test_select_and_wait_on_policy<policy_t>(u, f)
      || test_select_and_wait_on_sync<policy_t>(u, f)
      || test_select_submit_and_wait<policy_t>(u, f)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


