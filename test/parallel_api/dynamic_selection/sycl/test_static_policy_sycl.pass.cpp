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
#include "support/test_ds_utils.h"

int main() {
  using policy_t = oneapi::dpl::experimental::static_policy;
  std::vector<sycl::queue> u;
  build_universe(u);
  std::cout << "UNIVERSE SIZE " << u.size() << std::endl;
  if (u.empty()) {
    std::cout << "PASS\n";
    return 0;
  }
  sycl::queue test_resource = u[0];
  auto f = [test_resource](int i) { return test_resource; };

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


