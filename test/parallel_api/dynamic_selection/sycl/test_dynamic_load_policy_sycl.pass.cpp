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

int main() {
  using policy_t = oneapi::dpl::experimental::dynamic_load_policy;
  std::vector<sycl::queue> u;
  build_universe(u);
  if (u.empty()) {
    std::cout << "PASS\n";
    return 0;
  }
  sycl::queue test_resource = u[0];

  auto n = u.size();
  std::cout << "UNIVERSE SIZE " << n << std::endl;

  // should be similar to round_robin when waiting on policy
  auto fp = [test_resource, u, n](int i) { return u[(i-1)%n]; };

  // should always pick first when waiting on sync in each iteration
  auto fs = [test_resource, u](int i) { return u[0]; };

  if (test_cout<policy_t>()
      || test_properties<policy_t>(u, test_resource)
      || test_invoke<policy_t>(u, fs)
      || test_invoke_async_and_wait_on_policy<policy_t>(u, fp)
      || test_invoke_async_and_wait_on_sync<policy_t>(u, fs)
      || test_invoke_async_and_get_wait_list<policy_t>(u, fp)
      || test_invoke_async_and_get_wait_list_single_element<policy_t>(u, fs)
      || test_invoke_async_and_get_wait_list_empty<policy_t>(u, fs)
      || test_select<policy_t>(u, fs)
      || test_select_and_wait_on_policy<policy_t>(u, fp)
      || test_select_and_wait_on_sync<policy_t>(u, fs)
      || test_select_invoke<policy_t>(u, fs)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}
