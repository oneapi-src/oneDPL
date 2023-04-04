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


#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "support/test_ds_utils.h"
#include "support/sycl_sanity.h"

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
      || test_invoke<policy_t>(u, f)
      || test_invoke_async_and_wait_on_policy<policy_t>(u, f)
      || test_invoke_async_and_wait_on_sync<policy_t>(u, f)
      || test_select<policy_t>(u, f)
      || test_select_and_wait_on_policy<policy_t>(u, f)
      || test_select_and_wait_on_sync<policy_t>(u, f)
      || test_select_invoke<policy_t>(u, f)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


