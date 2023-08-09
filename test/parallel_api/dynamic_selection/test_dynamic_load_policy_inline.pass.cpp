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
#include "support/test_dynamic_load_policy_utils.h"
#include "support/inline_scheduler.h"

int main() {
  using policy_t = oneapi::dpl::experimental::dynamic_load_policy_t<TestUtils::int_inline_scheduler_t>;
  std::vector<int> u{4, 5, 6, 7};
  int test_resource = 5;

  // should be similar to round_robin when waiting on policy
  auto fp = [test_resource, u](int i) { return u[(i-1)%4]; };

  // should always pick first when waiting on sync in each iteration
  auto fs = [test_resource, u](int i) { return u[0]; };

  if (test_cout<policy_t>()
//      || test_properties<policy_t>(u, test_resource)
//      || test_invoke<policy_t>(u, fs)
      || test_invoke_async_and_wait_on_policy<policy_t>(u, fp)
  /*    || test_invoke_async_and_wait_on_sync<policy_t>(u, fs)
      || test_invoke_async_and_get_wait_list<policy_t>(u, fp)
      || test_invoke_async_and_get_wait_list_single_element<policy_t>(u, fs)
      || test_invoke_async_and_get_wait_list_empty<policy_t>(u, fs)
      || test_select<policy_t>(u, fs)
      || test_select_and_wait_on_policy<policy_t>(u, fp)
      || test_select_and_wait_on_sync<policy_t>(u, fs)
      || test_select_invoke<policy_t>(u, fs)*/
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}
