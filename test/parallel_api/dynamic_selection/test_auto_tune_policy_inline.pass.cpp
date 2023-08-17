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
#include "support/inline_scheduler.h"

int main() {
  using policy_t = oneapi::dpl::experimental::auto_tune_policy<TestUtils::int_inline_scheduler_t>;
  std::vector<int> u{6, 5, 4, 7};
  int test_resource = 4;
 
  // should first go through each resource and then
  // when only waiting at end, should always pick first since execution time is not yet reported
  auto fp = [u](int i) { 
              if (i <= 4)
                return u[i-1]; 
              else
                return u[0]; 
            };

  // should first go through each resource and then
  // then should select the best "4"
  auto fs = [test_resource, u](int i) { 
              if (i <= 4)
                return u[i-1]; 
              else
                return test_resource;
            };

  if (test_properties<policy_t>(u, test_resource)
      || test_select<policy_t, decltype(u), const decltype(fp)&, true>(u, fp) 
//      || test_invoke<policy_t>(u, fs)
//      || test_invoke_async_and_wait_on_policy<policy_t>(u, fp)
//      || test_invoke_async_and_wait_on_sync<policy_t>(u, fs)
//      || test_auto_tune_select_and_wait_on_policy<policy_t>(u, fp)
//      || test_auto_tune_select_and_wait_on_sync<policy_t>(u, fs)
//      || test_auto_tune_select_invoke<policy_t>(u, fs)
     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


