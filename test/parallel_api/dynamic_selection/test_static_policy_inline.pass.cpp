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
#include "oneapi/dpl/internal/dynamic_selection_impl/inline_scheduler.h"
#include "support/test_ds_utils.h"

int main() {
  using policy_t = oneapi::dpl::experimental::static_policy_t<oneapi::dpl::experimental::int_inline_scheduler_t>;
  std::vector<int> u{4, 5, 6, 7};
  int test_resource = 4;
  auto f = [test_resource](int i) { return test_resource; };

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


