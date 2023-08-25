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
#include "support/test_dynamic_load_utils.h"
#include "support/test_dynamic_selection_utils.h"
#include "support/sycl_sanity.h"

static inline void build_dl_universe(std::vector<sycl::queue> &u) {
  try {
    auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
    sycl::queue cpu1_queue(device_cpu1);
    run_sycl_sanity_test(cpu1_queue);
    u.push_back(cpu1_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with cpu_selector\n";
  }
  try {
    auto device_cpu2 = sycl::device(sycl::cpu_selector_v);
    sycl::queue cpu2_queue(device_cpu2);
    run_sycl_sanity_test(cpu2_queue);
    u.push_back(cpu2_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with cpu_selector\n";
  }
}

int main() {
  using policy_t = oneapi::dpl::experimental::dynamic_load_policy<oneapi::dpl::experimental::sycl_backend>;
  std::vector<sycl::queue> u;
  build_dl_universe(u);

  auto n = u.size();

  // should be similar to round_robin when waiting on policy
  auto f = [u, n](int i, int offset) {
    if(i==offset) {
        return u[offset];
    }
    return u[(i+offset)%u.size()];
  };

  auto f2 = [u, n](int i, int offset=0) {
    return u[offset];
  };
  // should always pick first when waiting on sync in each iteration
  //auto fs = [test_resource, u](int i) { return u[0]; };

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;
  if ( test_dl_initialization(u)
       || test_select<policy_t, decltype(u), decltype(f2)&, false>(u, f2)
       || test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f2)
       || test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f2, 1)
       || test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f2)
       || test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f2, 1)
       || test_submit_and_wait<just_call_submit, policy_t>(u, f2)
       || test_submit_and_wait<just_call_submit, policy_t>(u, f2, 1)
       || test_submit_and_wait<call_select_before_submit, policy_t>(u, f2)
       || test_submit_and_wait<call_select_before_submit, policy_t>(u, f2, 1)
       || test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f)
       || test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 1)
       || test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f)
       || test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 1)

     ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}
