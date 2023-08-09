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
#include "support/sycl_sanity.h"

static inline void build_dl_universe(std::vector<sycl::queue> &u) {
  try {
    auto device_cpu1 = sycl::device(sycl::cpu_selector());
    sycl::queue cpu1_queue(device_cpu1);
    run_sycl_sanity_test(cpu1_queue);
    u.push_back(cpu1_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with cpu_selector\n";
  }
  try {
    auto device_cpu2 = sycl::device(sycl::cpu_selector());
    sycl::queue cpu2_queue(device_cpu2);
    run_sycl_sanity_test(cpu2_queue);
    u.push_back(cpu2_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with cpu_selector\n";
  }
}

int main() {
  using policy_t = oneapi::dpl::experimental::dynamic_load_policy;
  std::vector<sycl::queue> u;
  build_dl_universe(u);

  sycl::queue test_resource = u[0];
  auto n = u.size();

  // should be similar to round_robin when waiting on policy
  auto f = [test_resource, u, n](int i) {
    if(i==0) return u[0];
    return u[1];
  };

  // should always pick first when waiting on sync in each iteration
  //auto fs = [test_resource, u](int i) { return u[0]; };

  if (test_cout<policy_t>()
//      || test_properties<policy_t>(u, test_resource)
//      || test_invoke<policy_t>(u, fs)
      || test_invoke_async_and_wait_on_policy<policy_t>(u, f)
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
