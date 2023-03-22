/*
    Copyright 2021 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#include "oneapi/dpl/dynamic_selection"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

#include <iostream>

class fake_selection_handle_t {
  sycl::queue q_;
public:
  using property_handle_t = oneapi::dpl::experimental::nop_property_handle_t;
  using native_context_t = sycl::queue;

  fake_selection_handle_t(native_context_t q = sycl::queue(sycl::cpu_selector{})) : q_(q) {}
  native_context_t get_native() { return q_; }
  property_handle_t get_property_handle() { return oneapi::dpl::experimental::nop_property_handle; }
};

int test_cout() {
  oneapi::dpl::experimental::sycl_scheduler s;
  oneapi::dpl::experimental::sycl_scheduler::execution_resource_t e;
  std::cout << s << e;
  return 0;
}

int test_submit_and_wait_on_scheduler() {
  const int N = 100;
  oneapi::dpl::experimental::sycl_scheduler s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    s.submit(h, [&](sycl::queue q, int i) {
             ecount += i;
             return sycl::event{};
           }, i
    );
  }
  s.wait_for_all();
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
    return 1;
  }
  std::cout << "wait_on_scheduler: OK\n";
  return 0;
}

int test_submit_and_wait_on_sync() {
  const int N = 100;
  oneapi::dpl::experimental::sycl_scheduler s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    auto w = s.submit(h,
           [&](sycl::queue q, int i) {
             ecount += i;
             return sycl::event{};
           }, i
    );
    w.wait_for_all();
    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  std::cout << "wait_on_sync: OK\n";
  return 0;
}

int test_properties() {
  oneapi::dpl::experimental::sycl_scheduler s;
  oneapi::dpl::experimental::sycl_scheduler::universe_container_t v;
  //= { sycl::queue(sycl::cpu_selector{}), sycl::queue(sycl::gpu_selector{}) };
  try {
    sycl::cpu_selector ds_cpu;
    sycl::queue cpu_queue(ds_cpu);
    v.push_back(cpu_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to use cpu selector\n";
  }
  try {
    sycl::gpu_selector ds_gpu;
    sycl::queue gpu_queue(ds_gpu);
    v.push_back(gpu_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to use gpu selector\n";
  }
  oneapi::dpl::experimental::property::report(s, oneapi::dpl::experimental::property::universe, v);
  auto v2 = oneapi::dpl::experimental::property::query(s, oneapi::dpl::experimental::property::universe);
  auto v2s = v2.size();
  if (v != v2) {
    std::cout << "ERROR: reported universe and queried universe are not equal\n";
    return 1;
  }
  auto us = oneapi::dpl::experimental::property::query(s, oneapi::dpl::experimental::property::universe_size);
  if (v2s != us) {
    std::cout << "ERROR: queried universe size inconsistent with queried universe\n";
    return 1;
  }
  std::cout << "properties: OK\n";
  return 0;
}

int main() {
  if (test_cout()
      || test_submit_and_wait_on_scheduler()
      || test_submit_and_wait_on_sync()
      || test_properties()
   ) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


