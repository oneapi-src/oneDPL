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


#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"
#include "support/inline_scheduler.h"

#include <iostream>
#include <sstream>

class fake_selection_handle_t {
  int  q_;
public:
  using property_handle_t = oneapi::dpl::experimental::nop_property_handle_t;
  using native_context_t = int;

  fake_selection_handle_t(native_context_t q = 0) : q_(q) {}
  native_context_t get_native() { return q_; }
  property_handle_t get_property_handle() { return oneapi::dpl::experimental::nop_property_handle; }
};

int test_cout() {
  int_inline_scheduler_t s;
  int_inline_scheduler_t::execution_resource_t e;
  std::cout << e;
  return 0;
}

int test_submit_and_wait_on_scheduler() {
  const int N = 100;
  int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    s.submit(h, [&](int q, int i) {
             ecount += i;
             return 0;
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
  int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    auto w = s.submit(h,
           [&](int q, int i) {
             ecount += i;
             return 0;
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
  int_inline_scheduler_t s;
  int_inline_scheduler_t::universe_container_t v = { 1,2};
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
  if (!oneapi::dpl::experimental::property::query(s, oneapi::dpl::experimental::property::is_device_available,1 )){
    std::cout << "ERROR: Inline device not available\n";
    return 1;
  }
  std::cout << "properties: OK\n";
  return 0;
}

int main() {
  if (test_cout()
      || test_submit_and_wait_on_scheduler()
      || test_submit_and_wait_on_sync()
      || test_properties()) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


