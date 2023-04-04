// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "oneapi/dpl/dynamic_selection"
#include "support/inline_scheduler.h"

#include <atomic>
#include <iostream>
#include <sstream>

class fake_selection_handle_t {
  int  q_;
public:
  using property_handle_t = oneapi::dpl::experimental::basic_property_handle_t;
  using native_context_t = int;

  fake_selection_handle_t(native_context_t q = 0) : q_(q) {}
  native_context_t get_native() { return q_; }
  property_handle_t get_property_handle() { return oneapi::dpl::experimental::basic_property_handle; }
};

int test_cout() {
  TestUtils::int_inline_scheduler_t s;
  TestUtils::int_inline_scheduler_t::execution_resource_t e;
//  std::cout << e;
  return 0;
}

int test_submit_and_wait_on_scheduler() {
  const int N = 100;
  TestUtils::int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    s.submit(h, [&](int q, int i) {
             ecount += i;
             return 0;
           }, i
    );
  }
  s.wait();
  int count = ecount.load();
  if (count != N*(N+1)/2) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
    return 1;
  }
  std::cout << "wait_on_scheduler: OK\n";
  return 0;
}

int test_submit_and_wait_on_scheduler_single_element() {
  const int N = 1;
  TestUtils::int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    s.submit(h, [&](int q, int i) {
             ecount += i;
             return 0;
           }, i
    );
  }
  s.wait();
  int count = ecount.load();
  if (count != 1) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
    return 1;
  }
  std::cout << "wait_on_scheduler single element: OK\n";
  return 0;
}

int test_submit_and_wait_on_scheduler_empty() {
  const int N = 0;
  TestUtils::int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    s.submit(h, [&](int q, int i) {
             ecount += i;
             return 0;
           }, i
    );
  }
  s.wait();
  int count = ecount.load();
  if (count != 0) {
    std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
    return 1;
  }
  std::cout << "wait_on_scheduler empty list: OK\n";
  return 0;
}

int test_submit_and_wait_on_sync() {
  const int N = 100;
  TestUtils::int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    auto w = s.submit(h,
           [&](int q, int i) {
             ecount += i;
             return 0;
           }, i
    );
    w.wait();
    int count = ecount.load();
    if (count != i*(i+1)/2) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  std::cout << "wait_on_sync: OK\n";
  return 0;
}

int test_submit_and_wait_on_sync_single_element() {
  const int N = 1;
  TestUtils::int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    auto w = s.submit(h,
           [&](int q, int i) {
             ecount += i;
             return 0;
           }, i
    );
    w.wait();
    int count = ecount.load();
    if (count != 1) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  std::cout << "wait_on_sync single element: OK\n";
  return 0;
}

int test_submit_and_wait_on_sync_empty() {
  const int N = 0;
  TestUtils::int_inline_scheduler_t s;
  fake_selection_handle_t h;

  std::atomic<int> ecount = 0;

  for (int i = 1; i <= N; ++i) {
    auto w = s.submit(h,
           [&](int q, int i) {
             ecount += i;
             return 0;
           }, i
    );
    w.wait();
    int count = ecount.load();
    if (count != 0) {
      std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
      return 1;
    }
  }
  std::cout << "wait_on_sync empty list: OK\n";
  return 0;
}

int test_properties() {
  TestUtils::int_inline_scheduler_t s;
  TestUtils::int_inline_scheduler_t::universe_container_t v = { 1,2};
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
      || test_submit_and_wait_on_scheduler_single_element()
      || test_submit_and_wait_on_scheduler_empty()
      || test_submit_and_wait_on_sync()
      || test_submit_and_wait_on_sync_single_element()
      || test_submit_and_wait_on_sync_empty()
      || test_properties()) {
    std::cout << "FAIL\n";
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}


