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


#include <any>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "oneapi/dpl/dynamic_selection"

using str_to_any_map = std::map<std::string, std::any>;

template<typename Handle, typename Property, typename Resource>
int test_query_with_resources(const Handle &h, const std::string& property_name, const Property &p, Resource rs1, Resource rs2, str_to_any_map &erm) {
  auto r1 = oneapi::dpl::experimental::property::query(h, p, rs1);
  auto er1 = std::any_cast<decltype(r1)>(erm[property_name + "_1"]);
  if (r1 != er1) {
    std::cout << "ERROR: " << property_name + "_1" << " does not match expected result\n";
    return 1;
  }
  auto r2 = oneapi::dpl::experimental::property::query(h, p, rs2);
  auto er2 = std::any_cast<decltype(r2)>(erm[property_name + "_2"]);
  if (r2 != er2) {
    std::cout << "ERROR: " << property_name + "_2" << " does not match expected result\n";
    return 1;
  }
  return 0;
}

template<typename Handle, typename Property>
int test_simple_query(const Handle &h, const std::string& property_name, const Property &p, str_to_any_map &erm) {
  auto r = oneapi::dpl::experimental::property::query(h, p);
  auto er = std::any_cast<decltype(r)>(erm[property_name]);
  if (r != er) {
    std::cout << "ERROR: " << property_name << "does not match expected result\n";
    return 1;
  }
  return 0;
}

template<typename Handle, typename Resource>
int test_queries(const Handle &h, Resource rs1, Resource rs2, str_to_any_map &erm) {
  return    test_simple_query(h, "universe_size", oneapi::dpl::experimental::property::universe_size, erm)
         || test_simple_query(h, "universe", oneapi::dpl::experimental::property::universe, erm);
}

struct fake_handle_t {
  using resource_t = std::string;
  uint64_t e1 = 123, e2 = 456;
  auto query(oneapi::dpl::experimental::property::universe_size_t) const noexcept {
    return int(2);
  }
  auto query(oneapi::dpl::experimental::property::universe_t) const noexcept {
    return std::vector<std::string>{"cpu", "gpu"};
  }
};

int test_queries_fake() {
  fake_handle_t fh;
  str_to_any_map erm;
  erm["universe_size"] = int(2);
  erm["universe"] = std::vector<std::string>{"cpu", "gpu"};
  return test_queries(fh, "cpu", "gpu", erm);
}

int test_report_fake() {
  fake_handle_t fh;
  str_to_any_map erm;
  return 0;
}

int main() {
  if (test_queries_fake() || test_report_fake()) {
    return 1;
  } else {
    std::cout << "PASS\n";
    return 0;
  }
}

