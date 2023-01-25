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

#include <any>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "oneapi/dpl/dymanic_selection/ds_properties.h"

using str_to_any_map = std::map<std::string, std::any>;

template<typename Handle, typename Property, typename Resource>
int test_query_with_resources(const Handle &h, const std::string& property_name, const Property &p, Resource rs1, Resource rs2, str_to_any_map &erm) {
  auto r1 = ds::property::query(h, p, rs1); 
  auto er1 = std::any_cast<decltype(r1)>(erm[property_name + "_1"]); 
  if (r1 != er1) {
    std::cout << "ERROR: " << property_name + "_1" << " does not match expected result\n";
    return 1;
  }
  auto r2 = ds::property::query(h, p, rs2); 
  auto er2 = std::any_cast<decltype(r2)>(erm[property_name + "_2"]); 
  if (r2 != er2) {
    std::cout << "ERROR: " << property_name + "_2" << " does not match expected result\n";
    return 1;
  }
  return 0;
}

template<typename Handle, typename Property>
int test_simple_query(const Handle &h, const std::string& property_name, const Property &p, str_to_any_map &erm) {
  auto r = ds::property::query(h, p); 
  auto er = std::any_cast<decltype(r)>(erm[property_name]); 
  if (r != er) {
    std::cout << "ERROR: " << property_name << "does not match expected result\n";
    return 1;
  }
  return 0;
}

template<typename Handle, typename Resource>
int test_queries(const Handle &h, Resource rs1, Resource rs2, str_to_any_map &erm) {
  return    test_simple_query(h, "universe_size", ds::property::universe_size, erm)
         || test_simple_query(h, "universe", ds::property::universe, erm)
         || test_query_with_resources(h, "dynamic_load", ds::property::dynamic_load, rs1, rs2, erm)
         || test_query_with_resources(h, "is_device_available", ds::property::is_device_available, rs1, rs2, erm)
         || test_query_with_resources(h, "task_execution_time", ds::property::task_execution_time, rs1, rs2, erm);
}

struct fake_handle_t {
  using resource_t = std::string;
  uint64_t e1 = 123, e2 = 456;
  auto query(ds::property::universe_size_t) const noexcept {
    return int(2);
  }
  auto query(ds::property::universe_t) const noexcept {
    return std::vector<std::string>{"cpu", "gpu"};
  }
  auto query(ds::property::dynamic_load_t, resource_t r) const noexcept {
    if (r == "cpu") return int(4);
    else if (r == "gpu") return int(5);
    else return -1;
  }
  auto query(ds::property::is_device_available_t, resource_t r) const noexcept {
    if (r == "cpu") return true;
    else return false;
  }
  auto query(ds::property::task_execution_time_t, resource_t r) const noexcept {
    if (r == "cpu") return e1;
    else if (r == "gpu") return e2;
    else return uint64_t(0);
  }
  auto report(ds::property::task_execution_time_t, uint64_t v) noexcept {
    e1 = e2 = v; 
    return v;
  }
};

int test_queries_fake() {
  fake_handle_t fh;
  str_to_any_map erm;
  erm["universe_size"] = int(2);
  erm["universe"] = std::vector<std::string>{"cpu", "gpu"};
  erm["dynamic_load_1"] = int(4);
  erm["dynamic_load_2"] = int(5);
  erm["is_device_available_1"] = true;
  erm["is_device_available_2"] = false;
  erm["task_execution_time_1"] = uint64_t(123);
  erm["task_execution_time_2"] = uint64_t(456);
  return test_queries(fh, "cpu", "gpu", erm);
}

int test_report_fake() {
  fake_handle_t fh;
  str_to_any_map erm;
  if (ds::property::query(fh, ds::property::task_execution_time, "cpu") != 123) { 
    std::cout << "ERROR: initial query of cpu task_execution_time has unexpected result\n";
    return 1;
  }
  if (ds::property::query(fh, ds::property::task_execution_time, "gpu") != 456) {
    std::cout << "ERROR: initial query of gpu task_execution_time has unexpected result\n";
    return 1;
  }
  if (ds::property::report(fh, ds::property::task_execution_time, 789) != 789) {
    std::cout << "ERROR: result of report of cpu task_execution_time not 789\n";
    return 1;
  }
  if (ds::property::query(fh, ds::property::task_execution_time, "cpu") != 789) { 
    std::cout << "ERROR: final query of cpu task_execution_time has unexpected result\n";
    return 1;
  }
  if (ds::property::query(fh, ds::property::task_execution_time, "gpu") != 789) {
    std::cout << "ERROR: final query of gpu task_execution_time has unexpected result\n";
    return 1;
  }
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

