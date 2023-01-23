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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"
#    include "support/sycl_alloc_utils.h"
#    include "support/scan_serial_impl.h"

#include <vector>
#endif

int main()
{

#if TEST_DPCPP_BACKEND_PRESENT
  std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};

  // Setup host inputs
  std::vector<int> incl_input_host = v;
  std::vector<int> excl_input_host = v;  
  
  // Setup device inputs
  sycl::queue syclQue = TestUtils::get_test_queue();
  int* incl_input_dev = sycl::malloc_device<int>(10, syclQue);
  int* excl_input_dev = sycl::malloc_device<int>(10, syclQue);  

  syclQue.memcpy(incl_input_dev, v.data(), v.size()*sizeof(int)).wait();
  syclQue.memcpy(excl_input_dev, v.data(), v.size()*sizeof(int)).wait();  

  // Inclusive scan (in-place works)
  ::std::inclusive_scan(incl_input_host.begin(),
                      incl_input_host.end(),
                      incl_input_host.begin());
  oneapi::dpl::inclusive_scan(oneapi::dpl::execution::make_device_policy<class TempKernelName1>(syclQue),
                               incl_input_dev,
                               incl_input_dev + v.size(),
                               incl_input_dev );
  std::vector<int> incl_result_host(v.size());
  syclQue.memcpy(incl_result_host.data(), incl_input_dev, v.size()*sizeof(int)).wait();

  EXPECT_EQ_RANGES(incl_input_host, incl_result_host, "wrong inclusive_scan results");

  sycl::free(incl_input_dev, syclQue);

  // Exclusive scan (in-place, incorrect results)
  ::std::exclusive_scan(excl_input_host.begin(),
                       excl_input_host.end(),
                       excl_input_host.begin(),
                       0 );
  oneapi::dpl::exclusive_scan( oneapi::dpl::execution::make_device_policy<class TempKernelName2>(syclQue),
                               excl_input_dev,
                               excl_input_dev + v.size(),
                               excl_input_dev,
                               0 );
  std::vector<int> excl_result_host(v.size());
  syclQue.memcpy(excl_result_host.data(), excl_input_dev, v.size()*sizeof(int)).wait();

  EXPECT_EQ_RANGES(excl_input_host, excl_result_host, "wrong exclusive_scan results");

  sycl::free(excl_input_dev, syclQue);
#endif
  return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
