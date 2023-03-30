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

#include <cstdio>
#include <memory>
#include <ostream>
#include <vector>
#include <CL/sycl.hpp>
#include "support/sycl_sanity.h"

int run_test(sycl::queue q) {
  const int num_items = 1000000;
  using DoubleVector = std::vector<double>;
  DoubleVector a(num_items),b(num_items), c(num_items);
  for (int i = 0; i < num_items; ++i) {
    a[i] = i;
    b[i] = num_items - i;
    c[i] = 0;
  }

  auto device = q.get_info<sycl::info::queue::device>();
  if (device.is_cpu()) {
     std::printf("running on cpu\n");
  } else if (device.is_gpu()) {
    std::printf("running on gpu\n");
  } else {
    std::printf("running on other\n");
  }

  {
    sycl::buffer<double> a_buf(a), b_buf(b), c_buf(c);
    q.submit([&](sycl::handler &h) {
      sycl::accessor a_(a_buf, h, sycl::read_only);
      sycl::accessor b_(b_buf, h, sycl::read_only);
      sycl::accessor c_(c_buf, h, sycl::write_only);
      h.parallel_for<TestUtils::unique_kernel_name<class sum1, TestUtils::uniq_kernel_index<sycl::usm::alloc::shared>()>>(num_items, [=](auto j) {
        c_[j] += a_[j] + b_[j];
      });
    });

    q.submit([&](sycl::handler &h) {
      sycl::accessor a_(a_buf, h, sycl::read_only);
      sycl::accessor b_(b_buf, h, sycl::read_only);
      sycl::accessor c_(c_buf, h, sycl::write_only);
      h.parallel_for<TestUtils::unique_kernel_name<class sum2, TestUtils::uniq_kernel_index<sycl::usm::alloc::shared>()>>(num_items, [=](auto j) {
        c_[j] += a_[j] + b_[j];
      });
    }).wait();
  }
  printf("%f, %f, %f, %f\n", c[0], c[1000], c[10000], c[100000]);
  return 0;
}

int test_runner() {
  int r = 0;
  try {
    sycl::default_selector ds;
    sycl::queue default_queue(ds);
    r += run_test(default_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with default_selector\n";
  }

  try {
    sycl::gpu_selector gs;
    sycl::queue gpu_queue(gs);
    r += run_test(gpu_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with gpu_selector\n";
  }

  try {
    sycl::cpu_selector cs;
    sycl::queue cpu_queue(cs);
    r += run_test(cpu_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with cpu_selector\n";
  }
  
  return r;
}

int main() {
  return test_runner();
}

