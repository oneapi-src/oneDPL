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

#pragma once

#include <cstdio>
#include <memory>
#include <ostream>
#include <vector>
#include <CL/sycl.hpp>

static inline int run_sycl_sanity_test(sycl::queue q) {
  const int num_items = 100;
  using DoubleVector = std::vector<double>;
  DoubleVector a(num_items),b(num_items), c(num_items);
  for (int i = 0; i < num_items; ++i) {
    a[i] = i;
    b[i] = num_items - i;
    c[i] = 0;
  }

  {
    sycl::buffer<double> a_buf(a), b_buf(b), c_buf(c);
    q.submit([&](sycl::handler &h) {
      sycl::accessor a_(a_buf, h, sycl::read_only);
      sycl::accessor b_(b_buf, h, sycl::read_only);
      sycl::accessor c_(c_buf, h, sycl::write_only);
      h.parallel_for(num_items, [=](auto j) {
        c_[j] += a_[j] + b_[j];
      });
    });

    q.submit([&](sycl::handler &h) {
      sycl::accessor a_(a_buf, h, sycl::read_only);
      sycl::accessor b_(b_buf, h, sycl::read_only);
      sycl::accessor c_(c_buf, h, sycl::write_only);
      h.parallel_for(num_items, [=](auto j) {
        c_[j] += a_[j] + b_[j];
      });
    }).wait();
  }
  return 0;
}

static inline sycl::queue build_universe(std::vector<sycl::queue> &u) {
  try {
    auto device_default = sycl::device(sycl::default_selector());
    sycl::queue default_queue(device_default);
    run_sycl_sanity_test(default_queue);
    u.push_back(default_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with default_selector_v\n";
  }

  try {
    auto device_gpu = sycl::device(sycl::gpu_selector());
    cl::sycl::queue gpu_queue(device_gpu);
    run_sycl_sanity_test(gpu_queue);
    u.push_back(gpu_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with gpu_selector_v\n";
  }

  try {
    auto device_cpu = sycl::device(sycl::cpu_selector());
    cl::sycl::queue cpu_queue(device_cpu);
    run_sycl_sanity_test(cpu_queue);
    u.push_back(cpu_queue);
  } catch (sycl::exception) {
    std::cout << "SKIPPED: Unable to run with cpu_selector_v\n";
  }
  return u[0];
}

