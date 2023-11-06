// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_SYCL_SANITY_H
#define _ONEDPL_SYCL_SANITY_H

#include <cstdio>
#include <memory>
#include <ostream>
#include <vector>
#include <sycl/sycl.hpp>

//TODO: Source from oneDPL
namespace TestUtils
{
  template <sycl::usm::alloc alloc_type>
  constexpr std::size_t
  uniq_kernel_index()
  {
    return static_cast<typename std::underlying_type_t<sycl::usm::alloc>>(alloc_type);
  }

  template <typename Op, std::size_t CallNumber>
  struct unique_kernel_name;

  template <typename Policy, int idx>
  using new_kernel_name = unique_kernel_name<typename std::decay_t<Policy>, idx>;
}


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
      h.parallel_for<TestUtils::unique_kernel_name<class sum3, TestUtils::uniq_kernel_index<sycl::usm::alloc::shared>()>>(num_items, [=](auto j) {
        c_[j] += a_[j] + b_[j];
      });
    });

    q.submit([&](sycl::handler &h) {
      sycl::accessor a_(a_buf, h, sycl::read_only);
      sycl::accessor b_(b_buf, h, sycl::read_only);
      sycl::accessor c_(c_buf, h, sycl::write_only);
      h.parallel_for<TestUtils::unique_kernel_name<class sum4, TestUtils::uniq_kernel_index<sycl::usm::alloc::shared>()>>(num_items, [=](auto j) {
        c_[j] += a_[j] + b_[j];
      });
    }).wait();
  }
  return 0;
}

static inline void build_universe(std::vector<sycl::queue> &u) {
  try {
    auto device_default = sycl::device(sycl::default_selector_v);
    sycl::queue default_queue(device_default);
    run_sycl_sanity_test(default_queue);
    u.push_back(default_queue);
  } catch (const sycl::exception&) {
    std::cout << "SKIPPED: Unable to run with default_selector\n";
  }

  try {
    auto device_gpu = sycl::device(sycl::gpu_selector_v);
    sycl::queue gpu_queue(device_gpu);
    run_sycl_sanity_test(gpu_queue);
    u.push_back(gpu_queue);
  } catch (const sycl::exception&) {
    std::cout << "SKIPPED: Unable to run with gpu_selector\n";
  }

  try {
    auto device_cpu = sycl::device(sycl::cpu_selector_v);
    sycl::queue cpu_queue(device_cpu);
    run_sycl_sanity_test(cpu_queue);
    u.push_back(cpu_queue);
  } catch (const sycl::exception&) {
    std::cout << "SKIPPED: Unable to run with cpu_selector\n";
  }
}

#endif /* _ONEDPL_SYCL_SANITY_H */
