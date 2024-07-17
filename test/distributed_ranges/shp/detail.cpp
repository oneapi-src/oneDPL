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

#include "xhp_tests.hpp"
#include <oneapi/dpl/internal/distributed_ranges_impl/shp.hpp>

namespace shp = oneapi::dpl::experimental::dr::shp;

TEST(DetailTest, parallel_for) {
  std::size_t size = 2 * 1024 * 1024;
  std::size_t n = 4 * std::size_t(std::numeric_limits<int32_t>::max());

  // Compute `v`
  std::vector<int> v(size, 0);

  auto iota = stdrng::views::iota(std::size_t(0), n);

  std::for_each(iota.begin(), iota.end(), [&](auto i) { v[i % size] += 1; });

  auto &&q = dr::shp::__detail::queue(0);

  dr::shp::shared_allocator<int> alloc(q);

  dr::shp::vector<int, dr::shp::shared_allocator<int>> dvec(size, 0, alloc);

  auto dv = dvec.data();

  dr::__detail::parallel_for(q, n, [=](auto i) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        v(dv[i % size]);
    v += 1;
  }).wait();

  std::vector<int> dvec_local(size);
  dr::shp::copy(dvec.begin(), dvec.end(), dvec_local.begin());

  EXPECT_EQ(v, dvec_local);
}
