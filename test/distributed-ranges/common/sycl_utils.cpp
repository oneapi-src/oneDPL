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

#include "xhp-tests.hpp"

#ifdef SYCL_LANGUAGE_VERSION

using T = float;

TEST(SYCLUtils, ParalelFor1D) {
  const std::size_t size = 10;
  sycl::queue q;
  sycl::range range(size - 1);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [a](auto i) { a[i] = i; };
  auto setb = [b](auto i) { b[i] = i; };
  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  // disabled due to: https://github.com/oneapi-src/distributed-ranges/issues/790
  // EXPECT_EQ(std::span(a, size), std::span(b, size));
}

// some mdspan based tests were removed from here and should be added in the future, see:
// https://github.com/oneapi-src/distributed-ranges/issues/789

#endif // SYCL_LANGUAGE_VERSION
