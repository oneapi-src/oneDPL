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

#include <gtest/gtest.h>
#include <oneapi/dpl/distributed_ranges>

#ifdef USE_FMT
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
namespace drfmt {
using fmt::format;
}
#else
namespace drfmt {
template <typename... Args> inline auto format(Args &&...) {
  return "check failed";
}
} // namespace drfmt
#endif

#define TEST_SHP
// To share tests with MHP
const std::size_t comm_rank = 0;
const std::size_t comm_size = 1;

// Namespace aliases and wrapper functions to make the tests uniform
namespace dr = oneapi::dpl::experimental::dr;
namespace xhp = dr::shp;

inline void barrier() {}
inline void fence() {}
inline void fence_on(auto &&) {}

using AllocatorTypes = ::testing::Types<xhp::device_allocator<int>>;

template <typename V>
concept compliant_view = rng::forward_range<V> && requires(V &v) {
  dr::ranges::segments(v);
  dr::ranges::rank(dr::ranges::segments(v)[0]);
};

#include "../include/common_tests.hpp"

using AllTypes = ::testing::Types<xhp::distributed_vector<int>>;
