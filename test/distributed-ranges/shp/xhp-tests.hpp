// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"

#include <gtest/gtest.h>
#include <oneapi/dpl/distributed-ranges>

// #ifdef __cpp_lib_format
// #include <format>
// namespace drfmt {
//   using std::format;
// }
// #else
#include <fmt/core.h>
#include <fmt/ranges.h>
namespace drfmt {
  using fmt::format;
}
// #endif

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

using AllocatorTypes =
    ::testing::Types<xhp::device_allocator<int>>;

template <typename V>
concept compliant_view = rng::forward_range<V> && requires(V &v) {
  dr::ranges::segments(v);
  dr::ranges::rank(dr::ranges::segments(v)[0]);
};

#include "../include/common-tests.hpp"

using AllTypes = ::testing::Types<xhp::distributed_vector<int>>;
