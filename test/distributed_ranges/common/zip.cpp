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

#include "xp_tests.hpp"

template <typename... Rs> auto test_zip(Rs &&...rs) {
  return xp::views::zip(std::forward<Rs>(rs)...);
}

// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE(Zip, AllTypes);

TYPED_TEST(Zip, Dist1) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::zip(ops.vec);
  auto dist = test_zip(ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist2) {
  Ops2<TypeParam> ops(10);

  auto local = stdrng::views::zip(ops.vec0, ops.vec1);
  auto dist = test_zip(ops.dist_vec0, ops.dist_vec1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist3) {
  Ops3<TypeParam> ops(10);

  auto local = stdrng::views::zip(ops.vec0, ops.vec1, ops.vec2);
  auto dist = test_zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist3Distance) {
  Ops3<TypeParam> ops(10);

  EXPECT_EQ(
      stdrng::distance(stdrng::views::zip(ops.vec0, ops.vec1, ops.vec2)),
      stdrng::distance(test_zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

TYPED_TEST(Zip, RangeSegments) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::zip(ops.vec);
  auto dist = test_zip(ops.dist_vec);
  auto flat = stdrng::views::join(dr::ranges::segments(dist));
  EXPECT_TRUE(is_equal(local, flat));
}

#ifndef TEST_SP
// Will not compile with SP
TYPED_TEST(Zip, IterSegments) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::zip(ops.vec);
  auto dist = test_zip(ops.dist_vec);
  auto flat = stdrng::views::join(dr::ranges::segments(dist.begin()));
  EXPECT_TRUE(is_equal(local, flat));
}
#endif

TYPED_TEST(Zip, Drop) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::drop(stdrng::views::zip(ops.vec), 2);
  auto dist = xp::views::drop(test_zip(ops.dist_vec), 2);

  auto flat = stdrng::views::join(dr::ranges::segments(dist));
  EXPECT_EQ(local, dist);
  EXPECT_TRUE(is_equal(local, flat));
}

TYPED_TEST(Zip, ConsumingAll) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::zip(stdrng::views::all(ops.vec));
  auto dist = test_zip(xp::views::all(ops.dist_vec));
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(Zip, FeedingAll) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::all(stdrng::views::zip(ops.vec));
  auto dist = xp::views::all(test_zip(ops.dist_vec));
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(Zip, ForEach) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xp::for_each(test_zip(ops.dist_vec0, ops.dist_vec1), copy);
  stdrng::for_each(stdrng::views::zip(ops.vec0, ops.vec1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Zip, ForEachDrop) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xp::for_each(xp::views::drop(test_zip(ops.dist_vec0, ops.dist_vec1), 1),
                copy);
  stdrng::for_each(xp::views::drop(stdrng::views::zip(ops.vec0, ops.vec1), 1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Zip, ConsumingSubrange) {
  Ops2<TypeParam> ops(10);

  auto local =
      stdrng::views::zip(stdrng::subrange(ops.vec0.begin() + 1, ops.vec0.end() - 1),
                      stdrng::subrange(ops.vec1.begin() + 1, ops.vec1.end() - 1));
  auto dist = test_zip(
      stdrng::subrange(ops.dist_vec0.begin() + 1, ops.dist_vec0.end() - 1),
      stdrng::subrange(ops.dist_vec1.begin() + 1, ops.dist_vec1.end() - 1));
  EXPECT_EQ(local, dist);
}

TEST(Zip, FeedingTransform) {
  Ops2<xp::distributed_vector<int>> ops(10);

  auto mul = [](auto v) { return std::get<0>(v) * std::get<1>(v); };
  auto local = stdrng::views::transform(stdrng::views::zip(ops.vec0, ops.vec1), mul);
  auto dist_zip = test_zip(ops.dist_vec0, ops.dist_vec1);
  auto dist = xp::views::transform(dist_zip, mul);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TEST(Zip, CopyConstructor) {
  Ops2<xp::distributed_vector<int>> ops(10);

  auto dist = test_zip(ops.dist_vec0, ops.dist_vec1);
  auto dist_copy(dist);
  EXPECT_EQ(dist, dist_copy);
}

TYPED_TEST(Zip, TransformReduce) {
  Ops2<TypeParam> ops(10);

  auto mul = [](auto v) { return std::get<0>(v) * std::get<1>(v); };

  auto local = stdrng::views::transform(stdrng::views::zip(ops.vec0, ops.vec1), mul);
  auto local_reduce = std::reduce(local.begin(), local.end());

  auto dist =
      xp::views::transform(test_zip(ops.dist_vec0, ops.dist_vec1), mul);
  auto dist_reduce = xp::reduce(dist);

  EXPECT_EQ(local_reduce, dist_reduce);
}

TYPED_TEST(Zip, IotaStaticAssert) {
  Ops1<TypeParam> ops(10);

  auto dist = test_zip(xp::views::iota(100), ops.dist_vec);
  static_assert(std::forward_iterator<decltype(dist.begin())>);
  static_assert(std::forward_iterator<decltype(dist.end())>);
  static_assert(std::forward_iterator<decltype(stdrng::begin(dist))>);
  static_assert(std::forward_iterator<decltype(stdrng::end(dist))>);
  using Dist = decltype(dist);
  static_assert(stdrng::forward_range<Dist>);
  static_assert(dr::distributed_range<Dist>);
}

TYPED_TEST(Zip, Iota) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::zip(stdrng::views::iota(100), ops.vec);
  auto dist = test_zip(xp::views::iota(100), ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(Zip, Iota2nd) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(stdrng::views::zip(ops.vec, stdrng::views::iota(100)),
                         test_zip(ops.dist_vec, xp::views::iota(100))));
}

TYPED_TEST(Zip, ForEachIota) {
  Ops1<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xp::for_each(test_zip(xp::views::iota(100), ops.dist_vec), copy);
  stdrng::for_each(stdrng::views::zip(stdrng::views::iota(100), ops.vec), copy);

  EXPECT_EQ(ops.vec, ops.dist_vec);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}
