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

// Fixture
template <typename T> class TransformView : public testing::Test {
public:
};

TYPED_TEST_SUITE(TransformView, AllTypes);

TYPED_TEST(TransformView, Basic) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto local = stdrng::views::transform(ops.vec, negate);
  auto dist = xp::views::transform(ops.dist_vec, negate);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(TransformView, All) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  EXPECT_TRUE(
      check_view(stdrng::views::transform(stdrng::views::all(ops.vec), negate),
                 xp::views::transform(stdrng::views::all(ops.dist_vec), negate)));
}

TYPED_TEST(TransformView, Move) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto l_value = stdrng::views::all(ops.vec);
  auto dist_l_value = stdrng::views::all(ops.dist_vec);
  EXPECT_TRUE(
      check_view(stdrng::views::transform(std::move(l_value), negate),
                 xp::views::transform(std::move(dist_l_value), negate)));
}

TYPED_TEST(TransformView, L_Value) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto l_value = stdrng::views::all(ops.vec);
  auto dist_l_value = stdrng::views::all(ops.dist_vec);
  EXPECT_TRUE(check_view(stdrng::views::transform(l_value, negate),
                         xp::views::transform(dist_l_value, negate)));
}

TYPED_TEST(TransformView, ForEach) {
  Ops1<TypeParam> ops(10);

  auto null_transform = [](auto &&v) { return v; };
  auto local = stdrng::views::transform(ops.vec, null_transform);
  auto dist = xp::views::transform(ops.dist_vec, null_transform);

  auto null_for_each = [](auto v) {};
  stdrng::for_each(local, null_for_each);
  xp::for_each(dist, null_for_each);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TYPED_TEST(TransformView, Reduce) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { return -v; };
  auto local = stdrng::views::transform(ops.vec, negate);
  auto dist = xp::views::transform(ops.dist_vec, negate);
  EXPECT_EQ(std::reduce(local.begin(), local.end(), 3, std::plus{}),
            xp::reduce(dist, 3, std::plus{}));
}