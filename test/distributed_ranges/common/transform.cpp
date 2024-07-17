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

// Fixture
template <typename T> class Transform : public testing::Test {
public:
};

TYPED_TEST_SUITE(Transform, AllTypes);

TYPED_TEST(Transform, Range) {
  Ops2<TypeParam> ops(10);

  auto negate = [](auto &&v) { return -v; };

  xhp::transform(ops.dist_vec0, ops.dist_vec1.begin(), negate);
  stdrng::transform(ops.vec0, ops.vec1.begin(), negate);
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Transform, RangeMutate) {
  Ops2<TypeParam> ops(10);

  auto negate = [](auto &&v) {
    v++;
    return -v;
  };

  xhp::transform(ops.dist_vec0, ops.dist_vec1.begin(), negate);
  stdrng::transform(ops.vec0, ops.vec1.begin(), negate);
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Transform, Iterators) {
  Ops2<TypeParam> ops(10);

  auto negate = [](auto &&v) { return -v; };

  xhp::transform(ops.dist_vec0.begin(), ops.dist_vec0.end(),
                 ops.dist_vec1.begin(), negate);
  stdrng::transform(ops.vec0.begin(), ops.vec0.end(), ops.vec1.begin(), negate);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}
