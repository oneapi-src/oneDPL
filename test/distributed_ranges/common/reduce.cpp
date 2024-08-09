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
template <typename T> class Reduce : public testing::Test {
protected:
};

TYPED_TEST_SUITE(Reduce, AllTypes);

TYPED_TEST(Reduce, Range) {
  Ops1<TypeParam> ops(10);

  auto max = [](double x, double y) { return std::max(x, y); };
  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3, max),
            xp::reduce(ops.dist_vec, 3, max));
}

TYPED_TEST(Reduce, Max) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3, std::plus{}),
            xp::reduce(ops.dist_vec, 3, std::plus{}));
}

TYPED_TEST(Reduce, Iterators) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(
      std::reduce(ops.vec.begin(), ops.vec.end(), 3, std::plus{}),
      xp::reduce(ops.dist_vec.begin(), ops.dist_vec.end(), 3, std::plus{}));
}

TYPED_TEST(Reduce, RangeDefaultOp) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3),
            xp::reduce(ops.dist_vec, 3));
}

TYPED_TEST(Reduce, IteratorsDefaultOp) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3),
            xp::reduce(ops.dist_vec.begin(), ops.dist_vec.end(), 3));
}

TYPED_TEST(Reduce, RangeDefaultInit) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end()),
            xp::reduce(ops.dist_vec));
}

TYPED_TEST(Reduce, IteratorsDefaultInit) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end()),
            xp::reduce(ops.dist_vec.begin(), ops.dist_vec.end()));
}
