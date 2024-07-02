// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Reduce : public testing::Test {
protected:
};

TYPED_TEST_SUITE(Reduce, AllTypes);

TYPED_TEST(Reduce, Range) {
  Ops1<TypeParam> ops(10);

  auto max = [](double x, double y) { return std::max(x, y); };
  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3, max),
            xhp::reduce(ops.dist_vec, 3, max));
}

TYPED_TEST(Reduce, Max) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3, std::plus{}),
            xhp::reduce(ops.dist_vec, 3, std::plus{}));
}

TYPED_TEST(Reduce, Iterators) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(
      std::reduce(ops.vec.begin(), ops.vec.end(), 3, std::plus{}),
      xhp::reduce(ops.dist_vec.begin(), ops.dist_vec.end(), 3, std::plus{}));
}

TYPED_TEST(Reduce, RangeDefaultOp) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3),
            xhp::reduce(ops.dist_vec, 3));
}

TYPED_TEST(Reduce, IteratorsDefaultOp) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 3),
            xhp::reduce(ops.dist_vec.begin(), ops.dist_vec.end(), 3));
}

TYPED_TEST(Reduce, RangeDefaultInit) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end()),
            xhp::reduce(ops.dist_vec));
}

TYPED_TEST(Reduce, IteratorsDefaultInit) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end()),
            xhp::reduce(ops.dist_vec.begin(), ops.dist_vec.end()));
}
