// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Transform : public testing::Test {
public:
};

TYPED_TEST_SUITE(Transform, AllTypes);

TYPED_TEST(Transform, Range) {
  Ops2<TypeParam> ops(10);

  auto negate = [](auto &&v) { return -v; };

  xhp::transform(ops.dist_vec0, ops.dist_vec1.begin(), negate);
  rng::transform(ops.vec0, ops.vec1.begin(), negate);
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
  rng::transform(ops.vec0, ops.vec1.begin(), negate);
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Transform, Iterators) {
  Ops2<TypeParam> ops(10);

  auto negate = [](auto &&v) { return -v; };

  xhp::transform(ops.dist_vec0.begin(), ops.dist_vec0.end(),
                 ops.dist_vec1.begin(), negate);
  rng::transform(ops.vec0.begin(), ops.vec0.end(), ops.vec1.begin(), negate);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}
