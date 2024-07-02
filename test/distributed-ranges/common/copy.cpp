// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Copy : public testing::Test {
public:
};

TYPED_TEST_SUITE(Copy, AllTypes);

TYPED_TEST(Copy, Range) {
  Ops2<TypeParam> ops(10);

  xhp::copy(ops.dist_vec0, ops.dist_vec1.begin());
  rng::copy(ops.vec0, ops.vec1.begin());
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Copy, Iterator) {
  Ops2<TypeParam> ops(10);

  std::copy(ops.vec0.begin(), ops.vec0.end(), ops.vec1.begin());
  xhp::copy(ops.dist_vec0.begin(), ops.dist_vec0.end(), ops.dist_vec1.begin());
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Copy, IteratorOffset) {
  Ops2<TypeParam> ops(10);

  std::copy(ops.vec0.begin() + 1, ops.vec0.end() - 1, ops.vec1.begin() + 1);
  xhp::copy(ops.dist_vec0.begin() + 1, ops.dist_vec0.end() - 1,
            ops.dist_vec1.begin() + 1);
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Copy, RangeToDist) {
  Ops2<TypeParam> ops(10);

  xhp::copy(ops.vec0, ops.dist_vec0.begin());
  rng::copy(ops.vec1, ops.dist_vec1.begin());
  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Copy, DistToLocal) {
  Ops2<TypeParam> ops(10);

  xhp::copy(ops.dist_vec0, ops.vec0.begin());
  rng::copy(ops.dist_vec1, ops.vec1.begin());
  EXPECT_EQ(ops.dist_vec0, ops.vec0);
  EXPECT_EQ(ops.dist_vec1, ops.vec1);
}
