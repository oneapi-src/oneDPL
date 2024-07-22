// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp_tests.hpp"

// Fixture
template <typename T> class Equals : public testing::Test {
public:
};

TYPED_TEST_SUITE(Equals, AllTypes);

TYPED_TEST(Equals, Same) {
  Ops1<TypeParam> ops(10);

  xp::distributed_vector<int> toCompareXp(10);

  for (std::size_t idx = 0; idx < 10; idx++) {
    toCompareXp[idx] = ops.dist_vec[idx];
  }
  barrier();

  bool xpEq = xp::equal(ops.dist_vec, toCompareXp);

  EXPECT_TRUE(xpEq);
}

TYPED_TEST(Equals, Different) {
  Ops1<TypeParam> ops(10);

  xp::distributed_vector<int> toCompareXp(10);

  for (std::size_t idx = 0; idx < 10; idx++) {
    toCompareXp[idx] = ops.dist_vec[idx];
  }

  toCompareXp[2] = ops.dist_vec[2] + 1;

  barrier();

  bool xpEq = xp::equal(ops.dist_vec, toCompareXp);

  EXPECT_TRUE(!xpEq);
}

TYPED_TEST(Equals, EmptiesEqual) {
  Ops1<TypeParam> ops(0);

  xp::distributed_vector<int> toCompareXp(0);

  bool xpEq = xp::equal(ops.dist_vec, toCompareXp);

  EXPECT_TRUE(xpEq);
}

TYPED_TEST(Equals, EmptyNotEmptyDifferent) {
  Ops1<TypeParam> ops(0);

  xp::distributed_vector<int> toCompareXp(10);

  bool xpEq = xp::equal(ops.dist_vec, toCompareXp);

  EXPECT_TRUE(!xpEq);
}
