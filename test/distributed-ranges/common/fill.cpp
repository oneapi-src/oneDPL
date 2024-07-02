// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Fill : public testing::Test {
public:
};

TYPED_TEST_SUITE(Fill, AllTypes);

TYPED_TEST(Fill, Range) {
  Ops1<TypeParam> ops(10);

  auto input = ops.vec;

  xhp::fill(ops.dist_vec, 33);
  rng::fill(ops.vec, 33);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(Fill, Iterators) {
  Ops1<TypeParam> ops(10);

  auto input = ops.vec;

  xhp::fill(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1, 33);
  rng::fill(ops.vec.begin() + 1, ops.vec.end() - 1, 33);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(Fill, Iterators_large) {
  TypeParam large_dist_vec(80000);
  xhp::fill(large_dist_vec.begin() + 71000, large_dist_vec.end(), 33);
  EXPECT_EQ(large_dist_vec[77777], 33);
}

TYPED_TEST(Fill, Iterators_large_segment) {
  TypeParam large_dist_vec(80000);
  auto last_segment = large_dist_vec.segments().back();
  xhp::fill(last_segment.begin(), last_segment.end(),
            static_cast<typename TypeParam::value_type>(33));
  EXPECT_EQ(large_dist_vec[79999], 33);
}
