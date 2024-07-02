// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class ForEach : public testing::Test {
public:
};

TYPED_TEST_SUITE(ForEach, AllTypes);

TYPED_TEST(ForEach, Range) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { v = -v; };
  auto input = ops.vec;

  xhp::for_each(ops.dist_vec, negate);
  rng::for_each(ops.vec, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(ForEach, Iterators) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { v = -v; };
  auto input = ops.vec;

  xhp::for_each(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1, negate);
  rng::for_each(ops.vec.begin() + 1, ops.vec.end() - 1, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(ForEach, RangeAlignedZip) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto v) { std::get<0>(v) = std::get<1>(v); };
  auto dist = xhp::views::zip(ops.dist_vec0, ops.dist_vec1);
  auto local = rng::views::zip(ops.vec0, ops.vec1);

  xhp::for_each(dist, copy);
  rng::for_each(local, copy);
  EXPECT_EQ(local, dist);
}

// Disabled. Not sure how to support this properly for MHP. We need to
// copy the values local so we can operate on them. Read-only data
// seems doable but writing misaligned data is harder. We should
// support some algorithms that do data movements that align data.
TYPED_TEST(ForEach, DISABLED_RangeUnalignedZip) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto v) { std::get<0>(v) = std::get<1>(v); };
  auto dist =
      xhp::views::zip(xhp::views::drop(ops.dist_vec0, 1), ops.dist_vec1);
  auto local = rng::views::zip(rng::views::drop(ops.vec0, 1), ops.vec1);

  xhp::for_each(dist, copy);
  rng::for_each(local, copy);
  EXPECT_EQ(local, dist);
}
