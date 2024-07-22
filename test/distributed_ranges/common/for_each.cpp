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
template <typename T> class ForEach : public testing::Test {
public:
};

template <typename T>
void test_foreach_n(std::vector<T> v, int n, int initial_skip, auto func) {
  auto size = v.size();
  xp::distributed_vector<T> d_v(size);

  for (std::size_t idx = 0; idx < size; idx++) {
    d_v[idx] = v[idx];
  }
  barrier();

  xp::for_each_n(d_v.begin() + initial_skip, n, func);
  stdrng::for_each_n(v.begin() + initial_skip, n, func);

  EXPECT_TRUE(gtest_equal(v, d_v));
}

TYPED_TEST_SUITE(ForEach, AllTypes);

TYPED_TEST(ForEach, Range) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { v = -v; };
  auto input = ops.vec;

  xp::for_each(ops.dist_vec, negate);
  stdrng::for_each(ops.vec, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(ForEach, Iterators) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { v = -v; };
  auto input = ops.vec;

  xp::for_each(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1, negate);
  stdrng::for_each(ops.vec.begin() + 1, ops.vec.end() - 1, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(ForEach, RangeAlignedZip) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto v) { std::get<0>(v) = std::get<1>(v); };
  auto dist = xp::views::zip(ops.dist_vec0, ops.dist_vec1);
  auto local = stdrng::views::zip(ops.vec0, ops.vec1);

  xp::for_each(dist, copy);
  stdrng::for_each(local, copy);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(ForEach, ForEachN) {
  auto negate = [](auto &&v) { v = -v; };
  test_foreach_n<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 0, negate);
}

TYPED_TEST(ForEach, ForEachNLongerThanSize) {
  auto negate = [](auto &&v) { v = -v; };
  test_foreach_n<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 10, 0, negate);
}

TYPED_TEST(ForEach, ForEachNPartial) {
  auto negate = [](auto &&v) { v = -v; };
  test_foreach_n<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5, 2, negate);
}

TYPED_TEST(ForEach, ForEachNWholeLength) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { v = -v; };
  auto input = ops.vec;

  xp::for_each_n(ops.dist_vec.begin(), 10, negate);
  stdrng::for_each(ops.vec.begin(), ops.vec.end(), negate);

  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

// Disabled. Not sure how to support this properly for MHP. We need to
// copy the values local so we can operate on them. Read-only data
// seems doable but writing misaligned data is harder. We should
// support some algorithms that do data movements that align data.
TYPED_TEST(ForEach, DISABLED_RangeUnalignedZip) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto v) { std::get<0>(v) = std::get<1>(v); };
  auto dist =
      xp::views::zip(xp::views::drop(ops.dist_vec0, 1), ops.dist_vec1);
  auto local = stdrng::views::zip(stdrng::views::drop(ops.vec0, 1), ops.vec1);

  xp::for_each(dist, copy);
  stdrng::for_each(local, copy);
  EXPECT_EQ(local, dist);
}
