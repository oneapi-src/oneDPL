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
template <typename T> class Subrange : public testing::Test {
public:
};

TYPED_TEST_SUITE(Subrange, AllTypes);

TYPED_TEST(Subrange, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1);
  auto dist = stdrng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Subrange, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(
      ops, stdrng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1),
      stdrng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1)));
}

TYPED_TEST(Subrange, ForEach) {
  Ops1<TypeParam> ops(23);

  auto local = stdrng::subrange(ops.vec.begin() + 1, ops.vec.end() - 2);
  auto dist = stdrng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 2);

  auto negate = [](auto v) { return -v; };
  stdrng::for_each(local, negate);
  xp::for_each(dist, negate);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TYPED_TEST(Subrange, Transform) {
  TypeParam v1(13), v2(13);
  xp::iota(v1, 10);
  xp::fill(v2, -1);

  auto s1 = stdrng::subrange(v1.begin() + 1, v1.end() - 2);
  auto s2 = stdrng::subrange(v2.begin() + 1, v2.end() - 2);

  auto null_op = [](auto v) { return v; };
  xp::transform(s1, s2.begin(), null_op);

  EXPECT_TRUE(equal(v2, std::vector<int>{-1, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                         20, -1, -1}));
}

TYPED_TEST(Subrange, Reduce) {
  Ops1<TypeParam> ops(23);

  auto local = stdrng::subrange(ops.vec.begin() + 1, ops.vec.end() - 2);
  auto dist = stdrng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 2);

  EXPECT_EQ(std::reduce(local.begin(), local.end(), 3, std::plus{}),
            xp::reduce(dist, 3, std::plus{}));
}
