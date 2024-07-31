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
template <typename T> class Enumerate : public testing::Test {
public:
};

TYPED_TEST_SUITE(Enumerate, AllTypes);

TYPED_TEST(Enumerate, Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(std::ranges::views::enumerate(ops.vec),
                         xp::views::enumerate(ops.dist_vec)));
}

TYPED_TEST(Enumerate, Mutate) {
  Ops1<TypeParam> ops(10);
  auto local = std::ranges::views::enumerate(ops.vec);
  auto dist = xp::views::enumerate(ops.dist_vec);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xp::for_each(dist, copy);
  stdrng::for_each(local, copy);

  EXPECT_EQ(local, dist);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}
