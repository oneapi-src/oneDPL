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
template <typename T> class All : public testing::Test {
public:
};

TYPED_TEST_SUITE(All, AllTypes);

TYPED_TEST(All, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = stdrng::views::all(ops.vec);
  auto dist = xp::views::all(ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(All, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, stdrng::views::all(ops.vec),
                                xp::views::all(ops.dist_vec)));
}
