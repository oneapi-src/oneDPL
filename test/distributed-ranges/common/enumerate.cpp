// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Enumerate : public testing::Test {
public:
};

TYPED_TEST_SUITE(Enumerate, AllTypes);

TYPED_TEST(Enumerate, Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::enumerate(ops.vec),
                         xhp::views::enumerate(ops.dist_vec)));
}

TYPED_TEST(Enumerate, Mutate) {
  Ops1<TypeParam> ops(10);
  auto local = rng::views::enumerate(ops.vec);
  auto dist = xhp::views::enumerate(ops.dist_vec);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(dist, copy);
  rng::for_each(local, copy);

  EXPECT_EQ(local, dist);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}
