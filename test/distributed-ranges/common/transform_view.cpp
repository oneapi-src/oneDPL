// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class TransformView : public testing::Test {
public:
};

TYPED_TEST_SUITE(TransformView, AllTypes);

TYPED_TEST(TransformView, Basic) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto local = rng::views::transform(ops.vec, negate);
  auto dist = xhp::views::transform(ops.dist_vec, negate);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(TransformView, All) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  EXPECT_TRUE(
      check_view(rng::views::transform(rng::views::all(ops.vec), negate),
                 xhp::views::transform(rng::views::all(ops.dist_vec), negate)));
}

TYPED_TEST(TransformView, Move) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto l_value = rng::views::all(ops.vec);
  auto dist_l_value = rng::views::all(ops.dist_vec);
  EXPECT_TRUE(
      check_view(rng::views::transform(std::move(l_value), negate),
                 xhp::views::transform(std::move(dist_l_value), negate)));
}

TYPED_TEST(TransformView, L_Value) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto l_value = rng::views::all(ops.vec);
  auto dist_l_value = rng::views::all(ops.dist_vec);
  EXPECT_TRUE(check_view(rng::views::transform(l_value, negate),
                         xhp::views::transform(dist_l_value, negate)));
}

TYPED_TEST(TransformView, ForEach) {
  Ops1<TypeParam> ops(10);

  auto null_transform = [](auto &&v) { return v; };
  auto local = rng::views::transform(ops.vec, null_transform);
  auto dist = xhp::views::transform(ops.dist_vec, null_transform);

  auto null_for_each = [](auto v) {};
  rng::for_each(local, null_for_each);
  xhp::for_each(dist, null_for_each);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TYPED_TEST(TransformView, Reduce) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &&v) { return -v; };
  auto local = rng::views::transform(ops.vec, negate);
  auto dist = xhp::views::transform(ops.dist_vec, negate);
  EXPECT_EQ(std::reduce(local.begin(), local.end(), 3, std::plus{}),
            xhp::reduce(dist, 3, std::plus{}));
}
