// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename... Rs> auto test_zip(Rs &&...rs) {
  return xhp::views::zip(std::forward<Rs>(rs)...);
}

// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE(Zip, AllTypes);

TYPED_TEST(Zip, Dist1) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec);
  auto dist = test_zip(ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist2) {
  Ops2<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec0, ops.vec1);
  auto dist = test_zip(ops.dist_vec0, ops.dist_vec1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist3) {
  Ops3<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec0, ops.vec1, ops.vec2);
  auto dist = test_zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist3Distance) {
  Ops3<TypeParam> ops(10);

  EXPECT_EQ(
      rng::distance(rng::views::zip(ops.vec0, ops.vec1, ops.vec2)),
      rng::distance(test_zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

TYPED_TEST(Zip, RangeSegments) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec);
  auto dist = test_zip(ops.dist_vec);
  auto flat = rng::views::join(dr::ranges::segments(dist));
  EXPECT_TRUE(is_equal(local, flat));
}

#ifndef TEST_SHP
// Will not compile with SHP
TYPED_TEST(Zip, IterSegments) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec);
  auto dist = test_zip(ops.dist_vec);
  auto flat = rng::views::join(dr::ranges::segments(dist.begin()));
  EXPECT_TRUE(is_equal(local, flat));
}
#endif

TYPED_TEST(Zip, Drop) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::drop(rng::views::zip(ops.vec), 2);
  auto dist = xhp::views::drop(test_zip(ops.dist_vec), 2);

  auto flat = rng::views::join(dr::ranges::segments(dist));
  EXPECT_EQ(local, dist);
  EXPECT_TRUE(is_equal(local, flat));
}

TYPED_TEST(Zip, ConsumingAll) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(rng::views::all(ops.vec));
  auto dist = test_zip(xhp::views::all(ops.dist_vec));
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(Zip, FeedingAll) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::all(rng::views::zip(ops.vec));
  auto dist = xhp::views::all(test_zip(ops.dist_vec));
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(Zip, ForEach) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(test_zip(ops.dist_vec0, ops.dist_vec1), copy);
  rng::for_each(rng::views::zip(ops.vec0, ops.vec1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Zip, ForEachDrop) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(xhp::views::drop(test_zip(ops.dist_vec0, ops.dist_vec1), 1),
                copy);
  rng::for_each(xhp::views::drop(rng::views::zip(ops.vec0, ops.vec1), 1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Zip, ConsumingSubrange) {
  Ops2<TypeParam> ops(10);

  auto local =
      rng::views::zip(rng::subrange(ops.vec0.begin() + 1, ops.vec0.end() - 1),
                      rng::subrange(ops.vec1.begin() + 1, ops.vec1.end() - 1));
  auto dist = test_zip(
      rng::subrange(ops.dist_vec0.begin() + 1, ops.dist_vec0.end() - 1),
      rng::subrange(ops.dist_vec1.begin() + 1, ops.dist_vec1.end() - 1));
  EXPECT_EQ(local, dist);
}

TEST(Zip, FeedingTransform) {
  Ops2<xhp::distributed_vector<int>> ops(10);

  auto mul = [](auto v) { return std::get<0>(v) * std::get<1>(v); };
  auto local = rng::views::transform(rng::views::zip(ops.vec0, ops.vec1), mul);
  auto dist_zip = test_zip(ops.dist_vec0, ops.dist_vec1);
  auto dist = xhp::views::transform(dist_zip, mul);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TEST(Zip, CopyConstructor) {
  Ops2<xhp::distributed_vector<int>> ops(10);

  auto dist = test_zip(ops.dist_vec0, ops.dist_vec1);
  auto dist_copy(dist);
  EXPECT_EQ(dist, dist_copy);
}

TYPED_TEST(Zip, TransformReduce) {
  Ops2<TypeParam> ops(10);

  auto mul = [](auto v) { return std::get<0>(v) * std::get<1>(v); };

  auto local = rng::views::transform(rng::views::zip(ops.vec0, ops.vec1), mul);
  auto local_reduce = std::reduce(local.begin(), local.end());

  auto dist =
      xhp::views::transform(test_zip(ops.dist_vec0, ops.dist_vec1), mul);
  auto dist_reduce = xhp::reduce(dist);

  EXPECT_EQ(local_reduce, dist_reduce);
}

TYPED_TEST(Zip, IotaStaticAssert) {
  Ops1<TypeParam> ops(10);

  auto dist = test_zip(xhp::views::iota(100), ops.dist_vec);
  static_assert(std::forward_iterator<decltype(dist.begin())>);
  static_assert(std::forward_iterator<decltype(dist.end())>);
  static_assert(std::forward_iterator<decltype(rng::begin(dist))>);
  static_assert(std::forward_iterator<decltype(rng::end(dist))>);
  using Dist = decltype(dist);
  static_assert(rng::forward_range<Dist>);
  static_assert(dr::distributed_range<Dist>);
}

TYPED_TEST(Zip, Iota) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(rng::views::iota(100), ops.vec);
  auto dist = test_zip(xhp::views::iota(100), ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}

TYPED_TEST(Zip, Iota2nd) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::zip(ops.vec, rng::views::iota(100)),
                         test_zip(ops.dist_vec, xhp::views::iota(100))));
}

TYPED_TEST(Zip, ForEachIota) {
  Ops1<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(test_zip(xhp::views::iota(100), ops.dist_vec), copy);
  rng::for_each(rng::views::zip(rng::views::iota(100), ops.vec), copy);

  EXPECT_EQ(ops.vec, ops.dist_vec);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}
