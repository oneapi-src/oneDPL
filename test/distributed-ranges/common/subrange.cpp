// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Subrange : public testing::Test {
public:
};

TYPED_TEST_SUITE(Subrange, AllTypes);

TYPED_TEST(Subrange, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1);
  auto dist = rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Subrange, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(
      ops, rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1),
      rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1)));
}

TYPED_TEST(Subrange, ForEach) {
  Ops1<TypeParam> ops(23);

  auto local = rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 2);
  auto dist = rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 2);

  auto negate = [](auto v) { return -v; };
  rng::for_each(local, negate);
  xhp::for_each(dist, negate);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TYPED_TEST(Subrange, Transform) {
  TypeParam v1(13), v2(13);
  xhp::iota(v1, 10);
  xhp::fill(v2, -1);

  auto s1 = rng::subrange(v1.begin() + 1, v1.end() - 2);
  auto s2 = rng::subrange(v2.begin() + 1, v2.end() - 2);

  auto null_op = [](auto v) { return v; };
  xhp::transform(s1, s2.begin(), null_op);

  EXPECT_TRUE(equal(v2, std::vector<int>{-1, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                         20, -1, -1}));
}

TYPED_TEST(Subrange, Reduce) {
  Ops1<TypeParam> ops(23);

  auto local = rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 2);
  auto dist = rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 2);

  EXPECT_EQ(std::reduce(local.begin(), local.end(), 3, std::plus{}),
            xhp::reduce(dist, 3, std::plus{}));
}
