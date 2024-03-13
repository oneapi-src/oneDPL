// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename DistVecT> class IotaView : public testing::Test {
public:
};

TYPED_TEST_SUITE(IotaView, AllTypes);

TYPED_TEST(IotaView, ZipWithDR) {
  xhp::distributed_vector<int> dv(10);
  auto v = dr::views::iota(1, 10);

  auto z = xhp::views::zip(dv, v);

  xhp::for_each(z, [](auto ze) {
    auto [dve, ve] = ze;
    dve = ve;
  });

  EXPECT_TRUE(equal(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, dv));
}

TYPED_TEST(IotaView, Copy) {
  TypeParam dv(10);
  auto v = dr::views::iota(1, 11);

  xhp::copy(v, dv.begin());

  barrier();
  EXPECT_TRUE(equal(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, dv));
}

TYPED_TEST(IotaView, Transform) {
  TypeParam dv(10);
  auto v = dr::views::iota(1, 11);
  auto negate = [](auto v) { return -v; };

  xhp::transform(v, dv.begin(), negate);

  EXPECT_TRUE(
      equal(dv, std::vector<int>{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}));
}

TYPED_TEST(IotaView, ForEach) {
  TypeParam dv(10);
  auto v = dr::views::iota(1, 11);

  auto negate = [](auto v) {
    auto &[in, out] = v;
    out = -in;
  };

  auto z = xhp::views::zip(v, dv);

  xhp::for_each(z, negate);

  EXPECT_TRUE(
      equal(dv, std::vector<int>{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}));
}
