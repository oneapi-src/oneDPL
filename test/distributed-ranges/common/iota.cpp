// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename DistVecT> class IotaTest : public testing::Test {
public:
};

TYPED_TEST_SUITE(IotaTest, AllTypes);

TYPED_TEST(IotaTest, Range) {
  TypeParam v(10);
  xhp::iota(v, 1);
  EXPECT_EQ(v, (LocalVec<TypeParam>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

TYPED_TEST(IotaTest, Iter) {
  TypeParam v(10);
  xhp::iota(v.begin(), v.end(), 10);
  EXPECT_EQ(v, (LocalVec<TypeParam>{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

TYPED_TEST(IotaTest, PartialIter) {
  TypeParam v(10, 99);
  xhp::iota(++v.begin(), --v.end(), 0);
  EXPECT_EQ(v, (LocalVec<TypeParam>{99, 0, 1, 2, 3, 4, 5, 6, 7, 99}));
}

TYPED_TEST(IotaTest, SlicedLeft) {
  TypeParam dist_vec(10, 0);
  xhp::iota(dist_vec.begin() + 2, dist_vec.end(), 2);
  EXPECT_TRUE(
      equal(dist_vec, LocalVec<TypeParam>{0, 0, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(IotaTest, SlicedRight) {
  TypeParam dist_vec(10, 0);
  xhp::iota(dist_vec.begin(), dist_vec.end() - 2, 2);
  EXPECT_TRUE(
      equal(dist_vec, LocalVec<TypeParam>{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}

TYPED_TEST(IotaTest, Large) {
  TypeParam v(98765);
  xhp::iota(v, 0);
  EXPECT_EQ(v[33000], 33000);
  EXPECT_EQ(v[66000], 66000);
  EXPECT_EQ(v[91000], 91000);
}
