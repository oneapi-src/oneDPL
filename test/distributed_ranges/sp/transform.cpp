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

template <typename AllocT> class TransformTest : public testing::Test {
public:
  using DistVec =
      dr::sp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
  constexpr static const auto add_10_func = [](auto x) { return x + 10; };
};

TYPED_TEST_SUITE(TransformTest, AllocatorTypes);

TYPED_TEST(TransformTest, whole_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9};
  auto r = dr::sp::transform(dr::sp::par_unseq, a, stdrng::begin(b),
                              TestFixture::add_10_func);
  EXPECT_EQ(r.in, a.end());
  EXPECT_EQ(r.out, b.end());

  EXPECT_TRUE(gtest_equal(b, typename TestFixture::LocalVec{10, 11, 12, 13, 14}));
}

TYPED_TEST(TransformTest, whole_non_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {50, 51, 52, 53, 54, 55,
                                     56, 57, 58, 59, 60};

  auto r = dr::sp::transform(dr::sp::par_unseq, a, stdrng::begin(b),
                              TestFixture::add_10_func);
  EXPECT_EQ(r.in, a.end());
  EXPECT_EQ(*r.out, 55);

  EXPECT_TRUE(gtest_equal(b, typename TestFixture::LocalVec{10, 11, 12, 13, 14, 55,
                                                      56, 57, 58, 59, 60}));
}

TYPED_TEST(TransformTest, part_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9};

  auto [r_in, r_out] = dr::sp::transform(
      dr::sp::par_unseq, stdrng::subrange(++stdrng::begin(a), --stdrng::end(a)),
      ++stdrng::begin(b), TestFixture::add_10_func);
  EXPECT_EQ(*r_in, 4);
  EXPECT_EQ(*r_out, 9);

  EXPECT_TRUE(gtest_equal(b, typename TestFixture::LocalVec{9, 11, 12, 13, 9}));
}

TYPED_TEST(TransformTest, part_not_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9, 9, 9, 9, 9};

  auto [r_in, r_out] = dr::sp::transform(
      dr::sp::par_unseq, stdrng::subrange(++stdrng::begin(a), stdrng::end(a)),
      stdrng::begin(b) + 5, TestFixture::add_10_func);
  EXPECT_EQ(r_in, a.end());
  EXPECT_EQ(r_out, stdrng::begin(b) + 8); // initial shift in b + subrange size

  EXPECT_TRUE(
      gtest_equal(b, typename TestFixture::LocalVec{9, 9, 9, 9, 9, 11, 12, 13, 9}));
}

TYPED_TEST(TransformTest, inplace_whole) {
  typename TestFixture::DistVec a = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto [r_in, r_out] = dr::sp::transform(dr::sp::par_unseq, a, stdrng::begin(a),
                                          TestFixture::add_10_func);
  EXPECT_EQ(r_in, stdrng::end(a));
  EXPECT_EQ(r_out, stdrng::end(a));
  EXPECT_TRUE(gtest_equal(
      a, typename TestFixture::LocalVec{10, 11, 12, 13, 14, 15, 16, 17, 18}));
}

TYPED_TEST(TransformTest, inplace_part) {
  typename TestFixture::DistVec a = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto [r_in, r_out] = dr::sp::transform(
      dr::sp::par_unseq, stdrng::subrange(++stdrng::begin(a), --stdrng::end(a)),
      ++stdrng::begin(a), TestFixture::add_10_func);
  EXPECT_EQ(*r_in, 8);
  EXPECT_EQ(r_out, --stdrng::end(a));
  EXPECT_TRUE(gtest_equal(
      a, typename TestFixture::LocalVec{0, 11, 12, 13, 14, 15, 16, 17, 8}));
}

TYPED_TEST(TransformTest, large_aligned_whole) {
  const typename TestFixture::DistVec a(12345, 7);
  typename TestFixture::DistVec b(12345, 3);
  dr::sp::transform(dr::sp::par_unseq, a, stdrng::begin(b),
                     TestFixture::add_10_func);

  EXPECT_EQ(b[0], 17);
  EXPECT_EQ(b[1111], 17);
  EXPECT_EQ(b[2222], 17);
  EXPECT_EQ(b[3333], 17);
  EXPECT_EQ(b[4444], 17);
  EXPECT_EQ(b[5555], 17);
  EXPECT_EQ(b[6666], 17);
  EXPECT_EQ(b[7777], 17);
  EXPECT_EQ(b[8888], 17);
  EXPECT_EQ(b[9999], 17);
  EXPECT_EQ(b[11111], 17);
  EXPECT_EQ(b[12222], 17);
  EXPECT_EQ(b[12344], 17);
}

TYPED_TEST(TransformTest, large_aligned_part) {
  const typename TestFixture::DistVec a(12345, 7);
  typename TestFixture::DistVec b(12345, 3);
  dr::sp::transform(dr::sp::par_unseq,
                     stdrng::subrange(stdrng::begin(a) + 1000, stdrng::begin(a) + 1005),
                     stdrng::begin(b) + 1000, TestFixture::add_10_func);

  EXPECT_EQ(b[998], 3);
  EXPECT_EQ(b[999], 3);
  EXPECT_EQ(b[1000], 17);
  EXPECT_EQ(b[1001], 17);
  EXPECT_EQ(b[1002], 17);
  EXPECT_EQ(b[1003], 17);
  EXPECT_EQ(b[1004], 17);
  EXPECT_EQ(b[1005], 3);
}

TYPED_TEST(TransformTest, large_aligned_part_shifted) {
  const typename TestFixture::DistVec a(12345, 7);
  typename TestFixture::DistVec b(12345, 3);
  dr::sp::transform(dr::sp::par_unseq,
                     stdrng::subrange(stdrng::begin(a) + 1000, stdrng::begin(a) + 1005),
                     stdrng::begin(b) + 999, TestFixture::add_10_func);

  EXPECT_EQ(b[998], 3);
  EXPECT_EQ(b[999], 17);
  EXPECT_EQ(b[1000], 17);
  EXPECT_EQ(b[1001], 17);
  EXPECT_EQ(b[1002], 17);
  EXPECT_EQ(b[1003], 17);
  EXPECT_EQ(b[1004], 3);
  EXPECT_EQ(b[1005], 3);
}

TYPED_TEST(TransformTest, large_not_aligned) {
  const typename TestFixture::DistVec a(10000, 7);
  typename TestFixture::DistVec b(17000, 3);
  dr::sp::transform(dr::sp::par_unseq,
                     stdrng::subrange(stdrng::begin(a) + 2000, stdrng::begin(a) + 9000),
                     stdrng::begin(b) + 9000, TestFixture::add_10_func);

  EXPECT_EQ(b[8999], 3);
  EXPECT_EQ(b[9000], 17);
  EXPECT_EQ(b[9001], 17);

  EXPECT_EQ(b[9999], 17);
  EXPECT_EQ(b[12345], 17);
  EXPECT_EQ(b[13456], 17);
  EXPECT_EQ(b[14567], 17);

  EXPECT_EQ(b[15999], 17);
  EXPECT_EQ(b[16000], 3);
  EXPECT_EQ(b[16001], 3);
}

TYPED_TEST(TransformTest, large_inplace) {
  typename TestFixture::DistVec a(77000, 7);
  auto r = dr::sp::transform(
      dr::sp::par_unseq,
      stdrng::subrange(stdrng::begin(a) + 22222, stdrng::begin(a) + 55555),
      stdrng::begin(a) + 22222, TestFixture::add_10_func);

  EXPECT_EQ(r.in, stdrng::begin(a) + 55555);
  EXPECT_EQ(r.out, stdrng::begin(a) + 55555);

  EXPECT_EQ(a[11111], 7);
  EXPECT_EQ(a[22221], 7);
  EXPECT_EQ(a[22222], 17);

  EXPECT_EQ(a[33333], 17);
  EXPECT_EQ(a[44444], 17);

  EXPECT_EQ(a[55554], 17);
  EXPECT_EQ(a[55555], 7);

  EXPECT_EQ(a[66666], 7);
}
