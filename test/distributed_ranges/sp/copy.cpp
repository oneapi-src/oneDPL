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

#include "copy.hpp"

TYPED_TEST_SUITE(CopyTest, AllocatorTypes);

TYPED_TEST(CopyTest, dist2local_async) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0};
  dr::sp::copy_async(stdrng::begin(dist_vec), stdrng::end(dist_vec),
                      stdrng::begin(local_vec))
      .wait();
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_async) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  dr::sp::copy_async(stdrng::begin(local_vec), stdrng::end(local_vec),
                      stdrng::begin(dist_vec))
      .wait();
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_sync) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 9};
  auto ret_it = dr::sp::copy(stdrng::begin(dist_vec), stdrng::end(dist_vec),
                              stdrng::begin(local_vec));
  EXPECT_TRUE(
      equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 9}));
  EXPECT_EQ(*ret_it, 9);
}

TYPED_TEST(CopyTest, local2dist_sync) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 9};
  auto ret_it = dr::sp::copy(stdrng::begin(local_vec), stdrng::end(local_vec),
                              stdrng::begin(dist_vec));
  EXPECT_TRUE(
      equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 9}));
  EXPECT_EQ(*ret_it, 9);
}

TYPED_TEST(CopyTest, dist2local_range_sync) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 9};
  auto ret_it = dr::sp::copy(dist_vec, stdrng::begin(local_vec));
  EXPECT_TRUE(
      equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 9}));
  EXPECT_EQ(*ret_it, 9);
}

TYPED_TEST(CopyTest, local2dist_range_sync) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 9};
  auto ret_it = dr::sp::copy(local_vec, stdrng::begin(dist_vec));
  EXPECT_TRUE(
      equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 9}));
  EXPECT_EQ(*ret_it, 9);
}

TYPED_TEST(CopyTest, dist2local_async_can_interleave) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0};
  auto event_1 =
      dr::sp::copy_async(stdrng::begin(dist_vec) + 0, stdrng::begin(dist_vec) + 4,
                          stdrng::begin(local_vec) + 0);
  auto event_2 =
      dr::sp::copy_async(stdrng::begin(dist_vec) + 1, stdrng::begin(dist_vec) + 5,
                          stdrng::begin(local_vec) + 4);
  event_1.wait();
  event_2.wait();
  EXPECT_TRUE(
      equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_async_can_interleave) {
  const typename TestFixture::LocalVec local_vec_1 = {1, 2, 3};
  const typename TestFixture::LocalVec local_vec_2 = {4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  auto event_1 = dr::sp::copy_async(
      stdrng::begin(local_vec_1), stdrng::end(local_vec_1), stdrng::begin(dist_vec));
  auto event_2 = dr::sp::copy_async(
      stdrng::begin(local_vec_2), stdrng::end(local_vec_2), stdrng::begin(dist_vec) + 3);
  event_1.wait();
  event_2.wait();
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_sliced_bothSides) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  dr::sp::copy(stdrng::begin(dist_vec) + 1, stdrng::end(dist_vec) - 1,
                stdrng::begin(local_vec));
  EXPECT_TRUE(equal(
      local_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}

TYPED_TEST(CopyTest, dist2local_sliced_left) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  dr::sp::copy(stdrng::begin(dist_vec) + 1, stdrng::end(dist_vec),
                stdrng::begin(local_vec));
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7,
                                                              8, 9, 10, 0}));
}

TYPED_TEST(CopyTest, dist2local_sliced_right) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  dr::sp::copy(stdrng::begin(dist_vec), stdrng::end(dist_vec) - 1,
                stdrng::begin(local_vec));
  EXPECT_TRUE(equal(
      local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 6, 7, 8, 9, 0}));
}

TYPED_TEST(CopyTest, local2dist_sliced_bothSides) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  dr::sp::copy(stdrng::begin(local_vec), stdrng::end(local_vec),
                stdrng::begin(dist_vec) + 1);
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{0, 2, 3, 4, 5, 6, 7, 8, 9, 0}));
}

TYPED_TEST(CopyTest, local2dist_sliced_left) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  dr::sp::copy(stdrng::begin(local_vec), stdrng::end(local_vec),
                stdrng::begin(dist_vec) + 2);
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{0, 0, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(CopyTest, local2dist_sliced_right) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  dr::sp::copy(stdrng::begin(local_vec), stdrng::end(local_vec),
                stdrng::begin(dist_vec));
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}
