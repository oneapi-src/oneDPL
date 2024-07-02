// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "copy.hpp"

TYPED_TEST_SUITE(CopyTest, AllocatorTypes);

TYPED_TEST(CopyTest, tests_from_this_file_run_on_3_devices) {
  EXPECT_EQ(dr::shp::nprocs(), 3);
  EXPECT_EQ(rng::size(dr::shp::devices()), 3);
}

TYPED_TEST(CopyTest, dist2local_wholesegment) {
  // when running on 3 devices copy exactly one segment
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4,  5,  6,
                                                  7, 8, 9, 10, 11, 12};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0};

  auto ret_it = dr::shp::copy(rng::begin(dist_vec) + 4,
                              rng::begin(dist_vec) + 8, rng::begin(local_vec));
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{5, 6, 7, 8}));
  EXPECT_EQ(ret_it, rng::end(local_vec));
}

TYPED_TEST(CopyTest, local2dist_wholesegment) {
  // when running on 3 devices copy into exactly one segment
  const typename TestFixture::LocalVec local_vec = {50, 60, 70, 80};
  typename TestFixture::DistVec dist_vec = {1, 2, 3, 4,  5,  6,
                                            7, 8, 9, 10, 11, 12};
  auto ret_it = dr::shp::copy(rng::begin(local_vec), rng::end(local_vec),
                              rng::begin(dist_vec) + 4);
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{
                                  1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12}));
  EXPECT_EQ(*ret_it, 9);
}
