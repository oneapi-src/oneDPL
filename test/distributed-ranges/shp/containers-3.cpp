// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "containers.hpp"

TYPED_TEST_SUITE(DistributedVectorTest, AllocatorTypes);

TYPED_TEST(DistributedVectorTest, tests_from_this_file_run_on_3_devices) {
  EXPECT_EQ(dr::shp::nprocs(), 3);
  EXPECT_EQ(std::size(dr::shp::devices()), 3);
}

TYPED_TEST(DistributedVectorTest, segments_sizes_in_uneven_distribution) {
  typename TestFixture::DistVec dv(10);
  EXPECT_EQ(rng::size(dv.segments()), 3);
  EXPECT_EQ(rng::size(dv.segments()[0]), 4);
  EXPECT_EQ(rng::size(dv.segments()[1]), 4);
  EXPECT_EQ(rng::size(dv.segments()[2]), 2);
}

TYPED_TEST(DistributedVectorTest, segments_sizes_in_even_distribution) {
  typename TestFixture::DistVec dv(12);
  EXPECT_EQ(rng::size(dv.segments()), 3);
  EXPECT_EQ(rng::size(dv.segments()[0]), 4);
  EXPECT_EQ(rng::size(dv.segments()[1]), 4);
  EXPECT_EQ(rng::size(dv.segments()[2]), 4);
}

TYPED_TEST(DistributedVectorTest,
           segments_sizes_in_uneven_zeroending_distribution) {
  typename TestFixture::DistVec dv(4);
  EXPECT_EQ(rng::size(dv.segments()), 2);
  EXPECT_EQ(rng::size(dv.segments()[0]), 2);
  EXPECT_EQ(rng::size(dv.segments()[1]), 2);
}

TYPED_TEST(DistributedVectorTest, segments_sizes_in_empty_vec) {
  // this is not consistent, for non-zero sizes we do not return empty segments
  // but in case of empty vec we return one empty segment, IMO it should made
  // consistent in some way
  typename TestFixture::DistVec dv(0);
  EXPECT_EQ(rng::size(dv.segments()), 1);
  EXPECT_EQ(rng::size(dv.segments()[0]), 0);
}

TYPED_TEST(DistributedVectorTest, segments_sizes_in_oneitem_vec) {
  typename TestFixture::DistVec dv(1);
  EXPECT_EQ(rng::size(dv.segments()), 1);
  EXPECT_EQ(rng::size(dv.segments()[0]), 1);
}

TYPED_TEST(DistributedVectorTest, segments_joint_view_same_as_all_view) {
  using DV = typename TestFixture::DistVec;
  check_segments(DV(0));
  check_segments(DV(1));
  check_segments(DV(4));
  check_segments(DV(10));
  check_segments(DV(12));
}
