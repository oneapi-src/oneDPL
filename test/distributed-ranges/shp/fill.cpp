// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "xhp-tests.hpp"

template <typename AllocT> class FillTest : public testing::Test {
public:
  using DistVec =
      dr::shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
};

TYPED_TEST_SUITE(FillTest, AllocatorTypes);

// tests of fill are WIP, below test will be refactored, new tests will be added
TYPED_TEST(FillTest, fill_all) {
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto segments = dist_vec.segments();
  int value = 1;
  for (auto &&segment : segments) {
    dr::shp::fill(segment.begin(), segment.end(), value);
  }
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::DistVec{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  ;
}
