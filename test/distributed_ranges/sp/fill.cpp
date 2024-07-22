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

template <typename AllocT> class FillTest : public testing::Test {
public:
  using DistVec =
      dr::sp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
};

TYPED_TEST_SUITE(FillTest, AllocatorTypes);

// tests of fill are WIP, below test will be refactored, new tests will be added
TYPED_TEST(FillTest, fill_all) {
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto segments = dist_vec.segments();
  int value = 1;
  for (auto &&segment : segments) {
    dr::sp::fill(segment.begin(), segment.end(), value);
  }
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::DistVec{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  ;
}
