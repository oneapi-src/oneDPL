// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "xhp-tests.hpp"

template <typename AllocT> class DistributedVectorTest : public testing::Test {
public:
  using DistVec =
      experimental::dr::shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
};
