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

#pragma once

#include "xhp_tests.hpp"

template <typename AllocT> class CopyTest : public testing::Test {
public:
  using DistVec =
      dr::shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
};
