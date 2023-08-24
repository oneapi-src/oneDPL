// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/array>
#include "support/utils.h"

int main() {
    oneapi::dpl::array<int, 1> array = {1};
    oneapi::dpl::get<0>(array) = 2;
    return TestUtils::done();
}
