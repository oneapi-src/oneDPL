// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/cmath>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto с = oneapi::dpl::cos(0);
    return TestUtils::done();
}
