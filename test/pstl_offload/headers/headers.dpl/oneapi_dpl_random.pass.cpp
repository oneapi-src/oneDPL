// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/random>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::dpl::minstd_rand0 gen;
    return TestUtils::done();
}
