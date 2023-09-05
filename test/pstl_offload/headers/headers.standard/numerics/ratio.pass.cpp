// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ratio>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::ratio<1, 10> r;
    return TestUtils::done();
}
