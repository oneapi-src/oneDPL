// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <complex>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::complex<int> c(0, 0);
    return TestUtils::done();
}
