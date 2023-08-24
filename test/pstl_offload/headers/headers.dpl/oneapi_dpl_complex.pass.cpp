// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/complex>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::dpl::complex<float> n{0, 0};
    return TestUtils::done();
}
