// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <latch>
#include "support/utils.h"

int main() {
#if __cpp_lib_latch >= 201907L
    [[maybe_unused]] std::latch l(1);
#endif
    return TestUtils::done();
}
