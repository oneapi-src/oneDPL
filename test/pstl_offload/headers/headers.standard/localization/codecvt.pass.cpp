// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <codecvt>
#include "support/utils.h"

int main() {
    // TODO: header is deprecated in C++17
    [[maybe_unused]] auto mode = std::codecvt_mode::little_endian;
    return TestUtils::done();
}
