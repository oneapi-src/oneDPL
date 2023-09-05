// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <filesystem>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::filesystem::path p1;
    return TestUtils::done();
}
