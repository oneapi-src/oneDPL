// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string_view>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::string_view sv;
    return TestUtils::done();
}
