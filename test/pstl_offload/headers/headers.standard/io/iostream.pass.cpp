// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::ostream& stream = std::cout;
    return TestUtils::done();
}
