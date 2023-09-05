// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdexcept>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::logic_error err{"error"};
    return TestUtils::done();
}
