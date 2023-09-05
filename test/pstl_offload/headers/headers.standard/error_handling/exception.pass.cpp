// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <exception>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::exception ex;
    return TestUtils::done();
}
