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
    try {
        throw std::logic_error{"error"};
    } catch(...) {}
    return TestUtils::done();
}
