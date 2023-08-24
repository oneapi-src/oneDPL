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
    try {
        throw std::exception{};
    } catch(...) {}
    return TestUtils::done();
}
