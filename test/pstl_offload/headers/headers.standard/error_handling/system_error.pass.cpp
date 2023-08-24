// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <system_error>
#include "support/utils.h"

int main() {
    try {
        throw std::system_error(std::error_code{});
    } catch(...) {}
    return TestUtils::done();
}
