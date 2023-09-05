// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <version>
#include "support/utils.h"

int main() {
    // Relying on the implementation details that if <version> header is present,
    // C++14 feature testing macros are defined independently of -std= compiler option
    [[maybe_unused]] std::size_t r = __cpp_lib_transformation_trait_aliases;
    return TestUtils::done();
}
