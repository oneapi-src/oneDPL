// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <source_location>
#include "support/utils.h"

int main() {
#if __cpp_lib_source_location >= 201907L
    [[maybe_unused]] std::size_t r = sizeof(std::source_location);
#endif
    return TestUtils::done();
}
