// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <span>
#include "support/utils.h"

int main() {
#if __cpp_lib_span >= 202002L
    [[maybe_unused]] std::span<int> s;
#endif
    return TestUtils::done();
}
