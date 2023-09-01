// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <compare>
#include "support/utils.h"

int main() {
#if __cpp_lib_three_way_comparison >= 201907L
    [[maybe_unused]] std::size_t r = sizeof(std::strong_ordering);
#endif
    return TestUtils::done();
}
