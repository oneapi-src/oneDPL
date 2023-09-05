// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ranges>
#include "support/utils.h"

int main() {
#if __cpp_lib_ranges >= 202302L
    int array[1];
    [[maybe_unused]] auto empty = std::ranges::empty(array);
#endif
    return TestUtils::done();
}
