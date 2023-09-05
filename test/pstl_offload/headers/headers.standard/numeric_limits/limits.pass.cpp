// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <limits>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto is_signed = std::numeric_limits<size_t>::is_signed;
    return TestUtils::done();
}
