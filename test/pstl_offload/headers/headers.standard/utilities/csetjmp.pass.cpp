// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <csetjmp>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(std::jmp_buf);
    return TestUtils::done();
}
