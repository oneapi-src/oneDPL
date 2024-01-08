// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <new>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(std::new_handler);
    return TestUtils::done();
}
