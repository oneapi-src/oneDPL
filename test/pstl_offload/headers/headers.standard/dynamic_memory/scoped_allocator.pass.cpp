// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <scoped_allocator>
#include <memory>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::scoped_allocator_adaptor<std::allocator<int>> scoped;
    return TestUtils::done();
}
