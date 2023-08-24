// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory_resource>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::pmr::memory_resource* res = std::pmr::get_default_resource();
    return TestUtils::done();
}
