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
#if __cpp_lib_memory_resource >= 201603L
    [[maybe_unused]] std::pmr::memory_resource* res = std::pmr::get_default_resource();
#endif
    return TestUtils::done();
}
