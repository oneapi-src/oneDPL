// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stop_token>
#include "support/utils.h"

int main() {
#if __cpp_lib_jthread >= 201911L
    [[maybe_unused]] std::size_t r = sizeof(std::stop_token);
#endif
    return TestUtils::done();
}
