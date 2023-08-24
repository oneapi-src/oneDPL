// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <future>
#include "support/utils.h"

int main() {
    auto ft = std::async([]() { return 0; });
    ft.wait();
    return TestUtils::done();
}
