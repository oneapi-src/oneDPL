// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <thread>
#include "support/utils.h"

int main() {
    std::thread t([](){});
    t.join();
    return TestUtils::done();
}
