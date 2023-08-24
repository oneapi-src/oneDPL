// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <csignal>
#include "support/utils.h"

int main() {
    using type = sig_atomic_t;
    return TestUtils::done();
}
