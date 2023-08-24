// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::set<int> s;
    return TestUtils::done();
}
