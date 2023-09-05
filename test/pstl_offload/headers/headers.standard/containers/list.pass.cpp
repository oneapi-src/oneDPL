// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <list>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::list<int> fl;
    return TestUtils::done();
}
