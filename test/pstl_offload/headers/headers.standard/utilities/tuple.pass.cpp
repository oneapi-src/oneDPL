// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::tuple<int> tpl = {1};
    return TestUtils::done();
}
