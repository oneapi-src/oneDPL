// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <initializer_list>
#include "support/utils.h"

int main() {
    auto init_list = {1, 2, 3};
    [[maybe_unused]] auto it = init_list.begin();
    return TestUtils::done();
}
