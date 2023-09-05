// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <typeindex>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::type_index t1(typeid(int));
    return TestUtils::done();
}
