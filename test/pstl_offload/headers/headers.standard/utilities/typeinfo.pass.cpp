// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <typeinfo>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto are_equal = (typeid(int) == typeid(float));
    return TestUtils::done();
}
