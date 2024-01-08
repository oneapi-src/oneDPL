// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include "support/utils.h"

int main() {
    using type = decltype(std::strcpy(nullptr, nullptr));
    return TestUtils::done();
}
