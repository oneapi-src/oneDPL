// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <locale>
#include "support/utils.h"

int main() {
    [[maybe_unused]] volatile std::size_t r = sizeof(std::locale);
    return TestUtils::done();
}
