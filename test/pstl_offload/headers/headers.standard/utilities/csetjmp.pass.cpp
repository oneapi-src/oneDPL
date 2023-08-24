// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <csetjmp>
#include "support/utils.h"

int main() {
    using type = jmp_buf;
    return TestUtils::done();
}
