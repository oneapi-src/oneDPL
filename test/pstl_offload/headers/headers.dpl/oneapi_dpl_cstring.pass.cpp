// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/cstring>
#include "support/utils.h"

int main() {
    unsigned char a = 1;
    [[maybe_unused]] auto res = oneapi::dpl::memcmp(&a, &a, 1);
    return TestUtils::done();
}
