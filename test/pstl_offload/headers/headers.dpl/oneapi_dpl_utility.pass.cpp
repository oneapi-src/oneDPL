// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/utility>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto b = oneapi::dpl::make_pair(1, 2);
    return TestUtils::done();
}
