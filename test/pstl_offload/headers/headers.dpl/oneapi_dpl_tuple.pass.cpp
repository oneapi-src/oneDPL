// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/tuple>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto t = oneapi::dpl::make_tuple(1, 2);
    return TestUtils::done();
}
