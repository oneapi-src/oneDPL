// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/optional>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto o = oneapi::dpl::make_optional(0);
    return TestUtils::done();
}
