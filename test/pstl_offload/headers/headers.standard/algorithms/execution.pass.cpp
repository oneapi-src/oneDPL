// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <execution>
#include "support/utils.h"

int main() {
    static_assert(std::is_execution_policy_v<std::execution::sequenced_policy>);
    return TestUtils::done();
}
