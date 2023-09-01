// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include "support/utils.h"

int main() {
    static_assert(oneapi::dpl::execution::is_execution_policy_v<oneapi::dpl::execution::sequenced_policy>);
    return TestUtils::done();
}
