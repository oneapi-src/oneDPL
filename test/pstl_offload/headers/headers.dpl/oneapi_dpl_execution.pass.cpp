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
    static_assert(!std::is_same_v<decltype(oneapi::dpl::execution::par), void>);
    return TestUtils::done();
}
