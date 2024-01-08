// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/type_traits>
#include "support/utils.h"

int main() {
    static_assert(oneapi::dpl::is_same_v<int, int>);
    return TestUtils::done();
}
