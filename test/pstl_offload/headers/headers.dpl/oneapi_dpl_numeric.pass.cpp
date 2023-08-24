// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/numeric>
#include "support/utils.h"

int main() {
    int array[] = {1, 2, 3};
    [[maybe_unused]] auto sum = oneapi::dpl::reduce(oneapi::dpl::execution::seq, array, array + 3);
    return TestUtils::done();
}
