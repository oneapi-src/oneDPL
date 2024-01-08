// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/async>
#include "support/utils.h"

int main() {
    int array[] = {1, 2, 3};
    using return_type = decltype(oneapi::dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,
        array, array + 3, 0));
    return TestUtils::done();
}
