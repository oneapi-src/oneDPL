// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/parallel_sort.h>
#include "support/utils.h"

int main() {
    int array[3];
    oneapi::tbb::parallel_sort(array, array + 3);
    return TestUtils::done();
}
