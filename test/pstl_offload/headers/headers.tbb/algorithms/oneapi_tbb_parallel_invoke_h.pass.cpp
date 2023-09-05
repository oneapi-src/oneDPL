// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/parallel_invoke.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::parallel_invoke([](){}, [](){});
    return TestUtils::done();
}
