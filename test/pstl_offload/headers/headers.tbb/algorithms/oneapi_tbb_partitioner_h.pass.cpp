// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/partitioner.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::auto_partitioner ap;
    return TestUtils::done();
}
