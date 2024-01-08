// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/tick_count.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto tc = oneapi::tbb::tick_count::now();
    return TestUtils::done();
}
