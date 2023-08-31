// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/rw_mutex.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::rw_mutex m;
    return TestUtils::done();
}
