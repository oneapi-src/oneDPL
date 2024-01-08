// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/enumerable_thread_specific.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::enumerable_thread_specific<int> ets;
    return TestUtils::done();
}
