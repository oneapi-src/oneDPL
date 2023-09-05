// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/parallel_pipeline.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(oneapi::tbb::filter<int, int>);
    return TestUtils::done();
}
