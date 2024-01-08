// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/collaborative_call_once.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(oneapi::tbb::collaborative_once_flag);
    return TestUtils::done();
}
