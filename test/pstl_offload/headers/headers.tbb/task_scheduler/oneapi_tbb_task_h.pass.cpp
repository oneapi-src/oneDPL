// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/task.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto r = oneapi::tbb::task::current_context();
    return TestUtils::done();
}
