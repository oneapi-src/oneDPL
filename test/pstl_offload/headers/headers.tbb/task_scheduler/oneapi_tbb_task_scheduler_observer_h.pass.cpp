// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/task_scheduler_observer.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(oneapi::tbb::task_scheduler_observer);
    return TestUtils::done();
}
