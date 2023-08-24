// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/task_group.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::task_group tg;
    tg.run([](){});
    tg.wait();
    return TestUtils::done();
}
