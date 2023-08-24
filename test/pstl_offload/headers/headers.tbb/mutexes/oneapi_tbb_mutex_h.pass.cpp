// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/mutex.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::mutex m;
    m.lock();
    m.unlock();
    return TestUtils::done();
}
