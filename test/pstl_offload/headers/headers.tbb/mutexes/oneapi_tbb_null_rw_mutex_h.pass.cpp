// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/null_rw_mutex.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::null_rw_mutex m;
    m.lock();
    m.unlock();
    return TestUtils::done();
}
