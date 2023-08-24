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
    oneapi::tbb::rw_mutex m;
    oneapi::tbb::rw_mutex::scoped_lock l(m);
    return TestUtils::done();
}
