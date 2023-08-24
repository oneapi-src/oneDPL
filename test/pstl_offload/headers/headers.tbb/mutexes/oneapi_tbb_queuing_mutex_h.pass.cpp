// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/queuing_mutex.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::queuing_mutex m;
    oneapi::tbb::queuing_mutex::scoped_lock l(m);
    return TestUtils::done();
}
