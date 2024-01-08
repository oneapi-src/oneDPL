// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/flow_graph.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::flow::graph g;
    return TestUtils::done();
}
