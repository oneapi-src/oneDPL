// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <barrier>
#include "support/utils.h"

int main() {
#if __cpp_lib_barrier >= 202302L
    [[maybe_unused]] auto m = std::barrier<>::max();
#endif
    return TestUtils::done();
}
