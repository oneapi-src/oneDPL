// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#include <version>
#undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

#if __cpp_lib_latch >= 201907L
#include <latch>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_latch >= 201907L
    [[maybe_unused]] std::latch l(1);
#endif
    //TODO: add latch sample
    return TestUtils::done();
}
