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

#if __cpp_lib_barrier >= 202302L
#include <barrier>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_barrier >= 202302L
    [[maybe_unused]] auto m = std::barrier<>::max();
#endif
    return TestUtils::done();
}
