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

#if __cpp_lib_math_constants >= 201907L
#include <numbers>
#endif

#include "support/utils.h"

int main() {
#if __cpp_lib_math_constants >= 201907L
    [[maybe_unused]] auto phi = std::numbers::phi_v;
#endif
    return TestUtils::done();
}
