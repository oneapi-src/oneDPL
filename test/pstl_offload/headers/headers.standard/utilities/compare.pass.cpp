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

#if __cpp_lib_three_way_comparison >= 201907L
#include <compare>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_three_way_comparison >= 201907L
    [[maybe_unused]] auto ordering = std::strong_ordering::equal;
#endif
    return TestUtils::done();
}
