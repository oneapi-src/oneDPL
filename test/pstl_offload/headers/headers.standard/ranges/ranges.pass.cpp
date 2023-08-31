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

#if __cpp_lib_ranges >= 202302L
#include <ranges>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_ranges >= 202302L
    int array[1];
    [[maybe_unused]] auto empty = std::ranges::empty(array);
#endif
    return TestUtils::done();
}
