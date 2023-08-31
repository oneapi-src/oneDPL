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

#if __cpp_lib_source_location >= 201907L
#include <source_location>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_source_location >= 201907L
    [[maybe_unused]] volatile std::size_t r = sizeof(std::source_location);
#endif
    return TestUtils::done();
}
