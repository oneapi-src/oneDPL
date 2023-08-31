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

#if __cpp_lib_memory_resource >= 201603L
#include <memory_resource>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_memory_resource >= 201603L
    [[maybe_unused]] std::pmr::memory_resource* res = std::pmr::get_default_resource();
#endif
    return TestUtils::done();
}
