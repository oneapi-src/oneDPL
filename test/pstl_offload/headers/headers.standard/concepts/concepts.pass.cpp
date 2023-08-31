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

#if __cpp_lib_concepts >= 202002L
#include <concepts>
#endif

#include "support/utils.h"

int main() {
#if __cpp_lib_concepts >= 202002L
    [[maybe_unused]] bool b = std::same_as<int, float>;
#endif
    return TestUtils::done();
}
