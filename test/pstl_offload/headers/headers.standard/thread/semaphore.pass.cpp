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

#if __cpp_lib_semaphore >= 201907L
#include <semaphore>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_semaphore >= 201907L
    [[maybe_unused]] auto max = std::binary_semaphore::max();
#endif
    return TestUtils::done();
}
