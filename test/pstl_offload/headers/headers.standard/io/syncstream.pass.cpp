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

#if __cpp_lib_syncbuf >= 201803L
#include <syncstream>
#endif
#include "support/utils.h"

int main() {
#if __cpp_lib_syncbuf >= 201803L
    [[maybe_unused]] volatile std::size_t r = sizeof(std::syncbuf);
#endif
    return TestUtils::done();
}
