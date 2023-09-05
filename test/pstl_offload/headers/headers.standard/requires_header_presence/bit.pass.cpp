// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <bit>
#include "support/utils.h"

int main() {
#if __cpp_lib_endian >= 201907L
    [[maybe_unused]] auto endian = std::endian::native;
#endif
    return TestUtils::done();
}
