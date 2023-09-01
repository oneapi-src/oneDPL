// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <syncstream>
#include "support/utils.h"

int main() {
#if __cpp_lib_syncbuf >= 201803L
    [[maybe_unused]] std::size_t r = sizeof(std::syncbuf);
#endif
    return TestUtils::done();
}
