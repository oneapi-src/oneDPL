// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <strstream>
#include "support/utils.h"

int main() {
    // TODO: header is deprecated in C++98
    [[maybe_unused]] std::size_t r = sizeof(std::strstream);
    return TestUtils::done();
}
