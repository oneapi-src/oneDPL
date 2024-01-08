// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <system_error>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::system_error err{std::error_code{}};
    return TestUtils::done();
}
