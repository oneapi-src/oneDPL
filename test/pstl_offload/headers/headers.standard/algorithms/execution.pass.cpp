// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <execution>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto& policy = std::execution::par;
}
