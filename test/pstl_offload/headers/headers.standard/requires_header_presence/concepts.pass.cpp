// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <concepts>
#include "support/utils.h"

int main() {
#if __cpp_lib_concepts >= 202002L
    static_assert(!std::same_as<int, float>);
#endif
    return TestUtils::done();
}
