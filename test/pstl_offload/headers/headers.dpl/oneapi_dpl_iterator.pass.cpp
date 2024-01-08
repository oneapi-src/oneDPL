// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/iterator>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto index = *oneapi::dpl::counting_iterator<std::size_t>(0);
    return TestUtils::done();
}
