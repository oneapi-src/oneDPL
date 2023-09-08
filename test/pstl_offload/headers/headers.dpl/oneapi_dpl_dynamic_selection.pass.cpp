// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/dynamic_selection>
#include "support/utils.h"

int main() {
    using namespace oneapi::dpl::experimental;
    [[maybe_unused]] std::size_t r = sizeof(fixed_resource_policy<sycl_backend>);
    return TestUtils::done();
}
