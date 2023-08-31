// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/ranges>
#include <sycl/sycl.hpp>
#include "support/utils.h"

int main() {
    sycl::buffer<int> buf(10);
    oneapi::dpl::experimental::ranges::sort(oneapi::dpl::execution::dpcpp_default, buf);
    return TestUtils::done();
}
