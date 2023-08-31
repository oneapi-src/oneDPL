// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/async>
#include <sycl/sycl.hpp>
#include "support/utils.h"

int main() {
    sycl::buffer<int> buf{10};
    oneapi::dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,
        oneapi::dpl::begin(buf), oneapi::dpl::end(buf), 0);
    return TestUtils::done();
}
