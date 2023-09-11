// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __SYCL_PSTL_OFFLOAD__ == 1
#elif __SYCL_PSTL_OFFLOAD__ == 2
    #undef __SYCL_PSTL_OFFLOAD__
    #define __SYCL_PSTL_OFFLOAD__ 3
#elif __SYCL_PSTL_OFFLOAD__ == 3
    #undef __SYCL_PSTL_OFFLOAD__
    #define __SYCL_PSTL_OFFLOAD__ 2
#else
#error "PSTL offload is not enabled or the selected value is unsupported"
#endif

#include <algorithm>
#include "support/utils.h"

// The test assumes that ONEAPI_DEVICE_SELECTOR is set according to __SYCL_PSTL_OFFLOAD__
// and check that changed __SYCL_PSTL_OFFLOAD__ leads to an exception exactly from parallel algorithm.
int main() {
#if __SYCL_PSTL_OFFLOAD__ == 1
    // skip test when value of _ONEDPL_PSTL_OFFLOAD is default
    // as ONEAPI_DEVICE_SELECTOR allows any offload policy in this case
#else
    size_t num = 10LLU*1024*1024;
    int* ptr = new int[num];
    try {
        std::fill(std::execution::par_unseq, ptr, ptr + num, 7);
        EXPECT_TRUE(false, "Device must be unavailable, expecting exception from std::fill().");
    } catch (const sycl::exception &) {
    }
    delete []ptr;
#endif

    return TestUtils::done();
}
