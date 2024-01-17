// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if !__SYCL_PSTL_OFFLOAD__
#error "PSTL offload compiler mode should be enabled to run this test"
#endif

#include <stdlib.h>
#include <stdio.h>
#include "sycl/sycl.hpp"

#include "support/utils.h"

#include "free_after_unload_lib.h"

// Test possibility to release memory when allocating TU is gone, and so static dtors in it
// have been executed.

int main()
{
    sycl::context memory_context = TestUtils::get_pstl_offload_device().get_platform().ext_oneapi_get_default_context();

    void* ptr = malloc(8);
    EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with memalign");
    register_mem_to_later_release(ptr);

    return TestUtils::done();
}
