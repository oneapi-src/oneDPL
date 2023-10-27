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

// in this translation unit we use USM allocations
#include "interop_allocs_headers.h"
#include "allocation_utils.h"

allocs perform_usm_allocations() {
    return perform_allocations_impl();
}

void perform_usm_deallocations(const allocs& na) {
    perform_deallocations_impl(na);
}
