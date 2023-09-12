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

// in this translation unit we use system allocations while compiling with pstl offload option,
// it should be no overload to USM

#include "allocation_utils.h"

void perform_system_allocations(allocs& na) {
    perform_allocations_impl(na);
}

void perform_system_deallocations(const allocs& na) {
    perform_deallocations_impl(na);
}
