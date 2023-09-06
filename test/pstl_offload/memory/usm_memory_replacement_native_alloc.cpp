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

#define DO_STD_INJECTION 1
#include "system_allocations.h"

void perform_system_allocations(system_allocs& na) {
    perform_allocations_impl(na);
}

void perform_system_deallocations(const system_allocs& na) {
    perform_deallocations_impl(na);
}
