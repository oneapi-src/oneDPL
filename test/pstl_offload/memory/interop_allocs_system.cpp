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

// in this translation unit (TU) we use system allocations while compiling with pstl offload option,
// it should be no redirection to USM, that is done in another TU

#define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#include "interop_allocs_headers.h"
#undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

#include "allocation_utils.h"

allocs perform_system_allocations() {
    return perform_allocations_impl();
}

void perform_system_deallocations(const allocs& na) {
    perform_deallocations_impl(na);
}
