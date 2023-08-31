// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DO_STD_INJECTION 1
#include "system_allocations.h"

void perform_system_allocations(system_allocs& na) {
    perform_allocations_impl(na);
}

void perform_system_deallocations(const system_allocs& na) {
    perform_deallocations_impl(na);
}
