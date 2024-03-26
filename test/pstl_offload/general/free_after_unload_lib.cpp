// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __SYCL_PSTL_OFFLOAD__ 
#error "PSTL offload must be not enabled for this TU"
#endif

#include <stdlib.h>
#include <stdio.h>

#include "free_after_unload_lib.h"

static void *ptr;

void register_mem_to_later_release(void *p)
{
    ptr = p;
}

struct DelayedReleaser
{
    ~DelayedReleaser()
    {
        free(ptr);
    }
};

static DelayedReleaser delayed_releaser;
