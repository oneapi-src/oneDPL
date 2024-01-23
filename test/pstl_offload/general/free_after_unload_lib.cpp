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

#include <new>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "support/utils.h"

#include "free_after_unload_lib.h"

static pointers ptrs;

void register_mem_to_later_release(pointers *p)
{
    ptrs = *p;
}

struct DelayedReleaser
{
    ~DelayedReleaser()
    {
        free(ptrs.p1);
        ::operator delete (ptrs.p2, std::align_val_t(8*1024));

        constexpr size_t updated_size = 1024;
        void *p = realloc(ptrs.p3, updated_size);
        EXPECT_TRUE(p, "reallocation failed");
        EXPECT_TRUE(malloc_usable_size(p) >= updated_size, "Invalid size after reallocation");
        free(p);
    }
};

static DelayedReleaser delayed_releaser;
