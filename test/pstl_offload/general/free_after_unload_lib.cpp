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

void register_mem_to_later_release(pointers* p)
{
    ptrs = *p;
}

struct DelayedReleaser
{
    ~DelayedReleaser()
    {
        free(ptrs.malloc_allocated);
        ::operator delete (ptrs.aligned_new_allocated, std::align_val_t(8 * 1024));

        constexpr std::size_t updated_size = 1024;
#if __linux__
        void* p = realloc(ptrs.aligned_alloc_allocated, updated_size);
        EXPECT_TRUE(p, "reallocation failed");
        EXPECT_TRUE(malloc_usable_size(p) >= updated_size, "Invalid size after reallocation");
        free(p);
#elif _WIN64
        constexpr std::size_t updated_alignment = 64 * 1024;
        void* p = _aligned_realloc(ptrs.aligned_alloc_allocated, updated_size, updated_alignment);
        EXPECT_TRUE(p, "_aligned_realloc failed");
        EXPECT_TRUE(_aligned_msize(p, updated_alignment, 0) >= updated_size, "Invalid size after reallocation");
        _aligned_free(p);
#endif
    }
};

static DelayedReleaser delayed_releaser;
