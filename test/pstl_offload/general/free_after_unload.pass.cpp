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

static void do_allocation();

// Check that it's possible to allocate memory after dtor of __offload_policy_holder executed.
// For that, the declaration must be done before the header's inclusion.
struct CallAllocInDtor
{
    ~CallAllocInDtor()
    {
        do_allocation();
    }
};

static CallAllocInDtor call_alloc_in_dtor;

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "sycl/sycl.hpp"

#include "support/utils.h"

#include "free_after_unload_lib.h"

// Test possibility to release memory when allocating TU is gone, and so static dtors in it
// have been executed.

int main()
{
    sycl::context memory_context = TestUtils::get_pstl_offload_device().get_platform().ext_oneapi_get_default_context();

    void* ptr = malloc(8);
    EXPECT_TRUE(ptr, "Can't get memory while allocating with overloaded malloc");
    EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with overloaded malloc");
    register_mem_to_later_release(ptr);

    return TestUtils::done();
}


void do_allocation()
{
    sycl::context memory_context = TestUtils::get_pstl_offload_device().get_platform().ext_oneapi_get_default_context();

    constexpr std::size_t size = 1024;
    void *ptr = malloc(size);
    EXPECT_TRUE(ptr, "Can't get memory while allocating in dtor");
    EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::unknown, "Wrong pointer type while allocating in dtor");
    memset(ptr, 1, size);
    void *ptr1 = realloc(ptr, 2*size);
    for (std::size_t i = 0; i < size; i++)
        EXPECT_TRUE(static_cast<char*>(ptr1)[i] == 1, "Data broken after realloc in dtor");
    EXPECT_TRUE(malloc_usable_size(ptr1) >= 2*size, "Invalid size after realloc in dtor");
    free(ptr);
}
