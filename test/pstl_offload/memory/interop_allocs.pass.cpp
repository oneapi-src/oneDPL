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

#include "allocation_utils.h"

// in this translation unit we have overload to USM
#include <new> // include just to provide the local allocation overload

static sycl::context memory_context = TestUtils::get_pstl_offload_device().get_platform().ext_oneapi_get_default_context();

void check_memory_ownership(const allocs &na, sycl::usm::alloc expected_type) {
        EXPECT_TRUE(sycl::get_pointer_type(na.malloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.calloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.realloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
#if __linux__
        EXPECT_TRUE(sycl::get_pointer_type(na.memalign_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.posix_memalign_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.aligned_alloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.libc_malloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.libc_calloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.libc_realloc_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.libc_memalign_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
#endif // __linux__

        EXPECT_TRUE(sycl::get_pointer_type(na.new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.arr_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.nothrow_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.arr_nothrow_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.aligned_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.aligned_nothrow_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.aligned_arr_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
        EXPECT_TRUE(sycl::get_pointer_type(na.aligned_nothrow_arr_new_ptr, memory_context) == expected_type,
            "Unexpected pointer type");
}

int main() {
    // check the ability to release system memory allocated inside another translation unit without local allocation overload
    {
        allocs na;
        perform_system_allocations(na);
        check_memory_ownership(na, sycl::usm::alloc::unknown);
        perform_deallocations_impl(na);
    }
    // check the ability to release USM memory inside another translation unit without local allocation overload
    {
        allocs na;
        perform_allocations_impl(na);
        check_memory_ownership(na, sycl::usm::alloc::shared);
        perform_system_deallocations(na);
    }

    return TestUtils::done();
}
