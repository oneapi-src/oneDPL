// -*- C++ -*-
//===-- oneapi_dpl_algorithm.pass.cpp ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_PSTL_OFFLOAD__
#error "__SYCL_PSTL_OFFLOAD__ macro should be defined to run this test"
#endif

#include <new>
#include <cstdlib>


#include "support/utils.h"

template <typename AllocatingFunction, typename DeallocatingFunction>
void test_alignment_allocation(AllocatingFunction allocate, DeallocatingFunction deallocate) {
    std::size_t page_size = sysconf(_SC_PAGESIZE);

    for (std::size_t alignment = 1; alignment < page_size; alignment = alignment << 1) {
       void* ptr = allocate(/*size = */alignment, /*alignment = */alignment);
       EXPECT_TRUE(std::uintptr_t(ptr) % alignment == 0, "The returned pointer is not properly aligned");
       deallocate(ptr, alignment);
    }
}

template <typename... NothrowArg>
void test_new_alignment_basic(NothrowArg... nothrow_arg) {
    EXPECT_TRUE(sizeof...(NothrowArg) == 0 || sizeof...(NothrowArg) == 1, "Incorrect test setup");
    auto new_allocate = [nothrow_arg...](std::size_t size, std::size_t alignment) {
        return ::operator new(size, std::align_val_t(alignment), nothrow_arg...);
    };
    auto new_array_allocate = [nothrow_arg...](std::size_t size, std::size_t alignment) {
        return ::operator new[](size, std::align_val_t(alignment), nothrow_arg...);
    };
    auto delete_deallocate = [nothrow_arg...](void* ptr, std::size_t alignment) {
        return ::operator delete(ptr, std::align_val_t(alignment));
    };
    auto delete_array_deallocate = [nothrow_arg...](void* ptr, std::size_t alignment) {
        return ::operator delete[](ptr, std::align_val_t(alignment));
    };

    test_alignment_allocation(new_allocate, delete_deallocate);
    test_alignment_allocation(new_array_allocate, delete_array_deallocate);
}

int main() {
    auto aligned_alloc_allocate = [](std::size_t size, std::size_t alignment) {
        return aligned_alloc(alignment, size);
    };
    auto memalign_allocate = [](std::size_t size, std::size_t alignment) {
        return memalign(alignment, size);
    };
#if __linux__
    auto posix_memalign_allocate = [](std::size_t size, std::size_t alignment) {
        void* ptr = nullptr;
        posix_memalign(&ptr, alignment, size);
        return ptr;
    };
    auto __libc_memalign_allocate = [](std::size_t size, std::size_t alignment) {
        return __libc_memalign(alignment, size);
    };
#endif

    auto free_deallocate = [](void* ptr, std::size_t) {
        free(ptr);
    };

    test_alignment_allocation(aligned_alloc_allocate, free_deallocate);
    test_alignment_allocation(memalign_allocate, free_deallocate);
#if __linux__
    test_alignment_allocation(posix_memalign_allocate, free_deallocate);
    test_alignment_allocation(__libc_memalign_allocate, free_deallocate);
#endif

    test_new_alignment_basic();
    test_new_alignment_basic(std::nothrow);

    return TestUtils::done();
}
