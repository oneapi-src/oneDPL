// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ALLOCATION_UTILS_H
#define _ALLOCATION_UTILS_H

#define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#include <new>
#include <cstdlib>
#include <malloc.h>
#include "support/utils.h"
#undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

extern "C"
{
void *__libc_malloc(std::size_t);
void *__libc_calloc(std::size_t, std::size_t);
void *__libc_memalign(std::size_t, std::size_t);
void *__libc_realloc(void *, std::size_t);
}

#if __cpp_sized_deallocation
#error "The test must be updated to cover a compiler with sized decallocation support."
#endif

#if TEST_DELETE_OVERLOAD
// Intel* oneAPI DPC++/C++ Compiler doesn't set __cpp_sized_deallocation,
// so check for sized deallocation only in overloaded delete test case
void operator delete(void* __ptr, std::size_t) noexcept;
void operator delete[](void* __ptr, std::size_t) noexcept;
#endif // TEST_DELETE_OVERLOAD

struct allocs {
    void *malloc_ptr;
    void *calloc_ptr;
    void *realloc_ptr;
#if __linux__
    void *memalign_ptr;
    void *posix_memalign_ptr;
    void *aligned_alloc_ptr;

    void *libc_malloc_ptr;
    void *libc_calloc_ptr;
    void *libc_realloc_ptr;
    void *libc_memalign_ptr;
#endif // __linux__

    void *new_ptr;
    void *arr_new_ptr;
    void *nothrow_new_ptr;
    void *arr_nothrow_new_ptr;
    void *aligned_new_ptr;
    void *aligned_nothrow_new_ptr;
    void *aligned_arr_new_ptr;
    void *aligned_nothrow_arr_new_ptr;

    static const std::size_t alloc_size = 1024;
    static const std::size_t alignment = 16;
    static const std::size_t array_size = 12;
};

static void perform_allocations_impl(allocs& na) {
    na.malloc_ptr = malloc(allocs::alloc_size);
    na.calloc_ptr = calloc(allocs::alloc_size, allocs::alloc_size);
    na.realloc_ptr = realloc(nullptr, allocs::alloc_size);
#if __linux__
    EXPECT_TRUE(malloc_usable_size(na.malloc_ptr) >= allocs::alloc_size, "Invalid object size");
    na.memalign_ptr = memalign(allocs::alignment, allocs::alloc_size);
    posix_memalign(&na.posix_memalign_ptr, allocs::alignment, allocs::alloc_size);
    na.aligned_alloc_ptr = aligned_alloc(allocs::alignment, allocs::alloc_size);
    na.libc_malloc_ptr = __libc_malloc(allocs::alloc_size);
    na.libc_calloc_ptr = __libc_calloc(allocs::alloc_size, allocs::alloc_size);
    na.libc_realloc_ptr = __libc_realloc(nullptr, allocs::alloc_size);
    na.libc_memalign_ptr = __libc_memalign(16, allocs::alloc_size);
#endif // __linux__

    na.new_ptr = ::operator new(allocs::alloc_size);
    na.arr_new_ptr = ::operator new[](allocs::alloc_size);
    na.nothrow_new_ptr = ::operator new(allocs::alloc_size, std::nothrow);
    na.arr_nothrow_new_ptr = ::operator new[](allocs::alloc_size, std::nothrow);
    na.aligned_new_ptr = ::operator new(allocs::alloc_size, std::align_val_t(allocs::alignment));
    na.aligned_nothrow_new_ptr = 
        ::operator new(allocs::alloc_size, std::align_val_t(allocs::alignment), std::nothrow);
    na.aligned_arr_new_ptr = ::operator new[](allocs::alloc_size, std::align_val_t(allocs::alignment));
    na.aligned_nothrow_arr_new_ptr =
        ::operator new[](allocs::alloc_size, std::align_val_t(allocs::alignment), std::nothrow);
}

static void perform_deallocations_impl(const allocs& na) {
    free(na.malloc_ptr);
    free(na.calloc_ptr);
    free(na.realloc_ptr);
#if __linux__
    free(na.memalign_ptr);
    free(na.posix_memalign_ptr);
    free(na.aligned_alloc_ptr);
    free(na.libc_malloc_ptr);
    free(na.libc_calloc_ptr);
    free(na.libc_realloc_ptr);
    free(na.libc_memalign_ptr);
#endif // __linux__

    operator delete(na.new_ptr);
    operator delete[](na.arr_new_ptr);
#if TEST_DELETE_OVERLOAD
    operator delete(na.nothrow_new_ptr, allocs::alloc_size);
    operator delete[](na.arr_nothrow_new_ptr, allocs::array_size*allocs::alloc_size);
#else
    operator delete(na.nothrow_new_ptr);
    operator delete[](na.arr_nothrow_new_ptr);
#endif
    operator delete(na.aligned_new_ptr, std::align_val_t(allocs::alignment));
    operator delete(na.aligned_nothrow_new_ptr, std::align_val_t(allocs::alignment));
    operator delete[](na.aligned_arr_new_ptr, std::align_val_t(allocs::alignment));
    operator delete[](na.aligned_nothrow_arr_new_ptr, std::align_val_t(allocs::alignment));
}

void perform_system_allocations(allocs&);
void perform_system_deallocations(const allocs&);

#endif // _ALLOCATION_UTILS_H
