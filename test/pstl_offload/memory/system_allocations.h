// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _SYSTEM_ALLOCATIONS_H
#define _SYSTEM_ALLOCATIONS_H

#define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#include <new>
#define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#include <cstdlib>
#include <malloc.h>

static void check_true(bool expected, bool condition, const char* file, int line, const char* message) {
    if (condition != expected) {
        fprintf(stderr, "error at %s:%d - %s\n", file, line, message);
        exit(1);
    }
}

#ifndef EXPECT_TRUE
#define EXPECT_TRUE(condition, message) check_true(true, condition, __FILE__, __LINE__, message)
#endif

extern "C"
{
void *__libc_malloc(std::size_t);
void *__libc_calloc(std::size_t, std::size_t);
void *__libc_memalign(std::size_t, std::size_t);
void *__libc_realloc(void *, std::size_t);
}

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

    char *new_ptr;
    char *arr_new_ptr;
    char *nothrow_new_ptr;
    char *arr_nothrow_new_ptr;
    char *aligned_new_ptr;
    char *aligned_nothrow_new_ptr;
    char *aligned_arr_new_ptr;
    char *aligned_nothrow_arr_new_ptr;

    static const std::size_t malloc_size = 1024;
    static const std::size_t alignment = 16;
    static const std::size_t array_size = 12;
};

static void perform_allocations_impl(allocs& na) {
    na.malloc_ptr = malloc(allocs::malloc_size);
    na.calloc_ptr = calloc(allocs::malloc_size, allocs::malloc_size);
    na.realloc_ptr = realloc(nullptr, allocs::malloc_size);
#if __linux__
    EXPECT_TRUE(malloc_usable_size(na.malloc_ptr) >= allocs::malloc_size, "Invalid object size");
    na.memalign_ptr = memalign(allocs::alignment, allocs::malloc_size);
    posix_memalign(&na.posix_memalign_ptr, allocs::alignment, allocs::malloc_size);
    na.aligned_alloc_ptr = aligned_alloc(allocs::alignment, allocs::malloc_size);
    na.libc_malloc_ptr = __libc_malloc(allocs::malloc_size);
    na.libc_calloc_ptr = __libc_calloc(allocs::malloc_size, allocs::malloc_size);
    na.libc_realloc_ptr = __libc_realloc(nullptr, allocs::malloc_size);
    na.libc_memalign_ptr = __libc_memalign(16, allocs::malloc_size);
#endif

    na.new_ptr = new char;
    na.arr_new_ptr = new char[allocs::array_size];
    na.nothrow_new_ptr = new (std::nothrow) char;
    na.arr_nothrow_new_ptr = new (std::nothrow) char[allocs::array_size];
    na.aligned_new_ptr = new (std::align_val_t(allocs::alignment)) char;
    na.aligned_nothrow_new_ptr = new (std::align_val_t(allocs::alignment), std::nothrow) char;
    na.aligned_arr_new_ptr = new (std::align_val_t(allocs::alignment)) char[allocs::array_size];
    na.aligned_nothrow_arr_new_ptr =
        new (std::align_val_t(allocs::alignment), std::nothrow) char[allocs::array_size];
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
#endif

    delete na.new_ptr;
    delete[] na.arr_new_ptr;
    delete na.nothrow_new_ptr;
    delete[] na.arr_nothrow_new_ptr;
    operator delete(na.aligned_new_ptr, std::align_val_t(allocs::alignment));
    operator delete(na.aligned_nothrow_new_ptr, std::align_val_t(allocs::alignment));
    operator delete[](na.aligned_arr_new_ptr, std::align_val_t(allocs::alignment));
    operator delete[](na.aligned_nothrow_arr_new_ptr, std::align_val_t(allocs::alignment));
}

void perform_system_allocations(allocs&);
void perform_system_deallocations(const allocs&);

#endif // _SYSTEM_ALLOCATIONS_H
