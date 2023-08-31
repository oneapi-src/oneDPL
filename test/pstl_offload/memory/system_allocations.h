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

// to have native allocation, we must not include possibly overloaded header
#if DO_STD_INJECTION
namespace std {
using size_t = long unsigned int;
enum class align_val_t : std::size_t {};
struct nothrow_t {
    explicit nothrow_t() = default;
};
extern const nothrow_t nothrow;
} // namespace std
#endif // DO_STD_INJECTION

extern "C" {
void *malloc(std::size_t size);
void *calloc (std::size_t, std::size_t);
void *realloc (void *, std::size_t);
void *memalign(std::size_t, std::size_t) noexcept;
int posix_memalign(void**, std::size_t, std::size_t);
void free (void *__ptr);
void *aligned_alloc(std::size_t, std::size_t);
void *__libc_malloc(std::size_t);
void *__libc_calloc(std::size_t, std::size_t);
void *__libc_memalign(std::size_t, std::size_t);
void *__libc_realloc(void *, std::size_t);
} // extern "C"

void* operator new  ( std::size_t count, const std::nothrow_t& tag ) noexcept;
void* operator new[]( std::size_t count, const std::nothrow_t& tag ) noexcept;
void* operator new  ( std::size_t count,
                      std::align_val_t al, const std::nothrow_t& ) noexcept;
void* operator new[]( std::size_t count,
                      std::align_val_t al, const std::nothrow_t& ) noexcept;

struct system_allocs {
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
};

static inline void perform_allocations_impl(system_allocs& na) {
    na.malloc_ptr = malloc(1024);
    na.calloc_ptr = calloc(10, 10);
    na.realloc_ptr = realloc(nullptr, 10);
    na.memalign_ptr = memalign(16, 10);
    posix_memalign(&na.posix_memalign_ptr, 16, 10);
    na.aligned_alloc_ptr = aligned_alloc(16, 10);
    na.libc_malloc_ptr = __libc_malloc(10);
    na.libc_calloc_ptr = __libc_calloc(10, 10);
    na.libc_realloc_ptr = __libc_realloc(nullptr, 10);
    na.libc_memalign_ptr = __libc_memalign(16, 10);

    na.new_ptr = new char;
    na.arr_new_ptr = new char[12];
    na.nothrow_new_ptr = new (std::nothrow) char;
    na.arr_nothrow_new_ptr = new (std::nothrow) char[12];
    na.aligned_new_ptr = new (std::align_val_t(64)) char;
    na.aligned_nothrow_new_ptr = new (std::align_val_t(64), std::nothrow) char;
    na.aligned_arr_new_ptr = new (std::align_val_t(64)) char[12];
    na.aligned_nothrow_arr_new_ptr = new (std::align_val_t(64), std::nothrow) char[12];
}

static inline void perform_deallocations_impl(const system_allocs& na) {
    free(na.malloc_ptr);
    free(na.calloc_ptr);
    (void)realloc(na.realloc_ptr, 0);
    free(na.memalign_ptr);
    free(na.posix_memalign_ptr);
    free(na.aligned_alloc_ptr);
    free(na.libc_malloc_ptr);
    free(na.libc_calloc_ptr);
    free(na.libc_realloc_ptr);
    free(na.libc_memalign_ptr);

    delete na.new_ptr;
    delete[] na.arr_new_ptr;
    delete na.nothrow_new_ptr;
    delete[] na.arr_nothrow_new_ptr;
    operator delete(na.aligned_new_ptr, std::align_val_t(64));
    operator delete(na.aligned_nothrow_new_ptr, std::align_val_t(64));
    operator delete[](na.aligned_arr_new_ptr, std::align_val_t(64));
    operator delete[](na.aligned_nothrow_arr_new_ptr, std::align_val_t(64));
}

void perform_system_allocations(system_allocs&);
void perform_system_deallocations(const system_allocs&);

#endif // _SYSTEM_ALLOCATIONS_H
