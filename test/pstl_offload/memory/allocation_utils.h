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

#if __linux__
extern "C"
{
void *__libc_malloc(std::size_t);
void *__libc_calloc(std::size_t, std::size_t);
void *__libc_memalign(std::size_t, std::size_t);
void *__libc_realloc(void *, std::size_t);
}
#endif // __linux__

#if !__cpp_sized_deallocation
// Intel* oneAPI DPC++/C++ Compiler doesn't set __cpp_sized_deallocation,
// so provide declaration, as we have sized deallocations in the global overload
void operator delete(void* __ptr, std::size_t) noexcept;
void operator delete[](void* __ptr, std::size_t) noexcept;
#endif // __cpp_sized_deallocation

struct allocs {
    void* malloc_ptr;
    void* calloc_ptr;
    void* realloc_ptr;
    void* memalign_ptr;
#if __linux__
    void* posix_memalign_ptr;
    void* aligned_alloc_ptr;

    void* libc_malloc_ptr;
    void* libc_calloc_ptr;
    void* libc_realloc_ptr;
    void* libc_memalign_ptr;
#elif _WIN64
    void* aligned_realloc_ptr;
#endif // __linux__

    void* new_ptr;
    void* arr_new_ptr;
    void* nothrow_new_ptr;
    void* arr_nothrow_new_ptr;
    void* aligned_new_ptr;
    void* aligned_nothrow_new_ptr;
    void* aligned_arr_new_ptr;
    void* aligned_nothrow_arr_new_ptr;

    static constexpr std::size_t alloc_size = 1024;
    static constexpr std::size_t alignment = 16;
};

static allocs perform_allocations_impl() {
    allocs na;
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
    na.libc_memalign_ptr = __libc_memalign(allocs::alignment, allocs::alloc_size);
#elif _WIN64
    na.memalign_ptr = _aligned_malloc(allocs::alloc_size, allocs::alignment);
    na.aligned_realloc_ptr = _aligned_realloc(nullptr, allocs::alloc_size, allocs::alignment);
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
    return na;
}

static void perform_deallocations_impl(const allocs& na) {
#if __linux__
    EXPECT_TRUE(malloc_usable_size(na.malloc_ptr) >= allocs::alloc_size, "Incorrect return value of malloc_usable_size");
#elif _WIN64
    EXPECT_TRUE(_msize(na.malloc_ptr) >= allocs::alloc_size, "Incorrect return value of _msize");
#endif
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
#elif _WIN64
    EXPECT_TRUE(_aligned_msize(na.memalign_ptr, allocs::alignment, /*offset*/0) >= allocs::alloc_size,
                "Invalid size reported by _aligned_msize");
    _aligned_free(na.memalign_ptr);
    _aligned_realloc(na.aligned_realloc_ptr, /*size*/0, /*alignment*/0);
#endif // __linux__

    operator delete(na.new_ptr);
    operator delete[](na.arr_new_ptr);
    operator delete(na.nothrow_new_ptr, allocs::alloc_size);
    operator delete[](na.arr_nothrow_new_ptr, allocs::alloc_size);
    operator delete(na.aligned_new_ptr, std::align_val_t(allocs::alignment));
    operator delete(na.aligned_nothrow_new_ptr, std::align_val_t(allocs::alignment));
    operator delete[](na.aligned_arr_new_ptr, std::align_val_t(allocs::alignment));
    operator delete[](na.aligned_nothrow_arr_new_ptr, std::align_val_t(allocs::alignment));
}

allocs perform_system_allocations();
void perform_system_deallocations(const allocs&);
allocs perform_usm_allocations();
void perform_usm_deallocations(const allocs&);

#endif // _ALLOCATION_UTILS_H
