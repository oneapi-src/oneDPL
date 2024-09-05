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

#include <new>
#include <cstdlib>
#include <malloc.h> // for malloc_usable_size

#include "support/utils.h"
#if _WIN64
#include <windows.h>

static std::size_t get_page_size_impl() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

#endif

static std::size_t get_page_size() {
#if __linux__
    static std::size_t page_size = sysconf(_SC_PAGESIZE);
#elif _WIN64
    static std::size_t page_size = get_page_size_impl();
#endif
    return page_size;
}

template <typename AllocatingFunction, typename DeallocatingFunction>
void test_alignment_allocation(AllocatingFunction allocate, DeallocatingFunction deallocate) {
    for (std::size_t alignment = 1; alignment < 16 * get_page_size(); alignment <<= 1) {
        const std::size_t sizes[] = { 1, 2, 8, 24, alignment / 2, alignment, alignment * 2};

        for (std::size_t size : sizes) {
            void* ptr = allocate(size, alignment);
            EXPECT_TRUE(ptr != nullptr, "nullptr returned by allocation");
            EXPECT_TRUE(std::uintptr_t(ptr) % alignment == 0, "The returned pointer is not properly aligned");
            deallocate(ptr, size, alignment);
        }
    }
}

#if __linux__
// aligned_alloc requires size to be integral multiple of alignment
// test_alignment_allocation tests different sizes values because other functions
// only requires the alignment to be power of two
void test_aligned_alloc_alignment() {
    int cnt = 0;
    for (std::size_t alignment = 1; alignment < 16*get_page_size(); alignment <<= 1) {
        const std::size_t sizes[] = { alignment, alignment * 2, alignment * 3};

        for (std::size_t size : sizes) {
            void* ptr = aligned_alloc(alignment, size);
            EXPECT_TRUE(ptr, "The returned pointer is nullptr");
            EXPECT_TRUE(std::uintptr_t(ptr) % alignment == 0, "The returned pointer is not properly aligned");
            EXPECT_TRUE(malloc_usable_size(ptr) >= size, "Invalid reported size");

            // check that aligned object might be reallocated
            constexpr std::size_t delta = 13;
            std::size_t new_size = (++cnt % 2) && size > delta ? size - delta : size + delta;
            ptr = realloc(ptr, new_size);
            EXPECT_TRUE(ptr, "The returned pointer is nullptr");
            EXPECT_TRUE(malloc_usable_size(ptr) >= new_size, "Invalid reported size");

            free(ptr);
        }
    }
}
#endif

void test_new_alignment() {
    auto new_allocate = [](std::size_t size, std::size_t alignment) {
        return ::operator new(size, std::align_val_t(alignment));
    };
    auto new_nothrow_allocate = [](std::size_t size, std::size_t alignment) {
        return ::operator new(size, std::align_val_t(alignment), std::nothrow);
    };

    auto new_array_allocate = [](std::size_t size, std::size_t alignment) {
        return ::operator new[](size, std::align_val_t(alignment));
    };
    auto new_array_nothrow_allocate = [](std::size_t size, std::size_t alignment) {
        return ::operator new[](size, std::align_val_t(alignment), std::nothrow);
    };

    auto delete_deallocate = [](void* ptr, std::size_t, std::size_t alignment) {
        return ::operator delete(ptr, std::align_val_t(alignment));
    };
    auto delete_nothrow_deallocate = [](void* ptr, std::size_t, std::size_t alignment) {
        return ::operator delete(ptr, std::align_val_t(alignment), std::nothrow);
    };

    auto delete_array_deallocate = [](void* ptr, std::size_t, std::size_t alignment) {
        return ::operator delete[](ptr, std::align_val_t(alignment));
    };
    auto delete_array_nothrow_deallocate = [](void* ptr, std::size_t, std::size_t alignment) {
        return ::operator delete[](ptr, std::align_val_t(alignment), std::nothrow);
    };

    test_alignment_allocation(new_allocate, delete_deallocate);
    test_alignment_allocation(new_array_allocate, delete_array_deallocate);

    test_alignment_allocation(new_nothrow_allocate, delete_nothrow_deallocate);
    test_alignment_allocation(new_array_nothrow_allocate, delete_array_nothrow_deallocate);
}

// check only presence of the overloads
void test_page_aligned_allocations() {
    void* ptr = valloc(8);
    EXPECT_TRUE(ptr, "The returned pointer is nullptr");
    EXPECT_TRUE(std::uintptr_t(ptr) % get_page_size() == 0, "The returned pointer is not properly aligned");
    free(ptr);
    ptr = __libc_valloc(8);
    EXPECT_TRUE(ptr, "The returned pointer is nullptr");
    EXPECT_TRUE(std::uintptr_t(ptr) % get_page_size() == 0, "The returned pointer is not properly aligned");
    free(ptr);

    ptr = pvalloc(8);
    EXPECT_TRUE(ptr, "The returned pointer is nullptr");
    EXPECT_TRUE(std::uintptr_t(ptr) % get_page_size() == 0, "The returned pointer is not properly aligned");
    free(ptr);
    ptr = __libc_pvalloc(8);
    EXPECT_TRUE(ptr, "The returned pointer is nullptr");
    EXPECT_TRUE(std::uintptr_t(ptr) % get_page_size() == 0, "The returned pointer is not properly aligned");
    free(ptr);
}


int main() {
#if __linux__
    auto aligned_alloc_allocate = [](std::size_t size, std::size_t alignment)
    {
        return aligned_alloc(alignment, size);
    };
    auto memalign_allocate = [](std::size_t size, std::size_t alignment) {
        return memalign(alignment, size);
    };
    auto posix_memalign_allocate = [](std::size_t size, std::size_t alignment) {
        void* ptr = nullptr;
        posix_memalign(&ptr, alignment, size);
        return ptr;
    };
    auto __libc_memalign_allocate = [](std::size_t size, std::size_t alignment) {
        return __libc_memalign(alignment, size);
    };

    auto free_deallocate = [](void* ptr, std::size_t, std::size_t) {
        free(ptr);
    };

    test_alignment_allocation(memalign_allocate, free_deallocate);
    test_alignment_allocation(posix_memalign_allocate, free_deallocate);
    test_alignment_allocation(__libc_memalign_allocate, free_deallocate);
    test_page_aligned_allocations();
    test_aligned_alloc_alignment();
#elif _WIN64
    auto _aligned_malloc_allocate = [](std::size_t size, std::size_t alignment) {
        return _aligned_malloc(size, alignment);
    };
    auto _aligned_free_deallocate = [](void* ptr, std::size_t, std::size_t) {
        _aligned_free(ptr);
    };

    test_alignment_allocation(_aligned_malloc_allocate, _aligned_free_deallocate);

    auto _aligned_realloc_allocate = [](std::size_t size, std::size_t alignment) {
        return _aligned_realloc(nullptr, size, alignment);
    };
    auto _aligned_realloc_existing_allocate = [](std::size_t size, std::size_t alignment) {
        // "It's an error to reallocate memory and change the alignment of a block.",
        // so have to re-use the alignment.
        void* ptr = _aligned_malloc(size, alignment);
        EXPECT_TRUE(ptr, "nullptr returned");
        EXPECT_TRUE(std::uintptr_t(ptr) % alignment == 0, "The returned pointer is not properly aligned");
        // this can be called with zero size, but _aligned_realloc() in zero case became _aligned_free()
        return _aligned_realloc(ptr, size ? size : 1, alignment);
    };
    auto _aligned_realloc_deallocate = [](void* ptr, std::size_t size, std::size_t alignment) {
        EXPECT_TRUE(_aligned_msize(ptr, alignment, 0) >= size, "Invalid size returned by _aligned_msize");
        void * ret = _aligned_realloc(ptr, 0, alignment);
        EXPECT_TRUE(ret == nullptr, "_aligned_realloc(ptr, 0, alignment) must return nullptr");
    };

    test_alignment_allocation(_aligned_realloc_allocate, _aligned_realloc_deallocate);
    test_alignment_allocation(_aligned_realloc_existing_allocate, _aligned_realloc_deallocate);
#endif

    test_new_alignment();

    return TestUtils::done();
}
