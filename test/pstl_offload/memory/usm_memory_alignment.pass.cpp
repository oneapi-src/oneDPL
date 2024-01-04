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

static std::size_t get_page_size() {
    static std::size_t page_size = sysconf(_SC_PAGESIZE);
    return page_size;
}

template <typename AllocatingFunction, typename DeallocatingFunction>
void test_alignment_allocation(AllocatingFunction allocate, DeallocatingFunction deallocate) {
    for (std::size_t alignment = 1; alignment < 16*get_page_size(); alignment <<= 1) {
        const std::size_t sizes[] = { 1, 2, 8, 24, alignment / 2, alignment, alignment * 2};

        for (std::size_t size : sizes) {
            void* ptr = allocate(size, alignment);
            EXPECT_TRUE(std::uintptr_t(ptr) % alignment == 0, "The returned pointer is not properly aligned");
            deallocate(ptr, alignment);
        }
    }
}

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

            // check that aligned object might be reallocated
            constexpr std::size_t delta = 13;
            std::size_t new_size = (++cnt % 2) && size > delta ? size - delta : size + delta;
            ptr = realloc(ptr, new_size);
            EXPECT_TRUE(malloc_usable_size(ptr) >= new_size, "Invalid reported size");

            free(ptr);
        }
    }
}

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

    auto delete_deallocate = [](void* ptr, std::size_t alignment) {
        return ::operator delete(ptr, std::align_val_t(alignment));
    };
    auto delete_nothrow_deallocate = [](void* ptr, std::size_t alignment) {
        return ::operator delete(ptr, std::align_val_t(alignment), std::nothrow);
    };

    auto delete_array_deallocate = [](void* ptr, std::size_t alignment) {
        return ::operator delete[](ptr, std::align_val_t(alignment));
    };
    auto delete_array_nothrow_deallocate = [](void* ptr, std::size_t alignment) {
        return ::operator delete[](ptr, std::align_val_t(alignment), std::nothrow);
    };

    test_alignment_allocation(new_allocate, delete_deallocate);
    test_alignment_allocation(new_array_allocate, delete_array_deallocate);

    test_alignment_allocation(new_nothrow_allocate, delete_nothrow_deallocate);
    test_alignment_allocation(new_array_nothrow_allocate, delete_array_nothrow_deallocate);
}

int main() {
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

    test_alignment_allocation(memalign_allocate, free_deallocate);
#if __linux__
    test_alignment_allocation(posix_memalign_allocate, free_deallocate);
    test_alignment_allocation(__libc_memalign_allocate, free_deallocate);
#endif
    test_aligned_alloc_alignment();

    test_new_alignment();

    return TestUtils::done();
}
