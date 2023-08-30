// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_PSTL_OFFLOAD__
#error "-fsycl-pstl-offload option should be passed to the compiler to run this test"
#endif

#include <new>
#include <cstdlib>
#include <limits>

#include "sycl/sycl.hpp"

#include "support/utils.h"

#if __linux__
#include <malloc.h>
#endif

sycl::context memory_context = TestUtils::get_pstl_offload_device().get_platform().ext_oneapi_get_default_context();

template <typename... NewArgs>
void test_new_basic(std::size_t count, NewArgs... new_args) {
    void* ptr = ::operator new(count, new_args...);
    void* ptr_array = ::operator new[](count, new_args...);

    EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with new");
    EXPECT_TRUE(sycl::get_pointer_type(ptr_array, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with new[]");

    ::operator delete(ptr, new_args...);
    ::operator delete[](ptr_array, new_args...);
}

int main() {
    const std::size_t num = 4;
    const std::size_t size = sizeof(int) * num;
    const std::size_t alignment = 8;
    const char* test_string = "teststring";
    EXPECT_TRUE(std::strlen(test_string) < size, "Incorrect test setup");
    {
        void* ptr = aligned_alloc(alignment, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with aligned_alloc");
        free(ptr);
    }
    {
        void* ptr = calloc(/*count = */num, sizeof(int));
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with calloc");
        for (std::size_t i = 0; i < num; ++i) {
            EXPECT_TRUE(*(reinterpret_cast<int*>(ptr) + i) == 0, "Memory was not filled with zeros by calloc");
        }
        free(ptr);

        // Overflow test
        ptr = calloc(std::numeric_limits<std::size_t>::max(), sizeof(int));
        EXPECT_TRUE(ptr == nullptr, "Overflow was not handled by calloc");
    }
    {
        void* ptr = malloc(size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with malloc");
        free(ptr);
    }
    {
        void* ptr = nullptr;
        ptr = realloc(ptr, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with realloc");
        std::strcpy(reinterpret_cast<char*>(ptr), test_string);

        ptr = realloc(ptr, size * 2);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating more memory with realloc");
        EXPECT_TRUE(std::strcmp(reinterpret_cast<const char*>(ptr), test_string) == 0, "Memory was not copied into new memory while doing realloc");

        ptr = realloc(ptr, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating less memory with realloc");
        EXPECT_TRUE(std::strcmp(reinterpret_cast<const char*>(ptr), test_string) == 0, "Memory was not copied into new memory while doing realloc");
        free(ptr);
    }
#if __linux__
    {
        void* ptr = nullptr;
        ptr = __libc_realloc(ptr, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with __libc_realloc");
        std::strcpy(reinterpret_cast<char*>(ptr), test_string);

        ptr = __libc_realloc(ptr, size * 2);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating more memory with __libc_realloc");
        EXPECT_TRUE(std::strcmp(reinterpret_cast<const char*>(ptr), test_string) == 0, "Memory was not copied into new memory while doing __libc_realloc");

        ptr = __libc_realloc(ptr, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating less memory with __libc_realloc");
        EXPECT_TRUE(std::strcmp(reinterpret_cast<const char*>(ptr), test_string) == 0, "Memory was not copied into new memory while doing __libc_realloc");
        free(ptr);
    }
    {
        void* ptr = memalign(alignment, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with memalign");
        free(ptr);
    }
    {
        void* ptr = nullptr;
        int err = posix_memalign(&ptr, alignment, size);
        EXPECT_TRUE(err == 0, "Unsuccessful posix_memalign");
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with posix_memalign");
        free(ptr);
    }
    {
        void* ptr = __libc_calloc(/*count = */num, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with __libc_calloc");
        for (std::size_t i = 0; i < num; ++i) {
            EXPECT_TRUE(*(reinterpret_cast<int*>(ptr) + i) == 0, "Memory was not filled with zeros by __libc_calloc");
        }
        free(ptr);

        // Overflow test
        ptr = __libc_calloc(std::numeric_limits<std::size_t>::max(), sizeof(int));
        EXPECT_TRUE(ptr == nullptr, "Overflow was not handled by __libc_calloc");
    }
    {
        void* ptr = __libc_malloc(size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with __libc_malloc");
        free(ptr);
    }
    {
        void* ptr = __libc_memalign(alignment, size);
        EXPECT_TRUE(sycl::get_pointer_type(ptr, memory_context) == sycl::usm::alloc::shared, "Wrong pointer type while allocating with __libc_memalign");
        free(ptr);
    }
    {
        void* ptr = malloc(size);
        EXPECT_TRUE(malloc_usable_size(ptr) >= size, "Incorrect return value of malloc_usable_size");
        free(ptr);
    }
#endif // __linux__

    test_new_basic(size);
    test_new_basic(size, std::align_val_t(alignment));
    test_new_basic(size, std::nothrow);
    test_new_basic(size, std::align_val_t(alignment), std::nothrow);

    // Overflow test, replaced malloc always allocates more bytes than requested
    {
        void* ptr = malloc(std::numeric_limits<std::size_t>::max());
        EXPECT_TRUE(ptr == nullptr, "Overflow in malloc was not handled");
        EXPECT_TRUE(errno == ENOMEM, "Incorrect errno");
    }

    return TestUtils::done();
}
