// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_HEADER_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_HEADER_H

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>

#if __linux__
#include <dlfcn.h>
#include <unistd.h>
#endif // __linux__

namespace oneapi {
namespace dpl {
namespace pstl_offload {

static constexpr std::size_t UNIQ_TYPE_CONST = 0x23499abc405a9bccLLU;

struct BlockHeader {
    BlockHeader(std::size_t u_const, void* ptr, std::size_t size, sycl::device* d)
        : uniq_const(u_const), original_pointer(ptr), requested_number_of_bytes(size), device(d)
    {
        assert(original_pointer);
    }

    std::size_t uniq_const;
    void* original_pointer;
    std::size_t requested_number_of_bytes;
    sycl::device* device;
};

static constexpr std::size_t HEADER_OFFSET = 32;

static_assert(sizeof(BlockHeader) <= HEADER_OFFSET);

#if __linux__

static long get_memory_page_size()
{
    static long memory_page_size = sysconf(_SC_PAGESIZE);
    return memory_page_size;
}

static void* allocate_from_device(sycl::device* device, std::size_t size, std::size_t alignment) {
    std::size_t base_offset = std::max(alignment, HEADER_OFFSET);
    // Unsupported alignment - impossible to guarantee that the returned pointer and memory header
    // would be on the same memory page if the alignment for more than a memory page is requested
    if (alignment >= get_memory_page_size()) {
        return nullptr;
    }

    // Memory block allocated with sycl::aligned_alloc_shared should be aligned to at least HEADER_OFFSET * 2
    // to guarantee that header and header + HEADER_OFFSET (user pointer) would be placed in one memory page
    std::size_t usm_alignment = base_offset << 1;
    // Required number of bytes to store memory header and preserve alignment on returned pointer
    // usm_alignment bytes are reserved to store memory header
    std::size_t usm_size = size + base_offset;

    sycl::context context = device->get_platform().ext_oneapi_get_default_context();
    void* ptr = sycl::aligned_alloc_shared(usm_alignment, usm_size, *device, context);

    if (!ptr)
        return nullptr;

    void* res = reinterpret_cast<char*>(ptr) + base_offset;
    BlockHeader* header = reinterpret_cast<BlockHeader*>(res) - 1;
    ::new(header) BlockHeader(UNIQ_TYPE_CONST, ptr, size, device);
    return res;
}

static bool same_memory_page(void* ptr1, void* ptr2) {
    std::uintptr_t page_size = get_memory_page_size();
    std::uintptr_t page_mask = ~(page_size - 1);
    std::uintptr_t ptr1_page_begin = std::uintptr_t(ptr1) & page_mask;
    return std::uintptr_t(ptr2) >= ptr1_page_begin;
}

static auto get_original_realloc() {
    using realloc_type = void* (*)(void*, std::size_t);

    static realloc_type orig_realloc = realloc_type(dlsym(RTLD_NEXT, "realloc"));
    return orig_realloc;
}

static void* internal_realloc(void* user_ptr, std::size_t new_size) {
    if (!user_ptr) {
        return malloc(new_size);
    }

    BlockHeader* header = reinterpret_cast<BlockHeader*>(user_ptr) - 1;

    if (same_memory_page(user_ptr, header) && header->uniq_const == UNIQ_TYPE_CONST) {
        if (header->requested_number_of_bytes < new_size) {
            assert(header->device);
            void* new_ptr = allocate_from_device(header->device, new_size, alignof(std::max_align_t));

            if (!new_ptr) {
                errno = ENOMEM;
                return nullptr;
            }

            std::memcpy(new_ptr, user_ptr, header->requested_number_of_bytes);

            // Free previously allocated memory
            void* original_pointer = header->original_pointer;
            sycl::context context = header->device->get_platform().ext_oneapi_get_default_context();
            header->~BlockHeader();
            sycl::free(original_pointer, context);
            return new_ptr;
        }
        return user_ptr;
    }

    return get_original_realloc()(user_ptr, new_size);
}

#endif

} // namespace pstl_offload
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_HEADER_H
