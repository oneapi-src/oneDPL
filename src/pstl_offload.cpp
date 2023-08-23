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

#include <new>
#include <cassert>
#include <cstdint>
#include <sycl/sycl.hpp>

#include <pstl_offload/internal/usm_memory_common.h>

#if __linux__

#include <dlfcn.h>
#include <string.h>
#include <unistd.h>

namespace oneapi {
namespace dpl {
namespace pstl_offload {

auto get_original_free() {
    using free_type = void (*)(void*);

    static free_type orig_free = free_type(dlsym(RTLD_NEXT, "free"));
    return orig_free;
}

auto get_original_msize() {
    using msize_type = std::size_t (*)(void*);

    static msize_type orig_msize = msize_type(dlsym(RTLD_NEXT, "malloc_usable_size"));
    return orig_msize;
}

void internal_free(void* user_ptr) {
    if (!user_ptr)
        return;

    BlockHeader* header = reinterpret_cast<BlockHeader*>(user_ptr) - 1;

    if (same_memory_page(user_ptr, header) && header->uniq_const == UNIQ_TYPE_CONST) {
        // Only USM pointers has headers
        assert(header->device);
        sycl::context context = header->device->get_platform().ext_oneapi_get_default_context();
        void* original_pointer = header->original_pointer;
        header->~BlockHeader();
        sycl::free(original_pointer, context);
    } else {
        // A regular pointer without a header
        get_original_free()(user_ptr);
    }
}

std::size_t internal_msize(void* user_ptr) {
    if (!user_ptr)
        return 0;

    BlockHeader* header = reinterpret_cast<BlockHeader*>(user_ptr) - 1;

    if (same_memory_page(user_ptr, header) && header->uniq_const == UNIQ_TYPE_CONST) {
        return header->requested_number_of_bytes;
    }
    return get_original_msize()(user_ptr);
}

} // namespace pstl_offload
} // namespace dpl
} // namespace oneapi

extern "C" {

void free(void* ptr) {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void __libc_free(void *ptr) {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void* realloc(void* ptr, std::size_t new_size) {
    return oneapi::dpl::pstl_offload::internal_realloc(ptr, new_size);
}

void* __libc_realloc(void* ptr, std::size_t new_size) {
    return oneapi::dpl::pstl_offload::internal_realloc(ptr, new_size);
}

std::size_t malloc_usable_size(void* ptr) noexcept {
    return oneapi::dpl::pstl_offload::internal_msize(ptr);
}

} // extern "C"

void operator delete(void* ptr) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete[](void* ptr) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete(void* ptr, std::align_val_t) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
    oneapi::dpl::pstl_offload::internal_free(ptr);
}

#endif // __linux__
