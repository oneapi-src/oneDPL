// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <new>
#include <cassert>
#include <cstdint>
#include <sycl/sycl.hpp>

#include <pstl_offload/internal/usm_memory_replacement_common.h>

#if __linux__

#define _PSTL_OFFLOAD_EXPORT __attribute__((visibility("default")))

#include <dlfcn.h>
#include <string.h>
#include <unistd.h>

namespace __pstl_offload {

auto __get_original_free() {
    using _free_type = void (*)(void*);

    static _free_type __orig_free = _free_type(dlsym(RTLD_NEXT, "free"));
    return __orig_free;
}

auto __get_original_msize() {
    using _msize_type = std::size_t (*)(void*);

    static _msize_type __orig_msize = _msize_type(dlsym(RTLD_NEXT, "malloc_usable_size"));
    return __orig_msize;
}

void __internal_free(void* __user_ptr) {
    if (!__user_ptr)
        return;

    __block_header* __header = reinterpret_cast<__block_header*>(__user_ptr) - 1;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const) {
        // Only USM pointers has headers
        assert(__header->_M_device);
        sycl::context __context = __header->_M_device->get_platform().ext_oneapi_get_default_context();
        __header->_M_uniq_const = 0;
        sycl::free(__header->_M_original_pointer, __context);
    } else {
        // A regular pointer without a header
        __get_original_free()(__user_ptr);
    }
}

std::size_t __internal_msize(void* __user_ptr) {
    if (!__user_ptr)
        return 0;

    __block_header* __header = reinterpret_cast<__block_header*>(__user_ptr) - 1;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const) {
        return __header->_M_requested_number_of_bytes;
    }
    return __get_original_msize()(__user_ptr);
}

} // namespace __pstl_offload

extern "C" {

_PSTL_OFFLOAD_EXPORT void free(void* __ptr) {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void __libc_free(void *__ptr) {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void* realloc(void* __ptr, std::size_t __new_size) {
    return ::__pstl_offload::__internal_realloc(__ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT void* __libc_realloc(void* __ptr, std::size_t __new_size) {
    return ::__pstl_offload::__internal_realloc(__ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT std::size_t malloc_usable_size(void* __ptr) noexcept {
    return ::__pstl_offload::__internal_msize(__ptr);
}

} // extern "C"

_PSTL_OFFLOAD_EXPORT void operator delete(void* __ptr) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete[](void* __ptr) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete(void* __ptr, std::size_t) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete[](void* __ptr, std::size_t) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete(void* __ptr, std::size_t, std::align_val_t) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete[](void* __ptr, std::size_t, std::align_val_t) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete(void* __ptr, const std::nothrow_t&) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete[](void* __ptr, const std::nothrow_t&) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete(void* __ptr, std::align_val_t) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void operator delete[](void* __ptr, std::align_val_t) noexcept {
    ::__pstl_offload::__internal_free(__ptr);
}

#endif // __linux__
