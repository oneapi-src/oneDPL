// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_COMMON_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_COMMON_H

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>

#if __linux__
#include <dlfcn.h>
#include <unistd.h>
#endif // __linux__

namespace __pstl_offload {

inline constexpr std::size_t __uniq_type_const = 0x23499abc405a9bccLLU;

struct __block_header {
    std::size_t _M_uniq_const;
    void* _M_original_pointer;
    std::size_t _M_requested_number_of_bytes;
    sycl::device* _M_device;
}; // struct __block_header

inline constexpr std::size_t __header_offset = 32;

static_assert(sizeof(__block_header) <= __header_offset);

#if __linux__

inline constexpr bool __is_power_of_two(std::size_t __number) {
    return __number != 0 && (__number & __number - 1) == 0;
}

inline std::size_t __get_memory_page_size() {
    static std::size_t __memory_page_size = sysconf(_SC_PAGESIZE);
    assert(__is_power_of_two(__memory_page_size));
    return __memory_page_size;
}

inline void* __allocate_shared_for_device(sycl::device* __device, std::size_t __size, std::size_t __alignment) {
    // Unsupported alignment - impossible to guarantee that the returned pointer and memory header
    // would be on the same memory page if the alignment for more than a memory page is requested
    if (__alignment >= __get_memory_page_size()) {
        return nullptr;
    }

    std::size_t __base_offset = std::max(__alignment, __header_offset);

    // Memory block allocated with sycl::aligned_alloc_shared should be aligned to at least HEADER_OFFSET * 2
    // to guarantee that header and header + HEADER_OFFSET (user pointer) would be placed in one memory page
    std::size_t __usm_alignment = __base_offset << 1;
    // Required number of bytes to store memory header and preserve alignment on returned pointer
    // usm_alignment bytes are reserved to store memory header
    std::size_t __usm_size = __size + __base_offset;

    sycl::context __context = __device->get_platform().ext_oneapi_get_default_context();
    void* __ptr = sycl::aligned_alloc_shared(__usm_alignment, __usm_size, *__device, __context);

    if (!__ptr)
        return nullptr;

    void* __res = static_cast<char*>(__ptr) + __base_offset;
    __block_header* __header = static_cast<__block_header*>(__res) - 1;
    *__header = __block_header{__uniq_type_const, __ptr, __size, __device};
    return __res;
}

inline bool __same_memory_page(void* __ptr1, void* __ptr2) {
    std::uintptr_t __page_size = __get_memory_page_size();
    std::uintptr_t __page_mask = ~(__page_size - 1);
    std::uintptr_t __ptr1_page_begin = std::uintptr_t(__ptr1) & __page_mask;
    std::uintptr_t __ptr2_page_begin = std::uintptr_t(__ptr2) & __page_mask;
    return __ptr1_page_begin == __ptr2_page_begin;
}

inline auto __get_original_realloc() {
    using _realloc_type = void* (*)(void*, std::size_t);

    static _realloc_type __orig_realloc = _realloc_type(dlsym(RTLD_NEXT, "realloc"));
    return __orig_realloc;
}

static void* __internal_realloc(void* __user_ptr, std::size_t __new_size) {
    if (!__user_ptr) {
        return std::malloc(__new_size);
    }

    __block_header* __header = reinterpret_cast<__block_header*>(__user_ptr) - 1;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const) {
        if (__header->_M_requested_number_of_bytes < __new_size) {
            assert(__header->_M_device);
            void* __new_ptr = __allocate_shared_for_device(__header->_M_device, __new_size, alignof(std::max_align_t));

            if (!__new_ptr) {
                errno = ENOMEM;
                return nullptr;
            }

            std::memcpy(__new_ptr, __user_ptr, __header->_M_requested_number_of_bytes);

            // Free previously allocated memory
            void* __original_pointer = __header->_M_original_pointer;
            sycl::context __context = __header->_M_device->get_platform().ext_oneapi_get_default_context();
            sycl::free(__original_pointer, __context);
            return __new_ptr;
        }
        return __user_ptr;
    }

    return __get_original_realloc()(__user_ptr, __new_size);
}

#endif

} // namespace __pstl_offload

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_COMMON_H
