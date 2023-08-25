// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H

#ifndef __SYCL_PSTL_OFFLOAD__
#error "__SYCL_PSTL_OFFLOAD__ macro should be defined to include this header"
#endif

#include <atomic>
#include <cstdlib>
#include <cassert>
#include <cerrno>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>

#include "usm_memory_replacement_common.h"

namespace __pstl_offload {

static std::atomic<sycl::device*> __active_device = nullptr;

static void __set_active_device(sycl::device* __new_active_device) {
    __active_device.store(__new_active_device, std::memory_order_release);
}

class __offload_policy_holder_type {
public:
    __offload_policy_holder_type()
        : _M_offload_device(__offload_device_selector)
        , _M_offload_policy(_M_offload_device)
    {
        __set_active_device(&_M_offload_device);
    }

    ~__offload_policy_holder_type()
    {
        __set_active_device(nullptr);
    }

    auto __get_policy() { return _M_offload_policy; }
private:
    static constexpr auto& __offload_device_selector =
#if __SYCL_PSTL_OFFLOAD__ == 1
    sycl::default_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 2
    sycl::cpu_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 3
    sycl::gpu_selector_v;
#else
#error "Unsupported value of __SYCL_PSTL_OFFLOAD__ macro"
#endif

    sycl::device _M_offload_device;
    oneapi::dpl::execution::device_policy<> _M_offload_policy;
}; // class __offload_policy_holder_type

static __offload_policy_holder_type __offload_policy_holder;

#if __linux__
inline auto __get_original_aligned_alloc() {
    using _aligned_alloc_type = void* (*)(std::size_t, std::size_t);

    static _aligned_alloc_type __orig_aligned_alloc = _aligned_alloc_type(dlsym(RTLD_NEXT, "aligned_alloc"));
    return __orig_aligned_alloc;
}
#endif // __linux__

static void* __internal_aligned_alloc(std::size_t __size, std::size_t __alignment) {
    sycl::device* __device = __active_device.load(std::memory_order_acquire);
    void* __res = nullptr;

    if (__device) {
        __res = __allocate_shared_for_device(__device, __size, __alignment);
    } else {
        __res = __get_original_aligned_alloc()(__alignment, __size);
    }

    if (!__res)
        return nullptr;

    assert(std::uintptr_t(__res) % __alignment == 0);
    return __res;
}

static void* __errno_handling_internal_aligned_alloc(std::size_t __size, std::size_t __alignment) {
    void* __ptr = __internal_aligned_alloc(__size, __alignment);
    if (!__ptr) {
        errno = ENOMEM;
        return nullptr;
    }
    return __ptr;
}

static void* __internal_operator_new(std::size_t __size, std::size_t __alignment = alignof(std::max_align_t)) {
    void* __res = __internal_aligned_alloc(__size, __alignment);

    while(!__res) {
        std::new_handler __handler = std::get_new_handler();
        if (__handler) {
            __handler();
        } else {
            throw std::bad_alloc{};
        }
        __res = __internal_aligned_alloc(__size, __alignment);
    }

    return __res;
}

static void* __internal_operator_new(const std::nothrow_t&, std::size_t __size, std::size_t __alignment = alignof(std::max_align_t)) noexcept {
    void* __res = nullptr;
    try {
        __res = __internal_operator_new(__size, __alignment);
    } catch(...) {}
    return __res;
}

} // namespace __pstl_offload

#if __linux__

// valloc, pvalloc, __libc_valloc and __libc_pvalloc are not supported
// due to unsupported alignment on memory page

extern "C" {

inline void* __attribute__((always_inline)) malloc(std::size_t __size) {
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, alignof(std::max_align_t));
}

inline void* __attribute__((always_inline)) calloc(std::size_t __num, std::size_t __size) {
    void* __res = ::__pstl_offload::__errno_handling_internal_aligned_alloc(__num * __size, alignof(std::max_align_t));
    return __res ? std::memset(__res, 0, __num * __size) : nullptr;
}

inline void* __attribute__((always_inline)) realloc(void* __ptr, std::size_t __size) {
    return ::__pstl_offload::__internal_realloc(__ptr, __size);
}

inline void* __attribute__((always_inline)) memalign(std::size_t __alignment, std::size_t __size) noexcept {
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline int __attribute__((always_inline)) posix_memalign(void** __memptr, std::size_t __alignment, std::size_t __size) noexcept {
    if (!__pstl_offload::__is_power_of_two(__alignment))
        return EINVAL;

    void* __ptr = ::__pstl_offload::__internal_aligned_alloc(__size, __alignment);

    if (__ptr) {
        *__memptr = __ptr;
        return 0;
    }
    return ENOMEM;
}

inline int __attribute__((always_inline)) mallopt(int /*param*/, int /*value*/) noexcept {
    return 1;
}

inline void* __attribute__((always_inline)) aligned_alloc(std::size_t __alignment, std::size_t __size) {
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline void* __attribute__((always_inline)) __libc_malloc(std::size_t __size) {
    return malloc(__size);
}

inline void* __attribute__((always_inline)) __libc_calloc(std::size_t __num, std::size_t __size) {
    return calloc(__num, __size);
}

inline void* __attribute__((always_inline)) __libc_memalign(std::size_t __alignment, std::size_t __size) {
    return memalign(__alignment, __size);
}

inline void* __attribute__((always_inline)) __libc_realloc(void *__ptr, std::size_t __size) {
    return realloc(__ptr, __size);
}

} // extern "C"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winline-new-delete"

inline void* __attribute__((always_inline)) operator new(std::size_t __size) {
    return ::__pstl_offload::__internal_operator_new(__size);
}

inline void* __attribute__((always_inline)) operator new[](std::size_t __size) {
    return ::__pstl_offload::__internal_operator_new(__size);
}

inline void* __attribute__((always_inline)) operator new(std::size_t __size, const std::nothrow_t&) noexcept {
    return ::__pstl_offload::__internal_operator_new(std::nothrow, __size);
}

inline void* __attribute__((always_inline)) operator new[](std::size_t __size, const std::nothrow_t&) noexcept {
    return ::__pstl_offload::__internal_operator_new(std::nothrow, __size);
}

inline void* __attribute__((always_inline)) operator new(std::size_t __size, std::align_val_t __al) {
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al));
}

inline void* __attribute__((always_inline)) operator new[](std::size_t __size, std::align_val_t __al) {
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al));
}

inline void* __attribute__((always_inline)) operator new(std::size_t __size, std::align_val_t __al, const std::nothrow_t&) noexcept {
    return ::__pstl_offload::__internal_operator_new(std::nothrow, __size, std::size_t(__al));
}

inline void* __attribute__((always_inline)) operator new[](std::size_t __size, std::align_val_t __al, const std::nothrow_t&) noexcept {
    return ::__pstl_offload::__internal_operator_new(std::nothrow, __size, std::size_t(__al));
}

#pragma GCC diagnostic pop

#endif // __linux__

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H
