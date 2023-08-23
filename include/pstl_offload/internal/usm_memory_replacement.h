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

#include "usm_memory_common.h"

namespace oneapi {
namespace dpl {
namespace pstl_offload {

static const auto& get_device_selector() {
#if __SYCL_PSTL_OFFLOAD__ == 1
    return sycl::default_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 2
    return sycl::cpu_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 3
    return sycl::gpu_selector_v;
#else
#error "Unsupported value of __SYCL_PSTL_OFFLOAD__ macro"
#endif // __SYCL_PSTL_OFFLOAD__
}

static void set_active_device(sycl::device*);

class OffloadPolicyHolder {
public:
    OffloadPolicyHolder()
        : offload_device(get_device_selector())
        , offload_policy(offload_device)
    {
        set_active_device(&offload_device);
    }

    ~OffloadPolicyHolder()
    {
        set_active_device(nullptr);
    }

    auto& get_policy() { return offload_policy; }
private:
    sycl::device offload_device;
    oneapi::dpl::execution::device_policy<> offload_policy;
}; // class OffloadPolicyHolder

static OffloadPolicyHolder offload_policy_holder;

#if __linux__
static auto get_original_aligned_alloc() {
    using aligned_alloc_type = void* (*)(std::size_t, std::size_t);

    static aligned_alloc_type orig_aligned_alloc = aligned_alloc_type(dlsym(RTLD_NEXT, "aligned_alloc"));
    return orig_aligned_alloc;
}
#endif // __linux__

static std::atomic<sycl::device*> active_device = nullptr;

static void set_active_device(sycl::device* new_active_device) {
    active_device.store(new_active_device, std::memory_order_release);
}

static void* usm_aligned_alloc(std::size_t size, std::size_t alignment) {
    sycl::device* device = active_device.load(std::memory_order_acquire);
    void* res = nullptr;

    if (device) {
        res = allocate_from_device(device, size, alignment);
    } else {
        res = get_original_aligned_alloc()(alignment, size);
    }

    if (!res)
        return nullptr;

    assert(std::uintptr_t(res) % alignment == 0);
    return res;
}

static void* errno_handling_usm_aligned_alloc(std::size_t size, std::size_t alignment) {
    void* ptr = usm_aligned_alloc(size, alignment);
    if (!ptr) {
        errno = ENOMEM;
        return nullptr;
    }
    return ptr;
}

static void* internal_operator_new(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
    void* res = usm_aligned_alloc(size, alignment);

    while(!res) {
        std::new_handler handler = std::get_new_handler();
        if (handler) {
            handler();
        } else {
            throw std::bad_alloc{};
        }
        res = usm_aligned_alloc(size, alignment);
    }

    return res;
}

static void* internal_operator_new(const std::nothrow_t&, std::size_t size, std::size_t alignment = alignof(std::max_align_t)) noexcept {
    void* res = nullptr;
    try {
        res = internal_operator_new(size, alignment);
    } catch(...) {}
    return res;
}

} // namespace pstl_offload
} // namespace dpl
} // namespace oneapi

#if __linux__

// valloc, pvalloc, __libc_valloc and __libc_pvalloc are not supported
// due to unsupported alignment on memory page

extern "C" {

inline void* __attribute__((always_inline)) malloc(std::size_t size) {
    return oneapi::dpl::pstl_offload::errno_handling_usm_aligned_alloc(size, alignof(std::max_align_t));
}

inline void* __attribute__((always_inline)) calloc(std::size_t num, std::size_t size) {
    char* res = static_cast<char*>(oneapi::dpl::pstl_offload::errno_handling_usm_aligned_alloc(num * size, alignof(std::max_align_t)));
    return res ? memset(res, 0, num * size) : nullptr;
}

inline void* __attribute__((always_inline)) realloc(void* ptr, std::size_t size) {
    return oneapi::dpl::pstl_offload::internal_realloc(ptr, size);
}

inline void* __attribute__((always_inline)) memalign(std::size_t alignment, std::size_t size) noexcept {
    return oneapi::dpl::pstl_offload::errno_handling_usm_aligned_alloc(size, alignment);
}

inline int __attribute__((always_inline)) posix_memalign(void** memptr, std::size_t alignment, std::size_t size) noexcept {
    if (alignment == 0 || (alignment & alignment - 1) != 0) // alignment is not a power of two
        return EINVAL;

    void* ptr = oneapi::dpl::pstl_offload::usm_aligned_alloc(size, alignment);

    if (ptr) {
        *memptr = ptr;
        return 0;
    }
    return ENOMEM;
}

inline int __attribute__((always_inline)) mallopt(int /*param*/, int /*value*/) noexcept {
    return 1;
}

inline void* __attribute__((always_inline)) aligned_alloc(std::size_t alignment, std::size_t size) {
    return oneapi::dpl::pstl_offload::errno_handling_usm_aligned_alloc(size, alignment);
}

inline void* __attribute__((always_inline)) __libc_malloc(std::size_t size) {
    return malloc(size);
}

inline void* __attribute__((always_inline)) __libc_calloc(std::size_t num, std::size_t size) {
    return calloc(num, size);
}

inline void* __attribute__((always_inline)) __libc_memalign(std::size_t alignment, std::size_t size) {
    return memalign(alignment, size);
}

inline void* __attribute__((always_inline)) __libc_realloc(void *ptr, std::size_t size) {
    return realloc(ptr, size);
}

} // extern "C"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winline-new-delete"

inline void* __attribute__((always_inline)) operator new(std::size_t size) {
    return oneapi::dpl::pstl_offload::internal_operator_new(size);
}

inline void* __attribute__((always_inline)) operator new[](std::size_t size) {
    return oneapi::dpl::pstl_offload::internal_operator_new(size);
}

inline void* __attribute__((always_inline)) operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return oneapi::dpl::pstl_offload::internal_operator_new(std::nothrow, size);
}

inline void* __attribute__((always_inline)) operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return oneapi::dpl::pstl_offload::internal_operator_new(std::nothrow, size);
}

inline void* __attribute__((always_inline)) operator new(std::size_t size, std::align_val_t al) {
    return oneapi::dpl::pstl_offload::internal_operator_new(size, std::size_t(al));
}

inline void* __attribute__((always_inline)) operator new[](std::size_t size, std::align_val_t al) {
    return oneapi::dpl::pstl_offload::internal_operator_new(size, std::size_t(al));
}

inline void* __attribute__((always_inline)) operator new(std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    return oneapi::dpl::pstl_offload::internal_operator_new(std::nothrow, size, std::size_t(al));
}

inline void* __attribute__((always_inline)) operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    return oneapi::dpl::pstl_offload::internal_operator_new(std::nothrow, size, std::size_t(al));
}

#pragma GCC diagnostic pop

#endif // __linux__

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H
