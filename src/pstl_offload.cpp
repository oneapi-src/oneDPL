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

#define _PSTL_OFFLOAD_BINARY_VERSION_MAJOR 1
#define _PSTL_OFFLOAD_BINARY_VERSION_MINOR 0
#define _PSTL_OFFLOAD_BINARY_VERSION_PATCH 0

#if __linux__

#define _PSTL_OFFLOAD_EXPORT __attribute__((visibility("default")))

#include <dlfcn.h>
#include <string.h>
#include <unistd.h>

namespace __pstl_offload
{

using __free_func_type = void (*)(void*);

// list of objects for delayed releasing
struct __delayed_free_list {
    __delayed_free_list* _M_next;
    void*                _M_to_free;
};

// are we inside dlsym call?
static thread_local bool __dlsym_called = false;
// objects released inside of dlsym call
static thread_local __delayed_free_list* __delayed_free = nullptr;

static void
__free_delayed_list(void* __ptr_to_free, __free_func_type __orig_free)
{
    // It's enough to check __delayed_free only at this point,
    // as __delayed_free filled only inside dlsym(RTLD_NEXT, "free").
    while (__delayed_free)
    {
        __delayed_free_list* __next = __delayed_free->_M_next;
        // it's possible that an object to be released during 1st call of __internal_free
        // would be released 2nd time from inside nested dlsym call. To prevent "double free"
        // situation, check for it explicitly.
        if (__ptr_to_free != __delayed_free->_M_to_free)
        {
            __orig_free(__delayed_free->_M_to_free);
        }
        __orig_free(__delayed_free);
        __delayed_free = __next;
    }
}

static __free_func_type
__get_original_free_checked(void* __ptr_to_free)
{
    __dlsym_called = true;
    __free_func_type __orig_free = __free_func_type(dlsym(RTLD_NEXT, "free"));
    __dlsym_called = false;
    if (!__orig_free)
    {
        throw std::system_error(std::error_code(), dlerror());
    }

    // Releasing objects from delayed release list.
    __free_delayed_list(__ptr_to_free, __orig_free);

    return __orig_free;
}

static void
__original_free(void* __ptr_to_free)
{
    static __free_func_type __orig_free = __get_original_free_checked(__ptr_to_free);
    __orig_free(__ptr_to_free);
}

static auto
__get_original_msize()
{
    using __msize_func_type = std::size_t (*)(void*);

    static __msize_func_type __orig_msize =
        __msize_func_type(dlsym(RTLD_NEXT, "malloc_usable_size"));
    return __orig_msize;
}


static auto
__get_original_realloc()
{
    using __realloc_func_type = void* (*)(void*, std::size_t);

    static __realloc_func_type __orig_realloc = __realloc_func_type(dlsym(RTLD_NEXT, "realloc"));
    return __orig_realloc;
}

struct __hash_aligned_ptr
{
    uintptr_t operator()(void *p) const
    {
        // we know that addresses are at least page-aligned, so drop 11 right bits that are zeros,
        // and multiply to some prime number for bits hashing
        // TODO: should well-known hashing function be used?
        constexpr unsigned shift = 11;
        return ((uintptr_t)p >> shift) * 1111111111111111111LLU;
    }
};

struct __large_aligned_ptr_header
{
    __sycl_device_shared_ptr _M_device_ptr;
    std::size_t   _M_requested_number_of_bytes;
};

std::mutex __large_aligned_ptrs_mtx;
// map user pointer to header
std::unordered_map<void*, __large_aligned_ptr_header, __hash_aligned_ptr> __large_aligned_ptrs;

static void
__register_large_alignment_ptr(void* __ptr, size_t __size, __sycl_device_shared_ptr __device_ptr)
{
    const std::lock_guard<std::mutex> l(__large_aligned_ptrs_mtx);
    auto __ret = __large_aligned_ptrs.emplace(__ptr, __large_aligned_ptr_header{__device_ptr, __size});
    assert(__ret.second); // the pointer must be unique
}

static std::optional<__large_aligned_ptr_header>
__unregister_large_aligned_ptr(void* __ptr)
{
    const std::lock_guard<std::mutex> l(__large_aligned_ptrs_mtx);
    auto __iter = __large_aligned_ptrs.find(__ptr);
    if (__iter == __large_aligned_ptrs.end())
    {
        return std::nullopt;
    }
    __large_aligned_ptr_header __header = __iter->second;
    __large_aligned_ptrs.erase(__iter);
    return __header;
}

inline void
__free_usm_pointer(__block_header* __header)
{
    assert(__header != nullptr);
    __header->_M_uniq_const = 0;
    sycl::context __context = __header->_M_device.__get_context();
    __header->_M_device.__reset();
    sycl::free(__header->_M_original_pointer, __context);
}

static void
__internal_free(void* __user_ptr)
{
    if (__user_ptr != nullptr)
    {
        if (__block_header* __header = static_cast<__block_header*>(__user_ptr) - 1; __same_memory_page(__user_ptr, __header))
        {
            if (__header->_M_uniq_const == __uniq_type_const)
            {
                __free_usm_pointer(__header);
                return;
            }
        }
        else if (std::optional<__large_aligned_ptr_header> __header = __unregister_large_aligned_ptr(__user_ptr);
                 __header.has_value())
        {
            sycl::context __context = __header->_M_device_ptr.__get_context();
            sycl::free(__user_ptr, __context);
            return;
        }

        if (__dlsym_called)
        {
            // Delay releasing till exit of dlsym. We do not overload malloc globally,
            // so can use it safely. Do not use new to able to use free() during
            // __delayed_free_list releasing.
            void* __buf = malloc(sizeof(__delayed_free_list));
            if (!__buf)
            {
                throw std::bad_alloc();
            }
            __delayed_free_list* __h = new (__buf) __delayed_free_list{__delayed_free, __user_ptr};
            __delayed_free = __h;
        }
        else
        {
            __original_free(__user_ptr);
        }
    }
}

static std::size_t
__internal_msize(void* __user_ptr)
{
    if (__user_ptr == nullptr)
    {
        return 0;
    }

    if (__block_header* __header = static_cast<__block_header*>(__user_ptr) - 1; __same_memory_page(__user_ptr, __header))
    {
        if (__header->_M_uniq_const == __uniq_type_const)
        {
            return __header->_M_requested_number_of_bytes;
        }
    }
    else if (std::optional<__large_aligned_ptr_header> __header = __unregister_large_aligned_ptr(__user_ptr);
             __header.has_value())
    {
        return __header->_M_requested_number_of_bytes;
    }
    return __get_original_msize()(__user_ptr);
}

static void*
__realloc_allocate_shared(__sycl_device_shared_ptr __device_ptr, void* __user_ptr, std::size_t __old_size, std::size_t __new_size)
{
    void* __new_ptr = __allocate_shared_for_device(__device_ptr, __new_size, alignof(std::max_align_t));

    if (__new_ptr != nullptr)
    {
        std::memcpy(__new_ptr, __user_ptr, std::min(__old_size, __new_size));
    }
    else
    {
        errno = ENOMEM;
    }
    return __new_ptr;
}

_PSTL_OFFLOAD_EXPORT void*
__realloc_impl(void* __user_ptr, std::size_t __new_size)
{
    assert(__user_ptr != nullptr);

    if (!__new_size)
    {
        __internal_free(__user_ptr);
        return nullptr;
    }

    if (__block_header* __header = static_cast<__block_header*>(__user_ptr) - 1; __same_memory_page(__user_ptr, __header))
    {
        if (__header->_M_uniq_const == __uniq_type_const)
        {
            void* __result = __realloc_allocate_shared(__header->_M_device, __user_ptr, __header->_M_requested_number_of_bytes, __new_size);
            if (__result)
            {
                __free_usm_pointer(__header);
            }
            return __result;
        }
    }
    else if (std::optional<__large_aligned_ptr_header> __header = __unregister_large_aligned_ptr(__user_ptr);
             __header.has_value())
    {
        void* __result = __realloc_allocate_shared(__header->_M_device_ptr, __user_ptr, __header->_M_requested_number_of_bytes, __new_size);
        if (__result)
        {
            sycl::context __context = __header->_M_device_ptr.__get_context();
            sycl::free(__user_ptr, __context);
        }
        return __result;
    }

    // __user_ptr is not a USM pointer, use original realloc function
    return __get_original_realloc()(__user_ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT void*
__allocate_shared_for_device_large_alignment(__sycl_device_shared_ptr __device_ptr, std::size_t __size, std::size_t __alignment)
{
    sycl::device __device = __device_ptr.__get_device();
    sycl::context __context = __device_ptr.__get_context();
    void* __ptr = sycl::aligned_alloc_shared(__alignment, __size, __device, __context);

    if (__ptr)
    {
        __register_large_alignment_ptr(__ptr, __size, __device_ptr);
    }
    return __ptr;
}

} // namespace __pstl_offload

extern "C"
{

_PSTL_OFFLOAD_EXPORT void free(void* __ptr)
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void __libc_free(void *__ptr)
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void* realloc(void* __ptr, std::size_t __new_size)
{
    return ::__pstl_offload::__internal_realloc(__ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT void* __libc_realloc(void* __ptr, std::size_t __new_size)
{
    return ::__pstl_offload::__internal_realloc(__ptr, __new_size);
}

_PSTL_OFFLOAD_EXPORT std::size_t malloc_usable_size(void* __ptr) noexcept
{
    return ::__pstl_offload::__internal_msize(__ptr);
}

} // extern "C"

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, std::size_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, std::size_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, std::size_t, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, std::size_t, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, const std::nothrow_t&) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, const std::nothrow_t&) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete(void* __ptr, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

_PSTL_OFFLOAD_EXPORT void
operator delete[](void* __ptr, std::align_val_t) noexcept
{
    ::__pstl_offload::__internal_free(__ptr);
}

#endif // __linux__
