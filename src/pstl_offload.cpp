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
struct __delayed_free_header {
    __delayed_free_header* _M_next;
    void*                  _M_to_free;

    __delayed_free_header(__delayed_free_header* __next, void* __to_free)
        : _M_next(__next), _M_to_free(__to_free) {}
};

// are we inside dlsym call?
static thread_local bool __dlsym_called = false;
// objects released inside of dlsym call
static thread_local __delayed_free_header* __delayed_free = nullptr;

static __free_func_type
__get_original_free()
{
    __dlsym_called = true;
    __free_func_type __orig_free = __free_func_type(dlsym(RTLD_NEXT, "free"));
    __dlsym_called = false;
    if (!__orig_free)
    {
        throw std::system_error(std::error_code(), dlerror());
    }
    return __orig_free;
}

static auto
__get_original_msize()
{
    using __msize_func_type = std::size_t (*)(void*);

    static __msize_func_type __orig_msize =
        __msize_func_type(dlsym(RTLD_NEXT, "malloc_usable_size"));
    return __orig_msize;
}

static void
__internal_free(void* __user_ptr)
{
    if (__user_ptr != nullptr)
    {
        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __free_usm_pointer(__header);
        }
        else
        {
            if (__dlsym_called)
            {
                // Delay releasing till exit of dlsym. We do not overload malloc globally,
                // so can use it safely. Do not use new to able to use free() during
                // __delayed_free_header releasing.
                void* __buf = malloc(sizeof(__delayed_free_header));
                __delayed_free_header* __h = new(__buf) __delayed_free_header(__delayed_free, __user_ptr);
                __delayed_free = __h;
            }
            else
            {
                static __free_func_type __orig_free = __get_original_free();

                // releasing objects from delayed release list
                while (__delayed_free)
                {
                    __delayed_free_header* __next = __delayed_free->_M_next;
                    // it's possible that an object to be released during 1st call of __internal_free
                    // would be released 2nd time from inside nested dlsym call. To prevent "double free"
                    // situation, check for it explicitly.
                    if (__user_ptr != __delayed_free->_M_to_free)
                    {
                        __orig_free(__delayed_free->_M_to_free);
                    }
                    __orig_free(__delayed_free);
                    __delayed_free = __next;
                }
                __orig_free(__user_ptr);
            }
        }
    }
}

static std::size_t
__internal_msize(void* __user_ptr)
{
    std::size_t __res = 0;
    if (__user_ptr != nullptr)
    {

        __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

        if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __res = __header->_M_requested_number_of_bytes;
        }
        else
        {
            __res = __get_original_msize()(__user_ptr);
        }
    }
    return __res;
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
