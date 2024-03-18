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

/*
Functions that allocates memory are replaced on per-TU base. For better reliability, releasing
functions are replaced globally and an origin of a releasing object is checked to call right
releasing function. realloc can do both allocation and releasing, so it must be both replaced
on per-TU base and globally, but with different semantics in wrt newly allocated memory.

Global replacement under Linux is done during link-time or during load via LD_PRELOAD. To implememnt
global replacement under Windows the dynamic runtime libraries are instrumented with help of
Microsoft Detours.
*/

#if _WIN64

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <detours.h>
#pragma GCC diagnostic pop

#endif

namespace __pstl_offload
{

using __free_func_type = void (*)(void*);

#if __linux__

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

static std::size_t
__original_msize(void* __user_ptr)
{
    using __msize_func_type = std::size_t (*)(void*);

    static __msize_func_type __orig_msize =
        __msize_func_type(dlsym(RTLD_NEXT, "malloc_usable_size"));
    return __orig_msize(__user_ptr);
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
                // __delayed_free_list releasing.
                void* __buf = malloc(sizeof(__delayed_free_list));
                if (!__buf)
                {
                    throw std::bad_alloc();
                }
                __delayed_free_list* __h = new(__buf) __delayed_free_list{__delayed_free, __user_ptr};
                __delayed_free = __h;
            }
            else
            {
                __original_free(__user_ptr);
            }
        }
    }
}

#elif _WIN64

static __free_func_type __original_free_ptr = free;
#if _DEBUG
static void (*__original_free_dbg_ptr)(void* userData, int blockType) = _free_dbg;
#endif
static __realloc_func_type __original_realloc_ptr = realloc;
static __free_func_type __original_aligned_free_ptr = _aligned_free;
static size_t (*__original_msize)(void *) = _msize;
static size_t (*__original_aligned_msize_ptr)(void *, std::size_t alignment, std::size_t offset) = _aligned_msize;
static void* (*__original_aligned_realloc_ptr)(void *, std::size_t size, std::size_t alignment) = _aligned_realloc;
static void* (*__original_expand_ptr)(void *, std::size_t size) = _expand;

static void
__internal_free_param(void* __user_ptr, __free_func_type __custom_free)
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
            __custom_free(__user_ptr);
        }
    }
}

static void
__internal_free(void* __user_ptr)
{
    __internal_free_param(__user_ptr, __original_free_ptr);
}

#endif // _WIN64

static std::size_t
__internal_msize(void* __user_ptr)
{
    if (__user_ptr == nullptr)
    {
#if _WIN64
        errno = EINVAL;
        return -1;
#else
        return 0;
#endif
    }

    std::size_t __res = 0;
    __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
    {
        __res = __header->_M_requested_number_of_bytes;
    }
    else
    {
        __res = __original_msize(__user_ptr);
    }
    return __res;
}

#if _WIN64

static std::size_t
__internal_aligned_msize(void* __user_ptr, std::size_t __alignment, std::size_t __offset)
{
    if (__user_ptr == nullptr || !__is_power_of_two(__alignment))
    {
        errno = EINVAL;
        return -1;
    }

    std::size_t __res = 0;
    __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
    {
        __res = __header->_M_requested_number_of_bytes;
    }
    else
    {
        __res = __original_aligned_msize_ptr(__user_ptr, __alignment, __offset);
    }
    return __res;
}

static void
__internal_aligned_free(void* __user_ptr)
{
    __internal_free_param(__user_ptr, __original_aligned_free_ptr);
}

#if _DEBUG

static void
__internal_free_dbg(void* __user_ptr, int __type)
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
            __original_free_dbg_ptr(__user_ptr, __type);
        }
    }
}

#endif // _DEBUG

void*
__original_malloc(std::size_t size)
{
    return malloc(size);
}

void*
__original_aligned_alloc(std::size_t size, std::size_t alignment)
{
    return _aligned_malloc(size, alignment);
}

void*
__original_realloc(void* __user_ptr, std::size_t __new_size)
{
    return __original_realloc_ptr(__user_ptr, __new_size);
}

void*
__original_aligned_realloc(void* __user_ptr, std::size_t __new_size, std::size_t __new_alignment)
{
    return __original_aligned_realloc_ptr(__user_ptr, __new_size, __new_alignment);
}

static void*
__internal_expand(void* /*__user_ptr*/, std::size_t /*__size*/)
{
    // do not support _expand()
    return nullptr;
}

std::size_t
__get_page_size()
{
    SYSTEM_INFO __si;
    GetSystemInfo(&__si);
    return __si.dwPageSize;
}

static bool
__do_functions_replacement()
{
    // May fail, because process commonly not started with DetourCreateProcessWithDll*. Ignore it.
    DetourRestoreAfterWith();

    LONG ret = DetourTransactionBegin();
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: DetourTransactionBegin returns %ld\n", ret);
        return false;
    }

    // Operators from delete family is implemented by compiler with call to an appropriate free function.
    // Those functions are in the dll and they are replaced, so no need to directly replaced delete.
    // TODO: rarely-used _aligned_offset_* and _set*_invalid_parameter_handler functions are not supported yet
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmicrosoft-cast"
    ret = DetourAttach(&(PVOID&)__original_free_ptr, __internal_free);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: free replacement failed with %ld\n", ret);
        return false;
    }
#if _DEBUG
    // _free_dbg is called by delete in debug mode
    ret = DetourAttach(&(PVOID&)__original_free_dbg_ptr, __internal_free_dbg);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _free_dbg replacement failed with %ld\n", ret);
        return false;
    }
#endif
    ret = DetourAttach(&(PVOID&)__original_realloc_ptr, __internal_realloc);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: realloc replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_aligned_free_ptr, __internal_aligned_free);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _aligned_free replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_msize, __internal_msize);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _msize replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_aligned_msize_ptr, __internal_aligned_msize);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _aligned_msize replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_aligned_realloc_ptr, __internal_aligned_realloc);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _aligned_realloc replacement failed with %ld\n", ret);
        return false;
    }
    ret = DetourAttach(&(PVOID&)__original_expand_ptr, __internal_expand);
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: _expand replacement failed with %ld\n", ret);
        return false;
    }
#pragma GCC diagnostic pop

    ret = DetourTransactionCommit();
    if (NO_ERROR != ret)
    {
        fprintf(stderr, "Failed function replacement: DetourTransactionCommit returns %ld\n", ret);
        return false;
    }
    return true;
}

#endif // _WIN64

} // namespace __pstl_offload

#if __linux__
extern "C"
{

#define _PSTL_OFFLOAD_EXPORT __attribute__((visibility("default")))

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

#elif _WIN64

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
extern "C" BOOL WINAPI DllMain(HINSTANCE hInst, DWORD callReason, LPVOID reserved)
{
    BOOL ret = TRUE;

    if (callReason==DLL_PROCESS_ATTACH && reserved && hInst)
    {
        ret = __pstl_offload::__do_functions_replacement() ? TRUE : FALSE;
    }

    return ret;
}
#pragma GCC diagnostic pop

#endif // _WIN64
