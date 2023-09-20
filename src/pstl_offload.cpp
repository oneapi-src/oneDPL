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

#include <atomic>
#include <unordered_map>
#include <mutex>

namespace __pstl_offload
{

static auto
__get_original_free()
{
    using __free_func_type = void (*)(void*);

    static __free_func_type __orig_free = __free_func_type(dlsym(RTLD_NEXT, "free"));
    return __orig_free;
}

template <typename _T>
class __overaligned_pointer_table_allocator {
public:
    using value_type = _T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    __overaligned_pointer_table_allocator()
        : _M_original_free(__get_original_free()) {}

    template <typename _U>
    __overaligned_pointer_table_allocator(const __overaligned_pointer_table_allocator<_U>&)
        : _M_original_free(__get_original_free()) {}

    _T* allocate(std::size_t __n) {
        return static_cast<_T*>(std::aligned_alloc(alignof(_T), sizeof(_T) * __n));
    }

    void deallocate(_T* __ptr, std::size_t) {
        _M_original_free(__ptr);
    }

    friend bool operator==(const __overaligned_pointer_table_allocator&, const __overaligned_pointer_table_allocator&) {
        return true;
    }

    friend bool operator!=(const __overaligned_pointer_table_allocator&, const __overaligned_pointer_table_allocator&) {
        return false;
    }
private:
    // Save original aligned_alloc and free because dlsym can call malloc/free internally
    void (*_M_original_free)(void*);
}; // class __overaligned_pointer_table_allocator

static bool __access_overaligned_pointer_table_alive_flag(bool write, bool new_value = true) {
    static std::atomic<bool> __overaligned_pointer_table_alive_flag = false;

    if (write) {
        __overaligned_pointer_table_alive_flag.store(new_value, std::memory_order_release);
        return false;
    }
    return __overaligned_pointer_table_alive_flag.load(std::memory_order_acquire);
}

static void __mark_overaligned_pointer_table_alive() {
    __access_overaligned_pointer_table_alive_flag(/*write = */true, /*new_value = */true);
}

static void __mark_overaligned_pointer_table_dead() {
    __access_overaligned_pointer_table_alive_flag(/*write = */true, /*new_value = */false);
}

_PSTL_OFFLOAD_EXPORT bool __is_overaligned_pointer_table_alive() {
    return __access_overaligned_pointer_table_alive_flag(/*write = */false);
}

struct __pointer_info
{
    __pointer_info(sycl::device* __device, std::size_t __bytes)
        : _M_device(__device), _M_requested_number_of_bytes(__bytes) {}

    sycl::device* _M_device;
    std::size_t _M_requested_number_of_bytes;
}; // struct __pointer_info

using __overaligned_pointer_table_container_type = std::unordered_map<void*, __pointer_info, std::hash<void*>, std::equal_to<void*>,
                                                                      __overaligned_pointer_table_allocator<std::pair<void* const, __pointer_info>>>;
using __overaligned_pointer_table_mutex_type = std::mutex;
using __overaligned_pointer_table_lock_type = std::unique_lock<__overaligned_pointer_table_mutex_type>;

struct __overaligned_pointer_table_type : __overaligned_pointer_table_container_type {
    __overaligned_pointer_table_type() {
        __mark_overaligned_pointer_table_alive();
    }
    ~__overaligned_pointer_table_type() {
        __mark_overaligned_pointer_table_dead();
    }
};

static __overaligned_pointer_table_type __overaligned_pointer_table;
static __overaligned_pointer_table_mutex_type __overaligned_pointer_table_mutex;

_PSTL_OFFLOAD_EXPORT void __register_overaligned_pointer(void* __ptr, std::size_t __allocated_size, sycl::device* __device) {
    assert(__is_overaligned_pointer_table_alive());

    __overaligned_pointer_table_lock_type __lock(__overaligned_pointer_table_mutex);
    [[maybe_unused]] auto __result = __overaligned_pointer_table.emplace(std::piecewise_construct, std::forward_as_tuple(__ptr),
                                                                         std::forward_as_tuple(__device, __allocated_size));
    assert(__result.second);
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
            if ((std::uintptr_t(__user_ptr) & (__get_memory_page_size() - 1)) == 0 &&
                __is_overaligned_pointer_table_alive())
            {
                __overaligned_pointer_table_lock_type __lock(__overaligned_pointer_table_mutex);
                auto __it = __overaligned_pointer_table.find(__user_ptr);

                if (__it != __overaligned_pointer_table.end())
                {
                    // Free overaligned pointer
                    auto __ptr_info = __it->second;
                    __overaligned_pointer_table.erase(__it);
                    __lock.unlock();
                    sycl::free(__user_ptr, __ptr_info._M_device->get_platform().ext_oneapi_get_default_context());
                    return;
                }
            }
            __get_original_free()(__user_ptr);
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
        // else if ((std::uintptr_t(__user_ptr) & (__get_memory_page_size() - 1)) == 0 &&
        //          __is_overaligned_pointer_table_alive() &&
        //            __overaligned_pointer_table_lock_type __lock(__overaligned_pointer_table_mutex);
        //            auto __it = __overaligned_pointer_table.find(__user_ptr); __it != __overaligned_pointer_table.end())
        // {
        //     return __it->second._M_requested_number_of_bytes;
        // }
        else
        {
            if ((std::uintptr_t(__user_ptr) & (__get_memory_page_size() - 1)) == 0 &&
                __is_overaligned_pointer_table_alive())
            {
                __overaligned_pointer_table_lock_type __lock(__overaligned_pointer_table_mutex);
                auto __it = __overaligned_pointer_table.find(__user_ptr);

                if (__it != __overaligned_pointer_table.end())
                {
                    return __it->second._M_requested_number_of_bytes;
                }
            }
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
