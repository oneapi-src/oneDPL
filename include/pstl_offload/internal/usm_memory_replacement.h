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

#if !__SYCL_PSTL_OFFLOAD__
#    error "PSTL offload compiler mode should be enabled to use this header"
#endif

#include <cstdlib>
#include <cassert>
#include <cerrno>
#include <optional>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>

#include "usm_memory_replacement_common.h"

namespace __pstl_offload
{

// allocation can be requested before static ctor or after static dtor runs, have a flag for that
// keep it out of __offload_policy_holder_type to not access object before ctor or after dtor
static std::atomic_bool __device_ready;

static void
__set_device_status(bool __ready)
{
    __device_ready.store(__ready, std::memory_order_release);
}

static auto
__get_offload_device_selector()
{
#if __SYCL_PSTL_OFFLOAD__ == 1
    return sycl::default_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 2
    return sycl::cpu_selector_v;
#elif __SYCL_PSTL_OFFLOAD__ == 3
    return sycl::gpu_selector_v;
#else
#    error "PSTL offload is not enabled or the selected value is unsupported"
#endif
}

// Stripped down spin mutex. Propose is to implement mutex able to be used from zero-initialized
// memory without need in constructors. This allows to use the mutex in file-scope constructors
// and destructors.
class __spin_mutex
{
    std::atomic_flag _M_flag = ATOMIC_FLAG_INIT;

  public:
    void
    lock()
    {
        while (_M_flag.test_and_set(std::memory_order_acquire))
        {
            // TODO: exponential backoff or switch to wait() from c++20
            std::this_thread::yield();
        }
    }

    void
    unlock()
    {
        _M_flag.clear(std::memory_order_release);
    }
};

static __spin_mutex __offload_policy_holder_mtx;

class __offload_policy_holder_type
{
    using __set_device_status_func_type = void (*)(bool);

  public:
    // Since the global object of __offload_policy_holder_type is static but the template constructor
    // of the class is inline, we need to avoid calling static functions inside of the constructor
    // and pass the pointer to exact function as an argument to guarantee that the correct offload device
    // would be stored in each translation unit
    template <typename _DeviceSelector>
    __offload_policy_holder_type(const _DeviceSelector& __device_selector,
                                 __set_device_status_func_type __set_device_status_func, __spin_mutex& __mtx)
        : _M_set_device_status_func(__set_device_status_func)
    {
        sycl::device _device;

        try
        {
            _device = sycl::device(__device_selector);
        }
        catch (const sycl::exception& e)
        {
            // __device_selector call throws with e.code() == sycl::errc::runtime when device selection unable
            // to get offload device with required type. Do not pass an exception, as ctor is called for
            // a static object and the exception can't be processed.
            // Remember the situation as empty _M_device and re-throw exception when asked for
            // the policy from user's code.
            // Re-throw in every other case, as we don't know the reason of an exception.
            if (e.code() == sycl::errc::runtime)
            {
                return;
            }
            else
            {
                throw;
            }
        }

        std::scoped_lock __lock{__mtx};

        _M_offload_device.__init(_device);
        _M_offload_policy = oneapi::dpl::execution::device_policy<>(_device);
        _M_set_device_status_func(true);
    }

    ~__offload_policy_holder_type()
    {
        std::scoped_lock __lock{__offload_policy_holder_mtx};

        _M_set_device_status_func(false);
    }

    static auto
    __get_policy(__offload_policy_holder_type& __this)
    {
        std::scoped_lock __lock{__offload_policy_holder_mtx};

        if (!__device_ready.load(std::memory_order_acquire))
        {
            throw sycl::exception(sycl::errc::runtime);
        }
        return __this._M_offload_policy;
    }

    static __sycl_device_shared_ptr
    __get_device_ptr(__offload_policy_holder_type& __this)
    {
        std::scoped_lock __lock{__offload_policy_holder_mtx};

        if (__device_ready.load(std::memory_order_acquire))
        {
            // it's safe to use copy ctor here, because we under __offload_policy_holder_mtx
            // and ~__offload_policy_holder_type() has not been called
            return __sycl_device_shared_ptr(__this._M_offload_device);
        }
        else
        {
            return __sycl_device_shared_ptr{};
        }
    }

  private:
    __sycl_device_shared_ptr _M_offload_device;
    oneapi::dpl::execution::device_policy<> _M_offload_policy;
    __set_device_status_func_type _M_set_device_status_func;
}; // class __offload_policy_holder_type

static __offload_policy_holder_type __offload_policy_holder{__get_offload_device_selector(), &__set_device_status,
                                                            __offload_policy_holder_mtx};

#if __linux__
inline void*
__original_aligned_alloc(std::size_t __alignment, std::size_t __size)
{
    using __aligned_alloc_func_type = void* (*)(std::size_t, std::size_t);

    static __aligned_alloc_func_type __orig_aligned_alloc =
        __aligned_alloc_func_type(dlsym(RTLD_NEXT, "aligned_alloc"));
    return __orig_aligned_alloc(__alignment, __size);
}
#endif // __linux__

static void*
__internal_aligned_alloc(std::size_t __size, std::size_t __alignment)
{
    if (__device_ready.load(std::memory_order_acquire))
    {
        if (__sycl_device_shared_ptr __dev = __offload_policy_holder_type::__get_device_ptr(__offload_policy_holder))
        {
            void* __res = __allocate_shared_for_device(std::move(__dev), __size, __alignment);
            assert((std::uintptr_t(__res) & (__alignment - 1)) == 0);
            return __res;
        }
    }
    void* __res = __original_aligned_alloc(__alignment, __size);

    assert((std::uintptr_t(__res) & (__alignment - 1)) == 0);
    return __res;
}

// This function is called by C allocation functions (malloc, calloc, etc)
// and sets errno on failure consistently with original memory allocating behavior
static void*
__errno_handling_internal_aligned_alloc(std::size_t __size, std::size_t __alignment)
{
    void* __ptr = __internal_aligned_alloc(__size, __alignment);
    if (__ptr == nullptr)
    {
        errno = ENOMEM;
    }
    return __ptr;
}

static void*
__internal_operator_new(std::size_t __size, std::size_t __alignment)
{
    void* __res = __internal_aligned_alloc(__size, __alignment);

    while (__res == nullptr)
    {
        std::new_handler __handler = std::get_new_handler();
        if (__handler != nullptr)
        {
            __handler();
        }
        else
        {
            throw std::bad_alloc{};
        }
        __res = __internal_aligned_alloc(__size, __alignment);
    }

    return __res;
}

static void*
__internal_operator_new(std::size_t __size, std::size_t __alignment, const std::nothrow_t&) noexcept
{
    void* __res = nullptr;
    try
    {
        __res = __internal_operator_new(__size, __alignment);
    }
    catch (...)
    {
    }
    return __res;
}

} // namespace __pstl_offload

#if __linux__

// valloc, pvalloc, __libc_valloc and __libc_pvalloc are not supported
// due to unsupported alignment on memory page

extern "C"
{

inline void* __attribute__((always_inline)) malloc(std::size_t __size)
{
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, alignof(std::max_align_t));
}

inline void* __attribute__((always_inline)) calloc(std::size_t __num, std::size_t __size)
{
    void* __res = nullptr;

    // Square root of maximal std::size_t value, values that are less never results in overflow during multiplication
    constexpr std::size_t __min_overflow_multiplier = std::size_t(1) << (sizeof(std::size_t) * CHAR_BIT / 2);
    std::size_t __allocate_size = __num * __size;

    // Check overflow on multiplication
    if ((__num >= __min_overflow_multiplier || __size >= __min_overflow_multiplier) &&
        (__num != 0 && __allocate_size / __num != __size))
    {
        errno = ENOMEM;
    }
    else
    {
        __res = ::__pstl_offload::__errno_handling_internal_aligned_alloc(__allocate_size, alignof(std::max_align_t));
    }

    return __res ? std::memset(__res, 0, __allocate_size) : nullptr;
}

inline void* __attribute__((always_inline)) realloc(void* __ptr, std::size_t __size)
{
    return ::__pstl_offload::__internal_realloc(__ptr, __size);
}

inline void* __attribute__((always_inline)) memalign(std::size_t __alignment, std::size_t __size) noexcept
{
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline int __attribute__((always_inline)) posix_memalign(void** __memptr, std::size_t __alignment, std::size_t __size) noexcept
{
    int __result = 0;
    if (::__pstl_offload::__is_power_of_two(__alignment))
    {
        void* __ptr = ::__pstl_offload::__internal_aligned_alloc(__size, __alignment);

        if (__ptr != nullptr)
        {
            *__memptr = __ptr;
        }
        else
        {
            __result = ENOMEM;
        }
    }
    else
    {
        __result = EINVAL;
    }
    return __result;
}

inline int __attribute__((always_inline)) mallopt(int /*param*/, int /*value*/) noexcept { return 1; }

inline void* __attribute__((always_inline)) aligned_alloc(std::size_t __alignment, std::size_t __size)
{
    return ::__pstl_offload::__errno_handling_internal_aligned_alloc(__size, __alignment);
}

inline void* __attribute__((always_inline)) __libc_malloc(std::size_t __size)
{
    return malloc(__size);
}

inline void* __attribute__((always_inline)) __libc_calloc(std::size_t __num, std::size_t __size)
{
    return calloc(__num, __size);
}

inline void* __attribute__((always_inline)) __libc_memalign(std::size_t __alignment, std::size_t __size)
{
    return memalign(__alignment, __size);
}

inline void* __attribute__((always_inline)) __libc_realloc(void* __ptr, std::size_t __size)
{
    return realloc(__ptr, __size);
}

} // extern "C"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winline-new-delete"

inline void* __attribute__((always_inline))
operator new(std::size_t __size)
{
    return ::__pstl_offload::__internal_operator_new(__size, alignof(std::max_align_t));
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size)
{
    return ::__pstl_offload::__internal_operator_new(__size, alignof(std::max_align_t));
}

inline void* __attribute__((always_inline))
operator new(std::size_t __size, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, alignof(std::max_align_t), std::nothrow);
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, alignof(std::max_align_t), std::nothrow);
}

inline void* __attribute__((always_inline))
operator new(std::size_t __size, std::align_val_t __al)
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al));
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size, std::align_val_t __al)
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al));
}

inline void* __attribute__((always_inline))
operator new(std::size_t __size, std::align_val_t __al, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al), std::nothrow);
}

inline void* __attribute__((always_inline))
operator new[](std::size_t __size, std::align_val_t __al, const std::nothrow_t&) noexcept
{
    return ::__pstl_offload::__internal_operator_new(__size, std::size_t(__al), std::nothrow);
}

#pragma GCC diagnostic pop

#endif // __linux__

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_USM_MEMORY_REPLACEMENT_H
