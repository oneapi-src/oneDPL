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
#include <thread> // std::this_thread::yield
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

inline bool
__is_ptr_page_aligned(void *p)
{
    return (uintptr_t)p % __get_memory_page_size() == 0;
}

struct __hash_aligned_ptr
{
    uintptr_t operator()(void *p) const
    {
        // We know that addresses are at least page-aligned, so, expecting page 4K-aligned,
        // drop 11 right bits that are zeros, and treat rest as a pointer, hoping that
        // an underlying Standard Library support this well.
        constexpr unsigned shift = 11;
        return std::hash<void*>()((void*)((uintptr_t)p >> shift));
    }
};

template<class T>
struct __orig_free_allocator
{
    typedef T value_type;

    __orig_free_allocator() = default;

    template<class U>
    constexpr __orig_free_allocator(const __orig_free_allocator <U>&) noexcept {}

    T* allocate(std::size_t n)
    {
        if (T *ptr = static_cast<T*>(std::malloc(n * sizeof(T))))
        {
            return ptr;
        }
        throw std::bad_alloc();
    }

    void deallocate(T* ptr, std::size_t) noexcept
    {
        __original_free(ptr);
    }
};

template<class T, class U>
bool operator==(const __orig_free_allocator <T>&, const __orig_free_allocator <U>&) { return true; }
template<class T, class U>
bool operator!=(const __orig_free_allocator <T>&, const __orig_free_allocator <U>&) { return false; }

// stripped down spin mutex for placement in zero-initialized memory
class __spin_mutex
{
    std::atomic_flag _M_flag = ATOMIC_FLAG_INIT;

    void lock()
    {
        while (_M_flag.test_and_set())
        {
            std::this_thread::yield();
        }
    }
    bool try_lock()
    {
        return !_M_flag.test_and_set();
    }
    void unlock() {
        _M_flag.clear(std::memory_order_release);
    }

public:
    class __scoped_lock
    {
        __spin_mutex& _M_mutex;
        bool _M_taken;

    public:
        __scoped_lock(__spin_mutex& __m) : _M_mutex(__m), _M_taken(true)
        {
            __m.lock();
        }

        __scoped_lock(__scoped_lock&) = delete;
        __scoped_lock& operator=(__scoped_lock&) = delete;

        ~__scoped_lock()
        {
            if (_M_taken)
            {
                _M_mutex.unlock();
            }
        }
    };
    friend class __scoped_lock;
};

// Signal when __large_aligned_ptrs about to be destroyed.
// We suppose that all users of libpstloffload have dependence on it, so it's impossible to register 
// >=4K-aligned USM memory before ctor of static objects executed. So, no need for special support for
// adding to not-yet-created __large_aligned_ptrs_map.
static bool __skip_large_aligned_ptrs;

static __spin_mutex __large_aligned_ptrs_map_mtx;

struct __unreleased_ptr_desc
{
    void* _M_ptr;
    bool _M_valid;
    sycl::device _M_device;
    std::size_t   _M_requested_number_of_bytes;
};

static __unreleased_ptr_desc* __unreleased_ptrs;
static std::size_t __unreleased_ptrs_size;

class __large_aligned_ptrs_map
{
public:
    struct __ptr_desc
    {
        std::optional<__sycl_device_shared_ptr> _M_device_ptr;
        std::size_t   _M_requested_number_of_bytes;
    };

private:
    // Find sycl::device and requested size by user pointer. Use __orig_free_allocator to not
    // call global delete during delete processing (that includes __unregister_ptr call).
    std::unordered_map<void*, __ptr_desc, __hash_aligned_ptr, std::equal_to<void*>,
        __orig_free_allocator<std::pair<void* const, __ptr_desc>>> _M_map;

public:
    ~__large_aligned_ptrs_map()
    {
        __spin_mutex::__scoped_lock l(__large_aligned_ptrs_map_mtx);
        __unreleased_ptrs_size = _M_map.size();
        if (__unreleased_ptrs_size)
        {
            // call malloc to able to use __original_free after
            __unreleased_ptrs = (__unreleased_ptr_desc*)malloc(sizeof(__unreleased_ptr_desc)*__unreleased_ptrs_size);
            std::size_t __cnt = 0;
            for (auto const& [__ptr, __val] : _M_map)
            {
                __unreleased_ptrs[__cnt++] =
                    __unreleased_ptr_desc{__ptr, true, __val._M_device_ptr->__get_device(), __val._M_requested_number_of_bytes};
            }
        }

        fprintf(stderr, "size %lu\n", _M_map.size());
        __skip_large_aligned_ptrs = true;
    }

    void
    __register_ptr(void* __ptr, size_t __size, __sycl_device_shared_ptr __device_ptr)
    {
        assert(__is_ptr_page_aligned(__ptr));

        __spin_mutex::__scoped_lock l(__large_aligned_ptrs_map_mtx);
        if (__skip_large_aligned_ptrs)
        {
            // adding new element is rare operation, so implements it via re-allocation of the array
            // with only valid elements
            std::size_t __active = 1 + std::count_if(__unreleased_ptrs, __unreleased_ptrs + __unreleased_ptrs_size,
                [](const __unreleased_ptr_desc& __e){ return __e._M_valid; });
            __unreleased_ptr_desc*
                __new_unreleased_ptrs = (__unreleased_ptr_desc*)malloc(sizeof(__unreleased_ptr_desc)*__active);
            __unreleased_ptr_desc* __dest_it = __new_unreleased_ptrs;

            std::copy_if(__unreleased_ptrs, __unreleased_ptrs + __unreleased_ptrs_size, __dest_it,
                [](const __unreleased_ptr_desc& __e){ return __e._M_valid; });
            *__dest_it = __unreleased_ptr_desc{__ptr, true, __device_ptr.__get_device(), __size};
            __original_free(__unreleased_ptrs);
            __unreleased_ptrs = __new_unreleased_ptrs;
            __unreleased_ptrs_size = __active;
        }
        else
        {
            [[maybe_unused]] auto __ret = _M_map.emplace(__ptr, __ptr_desc{__device_ptr, __size});
            assert(__ret.second); // the pointer must be unique
        }
    }

    // nullopt means "status unknown", empty __ptr_desc means "it's not our pointer"
    std::optional<__ptr_desc>
    __unregister_ptr(void* __ptr)
    {
        // only page-aligned can be registered
        if (!__is_ptr_page_aligned(__ptr))
        {
            return __ptr_desc{std::nullopt, 0};
        }

        __spin_mutex::__scoped_lock l(__large_aligned_ptrs_map_mtx);
        if (__skip_large_aligned_ptrs)
        {
            auto __it =
                std::find_if(__unreleased_ptrs, __unreleased_ptrs + __unreleased_ptrs_size,
                    [=](const __unreleased_ptr_desc& __e){ return __e._M_valid && __e._M_ptr == __ptr; });
            if (__it == __unreleased_ptrs + __unreleased_ptrs_size)
            {
                return std::nullopt;
            }
            __it->_M_valid = false;
            return __ptr_desc{__it->_M_device, __it->_M_requested_number_of_bytes};
        }

        auto __iter = _M_map.find(__ptr);
        if (__iter == _M_map.end())
        {
            return __ptr_desc{std::nullopt, 0};
        }
        __ptr_desc __header = __iter->second;
        _M_map.erase(__iter);
        return __header;
    }

    // nullopt means "it's not our pointer"
    std::optional<std::size_t>
    __get_size(void* __ptr)
    {
        // only page-aligned can be registered
        if (!__is_ptr_page_aligned(__ptr))
        {
            return std::nullopt;
        }

        __spin_mutex::__scoped_lock l(__large_aligned_ptrs_map_mtx);
        if (__skip_large_aligned_ptrs)
        {
            auto __it =
                std::find_if(__unreleased_ptrs, __unreleased_ptrs + __unreleased_ptrs_size,
                    [=](const __unreleased_ptr_desc& __e){ return __e._M_valid && __e._M_ptr == __ptr; });
            return __it == __unreleased_ptrs + __unreleased_ptrs_size ? std::nullopt
                : std::optional<std::size_t>(__it->_M_requested_number_of_bytes);
        }

        auto __iter = _M_map.find(__ptr);
        if (__iter == _M_map.end())
        {
            return std::nullopt;
        }
        return __iter->second._M_requested_number_of_bytes;
    }
};

static __large_aligned_ptrs_map __large_aligned_ptrs;

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
        if (__block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;
            __same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
        {
            __free_usm_pointer(__header);
            return;
        }

        std::optional<__large_aligned_ptrs_map::__ptr_desc> __desc = __large_aligned_ptrs.__unregister_ptr(__user_ptr);

        if (!__desc.has_value())
        {
            return; // leak memory in "status unknown" case
        }
        if (__desc->_M_device_ptr.has_value())
        {
            sycl::context __context = __desc->_M_device_ptr->__get_context();
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

    std::optional<std::size_t> __size = __large_aligned_ptrs.__get_size(__user_ptr);

    return __size.has_value() ? __size.value() : __get_original_msize()(__user_ptr);
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

    if (__block_header* __header = static_cast<__block_header*>(__user_ptr) - 1; 
        __same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
    {
        if (__header->_M_requested_number_of_bytes == __new_size)
        {
            return __user_ptr;
        }

        void* __result = __realloc_allocate_shared(__header->_M_device, __user_ptr,
                                                   __header->_M_requested_number_of_bytes, __new_size);
        if (__result)
        {
            __free_usm_pointer(__header);
        }
        return __result;
    }

    std::optional<__large_aligned_ptrs_map::__ptr_desc> __desc = __large_aligned_ptrs.__unregister_ptr(__user_ptr);
    if (!__desc.has_value())
    {
        return nullptr; // can't do anything in "status unknown" case
    }
    if (__desc->_M_device_ptr.has_value())
    {
        if (__desc->_M_requested_number_of_bytes == __new_size)
        {
            return __user_ptr;
        }

        void* __result = __realloc_allocate_shared(__desc->_M_device_ptr.value(), __user_ptr, __desc->_M_requested_number_of_bytes, __new_size);
        if (__result)
        {
            sycl::context __context = __desc->_M_device_ptr->__get_context();
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
        __large_aligned_ptrs.__register_ptr(__ptr, __size, __device_ptr);
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
