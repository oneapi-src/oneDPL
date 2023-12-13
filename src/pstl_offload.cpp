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

#define _PSTL_OFFLOAD_EXPORT

#elif __linux__

#include <dlfcn.h>
#include <unistd.h>

#define _PSTL_OFFLOAD_EXPORT __attribute__((visibility("default")))

#endif

#include <string.h>
#include <atomic>

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

static auto
__get_original_realloc()
{
    using __realloc_func_type = void* (*)(void*, std::size_t);

    static __realloc_func_type __orig_realloc = __realloc_func_type(dlsym(RTLD_NEXT, "realloc"));
    return __orig_realloc;
}
#elif _WIN64

static __free_func_type __original_free_ptr = free;

#endif

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
#if __linux__
        __original_free(ptr);
#elif _WIN64
        __original_free_ptr(ptr);
#endif
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

    void
    __lock()
    {
        while (_M_flag.test_and_set())
        {
            // TODO: exponential backoff or switch to wait() from c++20
            std::this_thread::yield();
        }
    }
    void
    __unlock()
    {
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
            __m.__lock();
        }

        __scoped_lock(__scoped_lock&) = delete;
        __scoped_lock& operator=(__scoped_lock&) = delete;

        ~__scoped_lock()
        {
            if (_M_taken)
            {
                _M_mutex.__unlock();
            }
        }
    };
    friend class __scoped_lock;
};

static __spin_mutex __large_aligned_ptrs_map_mtx;

class __large_aligned_ptrs_map
{
public:
    struct __ptr_desc
    {
        __sycl_device_shared_ptr _M_device_ptr;
        std::size_t _M_requested_number_of_bytes;
    };

private:
    // Find sycl::device and requested size by user pointer. Use __orig_free_allocator to not
    // call global delete during delete processing (overloaded global delete includes
    // __unregister_ptr call).
    using __map_ptr_to_object_prop =
        std::unordered_map<void*, __ptr_desc, __hash_aligned_ptr, std::equal_to<void*>,
        __orig_free_allocator<std::pair<void* const, __ptr_desc>>>;
    __map_ptr_to_object_prop* _M_map;

public:
    // We suppose that all users of libpstloffload have dependence on it, so it's impossible to register
    // >=4K-aligned USM memory before ctor of static objects executed. So, no need for special support for
    // adding to not-yet-created __large_aligned_ptrs_map.
    __large_aligned_ptrs_map() : _M_map(new __map_ptr_to_object_prop) { }

    // Intentionally leaks _M_map to able use it after static object dtor is executed. Global
    // free/delete/realloc/etc are overloaded, so we need to use it even when static object dtor
    // already run.
    ~__large_aligned_ptrs_map() { }

    void
    __register_ptr(void* __ptr, size_t __size, __sycl_device_shared_ptr __device_ptr)
    {
        assert(__is_ptr_page_aligned(__ptr));

        __spin_mutex::__scoped_lock l(__large_aligned_ptrs_map_mtx);
        [[maybe_unused]] auto __ret = _M_map->emplace(__ptr, __ptr_desc{__device_ptr, __size});
        assert(__ret.second); // the pointer must be unique
    }

    // nullopt means "it's not our pointer"
    std::optional<__ptr_desc>
    __unregister_ptr(void* __ptr)
    {
        // only page-aligned can be registered
        if (!__is_ptr_page_aligned(__ptr))
        {
            return std::nullopt;
        }

        __spin_mutex::__scoped_lock l(__large_aligned_ptrs_map_mtx);
        if (!_M_map)
        {
            // ctor of static object not yet run, so it can't be our pointer
            return std::nullopt;
        }
        auto __iter = _M_map->find(__ptr);
        if (__iter == _M_map->end())
        {
            return std::nullopt;
        }
        __ptr_desc __header = __iter->second;
        _M_map->erase(__iter);
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
        auto __iter = _M_map->find(__ptr);
        if (__iter == _M_map->end())
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
    __header->~__block_header();
    sycl::free(__header->_M_original_pointer, __context);
}

#if __linux__

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

        if (__desc.has_value())
        {
            sycl::context __context = __desc->_M_device_ptr.__get_context();
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

#elif _WIN64

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
        _invalid_parameter_noinfo();
        return -1;
#else
        return 0;
#endif
    }

    __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
    {
        return __header->_M_requested_number_of_bytes;
    }

    std::optional<std::size_t> __size = __large_aligned_ptrs.__get_size(__user_ptr);

    return __size.has_value() ? __size.value() : __original_msize(__user_ptr);
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
    if (__desc.has_value())
    {
        if (__desc->_M_requested_number_of_bytes == __new_size)
        {
            return __user_ptr;
        }

        void* __result = __realloc_allocate_shared(__desc->_M_device_ptr, __user_ptr, __desc->_M_requested_number_of_bytes, __new_size);
        if (__result)
        {
            sycl::context __context = __desc->_M_device_ptr.__get_context();
            sycl::free(__user_ptr, __context);
        }
        return __result;
    }

    // __user_ptr is not a USM pointer, use original realloc function
#if __linux__
    return __get_original_realloc()(__user_ptr, __new_size);
#elif _WIN64
    return __original_realloc(__user_ptr, __new_size);
#endif
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

#if _WIN64

void*
__aligned_realloc_impl(void* __user_ptr, std::size_t __new_size, std::size_t __alignment)
{
    assert(__user_ptr != nullptr);

    if (__new_size == 0)
    {
        _aligned_free(__user_ptr);
        return nullptr;
    }

    __block_header* __header = static_cast<__block_header*>(__user_ptr) - 1;

    void* __result = nullptr;

    if (__same_memory_page(__user_ptr, __header) && __header->_M_uniq_const == __uniq_type_const)
    {
        if (__header->_M_requested_number_of_bytes >= __new_size && (std::uintptr_t)__user_ptr % __alignment == 0)
        {
            __result = __user_ptr;
        }
        else
        {
            assert(__header->_M_device);
            void* __new_ptr = __allocate_shared_for_device(__header->_M_device, __new_size, __alignment);

            if (__new_ptr != nullptr)
            {
                std::memcpy(__new_ptr, __user_ptr, std::min(__header->_M_requested_number_of_bytes, __new_size));

                // Free previously allocated memory
                __free_usm_pointer(__header);
                __result = __new_ptr;
            }
            else
            {
                errno = ENOMEM;
            }
        }
    }
    else
    {
        // __user_ptr is not a USM pointer, use original realloc function
        __result = __original_aligned_realloc(__user_ptr, __new_size, __alignment);
    }
    return __result;
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

static std::size_t
__internal_aligned_msize(void* __user_ptr, std::size_t __alignment, std::size_t __offset)
{
    if (__user_ptr == nullptr || !__is_power_of_two(__alignment))
    {
        errno = EINVAL;
        _invalid_parameter_noinfo();
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
    static struct __system_info
    {
        SYSTEM_INFO _M_si;
        __system_info()
        {
            GetSystemInfo(&_M_si);
        }
    } __info;

    return __info._M_si.dwPageSize;
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

    if (callReason == DLL_PROCESS_ATTACH && reserved && hInst)
    {
        ret = __pstl_offload::__do_functions_replacement() ? TRUE : FALSE;
    }

    return ret;
}
#pragma GCC diagnostic pop

#endif // _WIN64
