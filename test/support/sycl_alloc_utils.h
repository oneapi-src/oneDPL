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

#ifndef __TEST_SYCL_ALLOC_UTILS_H
#define __TEST_SYCL_ALLOC_UTILS_H

#if TEST_DPCPP_BACKEND_PRESENT

#include <list>
#include <memory>

#include "oneapi/dpl/pstl/hetero/dpcpp/sycl_defs.h"

namespace TestUtils
{

////////////////////////////////////////////////////////////////////////////////
//
//
class usm_data_transfer_base
{
public:

    virtual sycl::usm::alloc get_alloc_type() const = 0;

    virtual bool is_addr_owner(void* __usm_ptr) const = 0;

    virtual sycl::queue& get_queue() const = 0;

    virtual void* get_usm_buf() const = 0;

    virtual size_t get_usm_buf_size() const = 0;

    virtual size_t get_usm_buf_count() const = 0;
};

////////////////////////////////////////////////////////////////////////////////
//
// RAII service class to allocate shared/device memory (USM)
// Usage model"
// 1. allocate USM memory and copying data to USM:
//    usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, first, count); 
// or 
//    usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, std::begin(data), std::end(data));
// or just allocate USM memory"
//    usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, count); 
// 2. get a USM pointer by usm_data_transfer::get_data() and passed one into a parallel algorithm with dpc++ policy.
// 3. Retrieve data back (in case of device allocation type) to the host for further checking result.
//    dt_helper.retrieve_data(dest_host);
//
// Count of elements may be zero. In this case no USM memory will be allocated.
// Also retrieve_data function will do nothing in this case.
// This behavior is implemented based on tests use cases.
//
template<sycl::usm::alloc _alloc_type, typename _ValueType>
class usm_data_transfer : public usm_data_transfer_base
{
    static_assert(_alloc_type == sycl::usm::alloc::shared || _alloc_type == sycl::usm::alloc::device,
                        "Invalid allocation type for usm_data_transfer class");

    using __difference_type = typename ::std::iterator_traits<_ValueType*>::difference_type;

    template<sycl::usm::alloc __type>
    using __alloc_type = ::std::integral_constant<sycl::usm::alloc, __type>;
    using __shared_alloc_type = __alloc_type<sycl::usm::alloc::shared>;
    using __device_alloc_type = __alloc_type<sycl::usm::alloc::device>;

    template<typename _Size>
    _ValueType* allocate( _Size __sz, __shared_alloc_type)
    {
        return sycl::malloc_shared<_ValueType>(__sz, __queue);
    }
    template<typename _Size>
    _ValueType* allocate( _Size __sz, __device_alloc_type)
    {
        return sycl::malloc_device<_ValueType>(__sz, __queue);
    }

    void srvc_register();
    void srvc_unregister();

public:

    template<typename _Size>
    usm_data_transfer(sycl::queue& __q, _Size __sz)
        : __queue(__q), __count(__sz)
    {
        if (__count > 0)
        {
            __ptr = allocate(__count, __alloc_type<_alloc_type>{});
            assert(__ptr != nullptr);

            __reg_helper = ::std::make_unique<RegisterInService>(static_cast<usm_data_transfer_base*>(this));
        }
    }

    template<typename _Iterator, typename _Size>
    usm_data_transfer(sycl::queue& __q, _Iterator __it, _Size __sz)
        : usm_data_transfer(__q, __sz)
    {
        if (__count > 0)
        {
            //TODO: support copying data provided by non-contiguous iterator
            auto __src = std::addressof(*__it);
            assert(std::addressof(*(__it + __count)) - __src == __count);

            __queue.copy(__src, __ptr, __count);
            __queue.wait();
        }
    }

    template<typename _Iterator>
    usm_data_transfer(sycl::queue& __q, _Iterator __itBegin, _Iterator __itEnd)
        : usm_data_transfer(__q, __itBegin, __itEnd - __itBegin)
    {
    }

    ~usm_data_transfer()
    {
        if (__count > 0)
        {
            assert(__ptr != nullptr);

            __reg_helper.reset();
            
            sycl::free(__ptr, __queue);
        }
    }

    _ValueType* get_data() const
    {
        return __ptr;
    }

    template<typename _Iterator>
    void retrieve_data(_Iterator __it)
    {
        if (__count > 0)
        {
            assert(__ptr != nullptr);

            //TODO: support copying data provided by non-contiguous iterator
            auto __dst = std::addressof(*__it);
            assert(std::addressof(*(__it + __count)) - __dst == __count);

            __queue.copy(__ptr, __dst, __count);
            __queue.wait();
        }
    }

    // Service functions

    virtual sycl::usm::alloc get_alloc_type() const override
    {
        return _alloc_type;
    }

    virtual bool is_addr_owner(void* __usm_ptr) const override
    {
        return __count > 0 && __ptr <= __usm_ptr && __usm_ptr < (__ptr + __count);
    }

    virtual sycl::queue& get_queue() const override
    {
        return __queue;
    }

    virtual void* get_usm_buf() const override
    {
        return __ptr;
    }

    virtual size_t get_usm_buf_size() const override
    {
        return __count * sizeof(_ValueType);
    }

    virtual size_t get_usm_buf_count() const override
    {
        return __count;
    }

private:

    sycl::queue& __queue;
    __difference_type __count = 0;
    _ValueType* __ptr = nullptr;

    struct RegisterInService
    {
        usm_data_transfer_base* p_usm_data_transfer_base = nullptr;

        RegisterInService(usm_data_transfer_base* __ptr);
        ~RegisterInService();
    };

    ::std::unique_ptr<RegisterInService> __reg_helper;
};

////////////////////////////////////////////////////////////////////////////////
//
//
class usm_data_transfer_service
{
public:

    static usm_data_transfer_service* instance();

    void register_usm_data_transfer(usm_data_transfer_base* __ptr);
    void unregister_usm_data_transfer(usm_data_transfer_base* __ptr);

    usm_data_transfer_base* get_usm_data_transfer_base(void* __usm_ptr);

    // for USM pointers
    template <typename T>
    T* get_host_pointer(usm_data_transfer_base* pUsmDataTransferBase, T* data) const;

    template <typename T>
    void refresh_usm_from_host_pointer(usm_data_transfer_base* pUsmDataTransferBase, T* __host_ptr, T* __usm_ptr, ::std::size_t __count);

private:

    void* alloc_host_mem(usm_data_transfer_base* pUsmDataTransferBase, size_t size);

    usm_data_transfer_base* find_get_usm_data_transfer_base(void* __usm_ptr) const;

    struct dt_info;
    dt_info* find_dt_info(usm_data_transfer_base* pUsmDataTransferBase) const;

private:

    struct host_mem_info
    {
        void* __host_buf = nullptr;
        ::std::size_t __host_buf_size = 0;

        // Enable move constructor and move assignment, disable copy constructor and copy assignment
        host_mem_info() = default;
        host_mem_info(void* p, ::std::size_t sz)
            : __host_buf(p), __host_buf_size(sz)
        {
        }

        host_mem_info(const host_mem_info&) = delete;
        host_mem_info(host_mem_info&& other)
        {
            std::swap(__host_buf, other.__host_buf);
            std::swap(__host_buf_size, other.__host_buf_size);
        }

        host_mem_info& operator=(const host_mem_info&) = delete;
        host_mem_info& operator=(host_mem_info&&) = delete;

        ~host_mem_info();
    };

    struct dt_info
    {
        usm_data_transfer_base* __ptr = nullptr;
        std::list<host_mem_info> __host_mem;

        // Enable move constructor and move assignment, disable copy constructor and copy assignment
        dt_info() = default;
        dt_info(usm_data_transfer_base* __p)
            : __ptr(__p)
        {
        }
        dt_info(const dt_info&) = delete;
        dt_info(dt_info&& other)
        {
            std::swap(__ptr, other.__ptr);
            std::swap(__host_mem, other.__host_mem);
        }

        dt_info& operator=(const dt_info&) = delete;
        dt_info& operator=(dt_info&&) = delete;

        void* alloc_host_mem(size_t size);
        const host_mem_info* find_host_mem_info(void* __ptr) const;
    };

    std::list<dt_info> __data;
};

namespace
{
    template <typename T>
    ::std::byte* get_byte_ptr(T* ptr)
    {
        return reinterpret_cast<::std::byte*>(ptr);
    }

    template <typename T, typename Data>
    T* get_t_ptr(Data* pData)
    {
        return reinterpret_cast<T*>(pData);
    }
};

//----------------------------------------------------------------------------//
template<sycl::usm::alloc _alloc_type, typename _ValueType>
usm_data_transfer<_alloc_type, _ValueType>::RegisterInService::RegisterInService(usm_data_transfer_base* __ptr)
    : p_usm_data_transfer_base(__ptr)
{
    usm_data_transfer_service* srvc = usm_data_transfer_service::instance();
    assert(srvc);

    srvc->register_usm_data_transfer(p_usm_data_transfer_base);
}

//----------------------------------------------------------------------------//
template<sycl::usm::alloc _alloc_type, typename _ValueType>
usm_data_transfer<_alloc_type, _ValueType>::RegisterInService::~RegisterInService()
{
    usm_data_transfer_service* srvc = usm_data_transfer_service::instance();
    assert(srvc);

    srvc->unregister_usm_data_transfer(p_usm_data_transfer_base);
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
void
usm_data_transfer<_alloc_type, _ValueType>::srvc_register()
{
    usm_data_transfer_service* srvc = usm_data_transfer_service::instance();
    assert(srvc);

    auto base = static_cast<usm_data_transfer_base*>(this);
    srvc->register_usm_data_transfer(base);
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
void
usm_data_transfer<_alloc_type, _ValueType>::srvc_unregister()
{
    usm_data_transfer_service* srvc = usm_data_transfer_service::instance();
    assert(srvc);

    auto base = static_cast<usm_data_transfer_base*>(this);  
    srvc->unregister_usm_data_transfer(base);
}

//----------------------------------------------------------------------------//
inline
usm_data_transfer_service*
usm_data_transfer_service::instance()
{
    static std::unique_ptr<usm_data_transfer_service> srvc = ::std::make_unique<usm_data_transfer_service>();

    return srvc.get();
}

//----------------------------------------------------------------------------//
void
usm_data_transfer_service::register_usm_data_transfer(usm_data_transfer_base* __ptr)
{
    assert(__ptr != nullptr);

    __data.emplace_back(dt_info(__ptr));
}

//----------------------------------------------------------------------------//
void
usm_data_transfer_service::unregister_usm_data_transfer(usm_data_transfer_base* __ptr)
{
    assert(__ptr != nullptr);

    __data.remove_if([__ptr](const dt_info& info) { return info.__ptr == __ptr; });
}

//----------------------------------------------------------------------------//
usm_data_transfer_base*
usm_data_transfer_service::get_usm_data_transfer_base(void* __usm_ptr)
{
    return find_get_usm_data_transfer_base(__usm_ptr);
}

//----------------------------------------------------------------------------//
// for USM pointers
template <typename T>
T*
usm_data_transfer_service::get_host_pointer(
    usm_data_transfer_base* pUsmDataTransferBase, T* data) const
{
    //return data;

    assert(sycl::usm::alloc::device == pUsmDataTransferBase->get_alloc_type());
    assert(data != nullptr);

    if (dt_info* p_dt_info = find_dt_info(pUsmDataTransferBase))
    {
        const auto buf_size = pUsmDataTransferBase->get_usm_buf_size();
        assert(buf_size > 0);

        void* host_mem = nullptr;
        if (p_dt_info->__host_mem.empty())
        {
            host_mem = p_dt_info->alloc_host_mem(buf_size);
            assert(host_mem != nullptr);
        }
        else
        {

            const auto& hmInfo = p_dt_info->__host_mem.front();
            assert(buf_size == hmInfo.__host_buf_size);

            host_mem = hmInfo.__host_buf;
        }

        // Copy data from USM-allocated memory to host memory
        sycl::queue& queue = pUsmDataTransferBase->get_queue();
        const auto count = pUsmDataTransferBase->get_usm_buf_count();
        queue.copy(get_t_ptr<T>(pUsmDataTransferBase->get_usm_buf()), get_t_ptr<T>(host_mem), count);
        queue.wait();

        // Calculate and return pointer to host memory buffer
        const auto usm_buf_offset = get_byte_ptr(data) - get_byte_ptr(pUsmDataTransferBase->get_usm_buf());
        assert(usm_buf_offset >= 0);

        return get_t_ptr<T>(get_byte_ptr(host_mem) + usm_buf_offset);
    }

    assert(!"Invalid value of pUsmDataTransferBase param");

    return nullptr;
}

//----------------------------------------------------------------------------//
template <typename T>
void
usm_data_transfer_service::refresh_usm_from_host_pointer(
    usm_data_transfer_base* pUsmDataTransferBase, T* __host_ptr, T* __usm_ptr, ::std::size_t __count)
{
    assert(__host_ptr != nullptr);
    assert(__usm_ptr != nullptr);
    assert(__count > 0);
    assert(pUsmDataTransferBase != nullptr);

    sycl::queue& queue = pUsmDataTransferBase->get_queue();
    queue.copy(__host_ptr, get_t_ptr<T>(get_byte_ptr(__usm_ptr)), __count);
    queue.wait();
}

//----------------------------------------------------------------------------//
usm_data_transfer_service::host_mem_info::~host_mem_info()
{
    delete [] get_byte_ptr(__host_buf);
}

//----------------------------------------------------------------------------//
void*
usm_data_transfer_service::alloc_host_mem(
    usm_data_transfer_base* pUsmDataTransferBase, size_t size)
{
    assert(pUsmDataTransferBase != nullptr);
    assert(size > 0);

    if (dt_info* p_dt_info = find_dt_info(pUsmDataTransferBase))
        return p_dt_info->alloc_host_mem(size);

    assert(!"Invalid value of pUsmDataTransferBase param");

    return nullptr;
}

//----------------------------------------------------------------------------//
const usm_data_transfer_service::host_mem_info*
usm_data_transfer_service::dt_info::find_host_mem_info(void* __ptr) const
{
    assert(__ptr != nullptr);

    const host_mem_info* p_host_mem_info = nullptr;

    auto it = ::std::find_if(__host_mem.begin(), __host_mem.end(),
                             [__ptr](const host_mem_info& info)
                             {
                                 return info.__host_buf <= __ptr && __ptr <= get_byte_ptr(info.__host_buf) + info.__host_buf_size;
                             });
    if (it != __host_mem.end())
    {
        p_host_mem_info = ::std::addressof(*it);
    }

    return p_host_mem_info;
}

//----------------------------------------------------------------------------//
usm_data_transfer_base*
usm_data_transfer_service::find_get_usm_data_transfer_base(void* __usm_ptr) const
{
    auto it = ::std::find_if(__data.begin(), __data.end(),
                             [__usm_ptr](const dt_info& info) { return info.__ptr->is_addr_owner(__usm_ptr); });

    if (it != __data.end())
        return it->__ptr;

    return nullptr;
}

//----------------------------------------------------------------------------//
usm_data_transfer_service::dt_info*
usm_data_transfer_service::find_dt_info(usm_data_transfer_base* pUsmDataTransferBase) const
{
    dt_info* p_dt_info = nullptr;

    auto it = ::std::find_if(__data.begin(), __data.end(),
                             [pUsmDataTransferBase](const dt_info& info) { return info.__ptr == pUsmDataTransferBase; });
    if (it != __data.end())
    {
        p_dt_info = (dt_info*)::std::addressof(*it);
    }

    return p_dt_info;
}

//----------------------------------------------------------------------------//
void*
usm_data_transfer_service::dt_info::alloc_host_mem(size_t size)
{
    assert(size > 0);

    auto ptr = new ::std::byte[size];
    assert(ptr != nullptr);

    __host_mem.emplace_back(ptr, size);

    return ptr;
}

//----------------------------------------------------------------------------//

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
