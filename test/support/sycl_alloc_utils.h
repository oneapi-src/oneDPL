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
class usm_data_transfer
{
    static_assert(_alloc_type == sycl::usm::alloc::shared || _alloc_type == sycl::usm::alloc::device,
                        "Invalid allocation type for usm_data_transfer class");

    using __difference_type = typename ::std::iterator_traits<_ValueType*>::difference_type;

    template<sycl::usm::alloc __type>
    using __alloc_type = ::std::integral_constant<sycl::usm::alloc, __type>;
    using __shared_alloc_type = __alloc_type<sycl::usm::alloc::shared>;
    using __device_alloc_type = __alloc_type<sycl::usm::alloc::device>;

    template<typename _Size>
    _ValueType* allocate( _Size __sz, __shared_alloc_type);

    template<typename _Size>
    _ValueType* allocate( _Size __sz, __device_alloc_type);

public:

    /// Constructor
    /**
     * @param sycl::queue __q - sycl queue
     * @param _Size __sz - objects count
     */
    template<typename _Size>
    usm_data_transfer(sycl::queue __q, _Size __sz);

    /// Constructor
    /**
     * @param sycl::queue __q - sycl queue
     * @param_Iterator __it - source iterator to copy from
     * @param _Size __sz - objects count
     */
    template<typename _Iterator, typename _Size>
    usm_data_transfer(sycl::queue __q, _Iterator __it, _Size __sz);

    /// Constructor
    /**
     * @param sycl::queue __q - sycl queue
     * @param_Iterator __it_from - source iterator to copy from
     * @param_Iterator __it_to - source iterator to copy till
     */
    template<typename _Iterator>
    usm_data_transfer(sycl::queue __q, _Iterator __it_from, _Iterator __it_to);

    /// Destructor
    ~usm_data_transfer();

    /// Get USM pointer
    /**
     * @returun _ValueType* - pointer to USM shared/device allocated memory
     */
    _ValueType* get_data() const;

    /// Copy data to host buffer
    /**
     * Copy data from USM shared/device allocated memory into host buffer
     *
     *  @param _Iterator __it - pointer to host buffer.
     */
    template<typename _Iterator>
    void retrieve_data(_Iterator __it);

private:

    sycl::queue  __queue;               //< SYCL queue
    __difference_type __count = 0;      //< Count of objects in SYCL memory
    _ValueType* __ptr = nullptr;        //< Pointer to USM shared/device allocated memory
};

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
template <typename _Size>
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::usm_data_transfer(sycl::queue __q, _Size __sz)
    : __queue(__q), __count(__sz)
{
    if (__count > 0)
    {
        __ptr = allocate(__count, __alloc_type<_alloc_type>{});
        assert(__ptr != nullptr);
    }
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
template <typename _Iterator, typename _Size>
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::usm_data_transfer(sycl::queue __q, _Iterator __it, _Size __sz)
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

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
template <typename _Iterator>
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::usm_data_transfer(sycl::queue __q, _Iterator __it_from,
                                                                         _Iterator __it_to)
    : usm_data_transfer(__q, __it_from, __it_to - __it_from)
{
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::~usm_data_transfer()
{
    if (__count > 0)
    {
        assert(__ptr != nullptr);
        sycl::free(__ptr, __queue);
    }
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
template <typename _Size>
_ValueType*
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::allocate(_Size __sz, __shared_alloc_type)
{
    return sycl::malloc_shared<_ValueType>(__sz, __queue);
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
template <typename _Size>
_ValueType*
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::allocate(_Size __sz, __device_alloc_type)
{
    return sycl::malloc_device<_ValueType>(__sz, __queue);
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
_ValueType*
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::get_data() const
{
    return __ptr;
}

//----------------------------------------------------------------------------//
template <sycl::usm::alloc _alloc_type, typename _ValueType>
template <typename _Iterator>
void
TestUtils::usm_data_transfer<_alloc_type, _ValueType>::retrieve_data(_Iterator __it)
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

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
