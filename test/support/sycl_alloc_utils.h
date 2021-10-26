// -*- C++ -*-
//===-- sycl_alloc_utils.h ------------------------------------------------===//
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

#ifdef TEST_DPCPP_BACKEND_PRESENT

#include <memory>
#include <algorithm>

#include <CL/sycl.hpp>

namespace TestUtils
{
    namespace
    {
        struct DeviceTag{ };
        struct SharedTag{ };

        template<typename T>
        void
        copy_from_host_impl(DeviceTag, sycl::queue& q, const T* src_ptr, T* dest_ptr, size_t count)
        {
            q.copy(src_ptr, dest_ptr, count);
            q.wait();
        }

        //template <typename T>
        //void
        //copy_from_host_impl(SharedTag, sycl::queue& q, const T* src_ptr, T* dest_ptr, size_t count)
        //{
        //    static_assert(false, "!!!");
        //}

        template <typename T>
        void
        copy_to_host_impl(DeviceTag, sycl::queue& q, const T* src_ptr, T* dest_ptr, size_t count)
        {
            q.copy(src_ptr, dest_ptr, count);
            q.wait();
        }

        //template <typename T>
        //void
        //copy_to_host_impl(SharedTag, sycl::queue& q, const T* src_ptr, T* dest_ptr, size_t count)
        //{
        //    static_assert(false, "!!!");
        //}
    }

    template <typename Op, sycl::usm::alloc alloc_type>
    using unique_kernel_name = oneapi::dpl::__par_backend_hetero::__unique_kernel_name<Op, static_cast<::std::size_t>(alloc_type)>;

    template <sycl::usm::alloc alloc_type, typename T>
    class sycl_operations_helper
    {
    private:

        sycl::queue my_queue;

        template <typename _T>
        struct __sycl_deleter
        {
            const sycl::queue q;

            void
            operator()(_T* __memory) const
            {
                sycl::free(__memory, q.get_context());
            }
        };

    public:

        static_assert(alloc_type == sycl::usm::alloc::shared || alloc_type == sycl::usm::alloc::device,
                      "Invalid state of alloc_type param in sycl_operations_helper class");

        template <typename _T = T>
        using unique_ptr = ::std::unique_ptr<_T, __sycl_deleter<_T>>;

        sycl_operations_helper(sycl::queue q) : my_queue(std::move(q))
        {
        }

        T*
        alloc(size_t count)
        {
            if constexpr (alloc_type == sycl::usm::alloc::shared)
                return sycl::malloc_shared<T>(count, my_queue.get_device(), my_queue.get_context());

            assert(alloc_type == sycl::usm::alloc::device);
            return sycl::malloc_device<T>(count, my_queue.get_device(), my_queue.get_context());
        }

        unique_ptr<T>
        alloc_ptr(size_t count)
        {
            return unique_ptr<T>(alloc(count), __sycl_deleter<T>{my_queue});
        }

        void
        copy_from_host(const T* src_ptr, T* dest_ptr, size_t count)
        {
            if constexpr (alloc_type == sycl::usm::alloc::shared)
                copy_from_host_impl(SharedTag(), my_queue, src_ptr, dest_ptr, count);
            else
                copy_from_host_impl(DeviceTag(), my_queue, src_ptr, dest_ptr, count);
        }

        void
        copy_to_host(const T* src_ptr, T* dest_ptr, size_t count)
        {
            if constexpr (alloc_type == sycl::usm::alloc::shared)
                copy_to_host_impl(SharedTag(), my_queue, src_ptr, dest_ptr, count);
            else
                copy_to_host_impl(DeviceTag(), my_queue, src_ptr, dest_ptr, count);
        }
    };

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
