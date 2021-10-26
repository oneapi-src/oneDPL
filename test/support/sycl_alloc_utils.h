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
            {
                assert(!"We shoudn't copy data from host to USM shared memory");
                //::std::copy_n(src_ptr, count, dest_ptr);
            }
            else
            {
                assert(alloc_type == sycl::usm::alloc::device);
                my_queue.copy(src_ptr, dest_ptr, count);
                my_queue.wait();
            }
        }

        void
        copy_to_host(const T* src_ptr, T* dest_ptr, size_t count)
        {
            if constexpr (alloc_type == sycl::usm::alloc::shared)
            {
                assert(!"We shoudn't copy data from USM shared memory to host");
                //::std::copy_n(src_ptr, count, dest_ptr);
            }
            else
            {
                assert(alloc_type == sycl::usm::alloc::device);
                my_queue.copy(src_ptr, dest_ptr, count);
                my_queue.wait();
            }
        }
    };

} // namespace TestUtils

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // __TEST_SYCL_ALLOC_UTILS_H
