// -*- C++ -*-
//===-- iterator_impl.h -------------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#ifndef _DPSTD_iterator_impl_H
#define _DPSTD_iterator_impl_H

namespace oneapi
{
namespace dpl
{
namespace internal
{

// Helper struct to extract sycl_iterator types needed to construct accessors
template <typename Iterator>
struct extract_accessor
{
    using accessor_type = Iterator;

    static accessor_type
    get(Iterator& i)
    {
        return i;
    }
};

#if _PSTL_BACKEND_SYCL
// Specialization for sycl_iterator to provide access to its component types needed to
// construct the accessor type
template <cl::sycl::access::mode Mode, typename T, typename Allocator>
struct extract_accessor<dpstd::__internal::sycl_iterator<Mode, T, Allocator>>
{
    static constexpr cl::sycl::access::mode mode = Mode;
    static constexpr int dim = 1;
    using buffer_type = cl::sycl::buffer<T, dim, Allocator>;
    using accessor_type = cl::sycl::accessor<T, dim, mode, cl::sycl::access::target::host_buffer>;

    static accessor_type
    get(dpstd::__internal::sycl_iterator<Mode, T, Allocator>& iter)
    {
        return iter.get_buffer().template get_access<mode>();
    }
};
#endif
} // namespace internal
} // end namespace dpl
} // end namespace oneapi

#endif /* __DPSTD_iterator_impl_H */
