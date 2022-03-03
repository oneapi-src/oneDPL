// -*- C++ -*-
//===-- utils_invoke.h ----------------------------------------------------===//
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

#ifndef UTILS_INVOKE
#define UTILS_INVOKE

#include <type_traits>

#include "iterator_utils.h"

namespace TestUtils
{
#if TEST_DPCPP_BACKEND_PRESENT

template <sycl::usm::alloc alloc_type>
constexpr ::std::size_t
uniq_kernel_index()
{
    return static_cast<typename ::std::underlying_type<sycl::usm::alloc>::type>(alloc_type);
}

template <typename Op, ::std::size_t CallNumber>
using unique_kernel_name = oneapi::dpl::__par_backend_hetero::__unique_kernel_name<Op, CallNumber>;

template <typename Policy, int idx>
using new_kernel_name = oneapi::dpl::__par_backend_hetero::__new_kernel_name<Policy, idx>;

#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
// Invoke op(policy,rest...) for each non-hetero policy.
struct invoke_on_all_host_policies
{
    template <typename Op, typename... T>
    void
    operator()(Op op, T&&... rest)
    {
        using namespace oneapi::dpl::execution;

#if !TEST_ONLY_HETERO_POLICIES
        // Try static execution policies
        invoke_on_all_iterator_types()(seq,       op, ::std::forward<T>(rest)...);
        invoke_on_all_iterator_types()(unseq,     op, ::std::forward<T>(rest)...);
        invoke_on_all_iterator_types()(par,       op, ::std::forward<T>(rest)...);
        invoke_on_all_iterator_types()(par_unseq, op, ::std::forward<T>(rest)...);
#endif
    }
};

#if TEST_DPCPP_BACKEND_PRESENT

// Implemented in utils_sycl.h, required to include this file.
sycl::queue get_test_queue();

////////////////////////////////////////////////////////////////////////////////
// Invoke op(policy,rest...) for each possible policy.
template <::std::size_t CallNumber = 0>
struct invoke_on_all_hetero_policies
{
    sycl::queue queue;

    invoke_on_all_hetero_policies(sycl::queue _queue = get_test_queue())
        : queue(_queue)
    {
    }

    template <typename Op, typename... T>
    void
    operator()(Op op, T&&... rest)
    {
        //Since make_device_policy need only one parameter for instance, this alias is used to create unique type
        //of kernels from operator type and ::std::size_t
        // There may be an issue when there is a kernel parameter which has a pointer in its name.
        // For example, param<int*>. In this case the runtime interpreters it as a memory object and
        // performs some checks that fails. As a workaround, define for functors which have this issue
        // __functor_type(see kernel_type definition) type field which doesn't have any pointers in it's name.
        using kernel_name = unique_kernel_name<Op, CallNumber>;
        iterator_invoker<::std::random_access_iterator_tag, /*IsReverse*/ ::std::false_type>()(
#if ONEDPL_FPGA_DEVICE
            oneapi::dpl::execution::make_fpga_policy</*unroll_factor = */ 1, kernel_name>(queue), op,
            ::std::forward<T>(rest)...);
#else
            oneapi::dpl::execution::make_device_policy<kernel_name>(queue), op, ::std::forward<T>(rest)...);
#endif
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
template <::std::size_t CallNumber = 0>
struct invoke_on_all_policies
{
    template <typename Op, typename... T>
    void
    operator()(Op op, T&&... rest)
    {
        invoke_on_all_host_policies()(op, ::std::forward<T>(rest)...);
#if TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_hetero_policies<CallNumber>()(op, ::std::forward<T>(rest)...);
#endif
    }
};

} /* namespace TestUtils */

#endif // UTILS_INVOKE
