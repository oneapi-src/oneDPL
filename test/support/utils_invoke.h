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

#ifndef _UTILS_INVOKE_H
#define _UTILS_INVOKE_H

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
struct unique_kernel_name;

template <typename Policy, int idx>
using new_kernel_name = unique_kernel_name<typename ::std::decay<Policy>::type, idx>;

/**
 * make_policy functions test wrappers
 * The main purpose of this function wrapper in TestUtils namespace - to cut template params from
 * oneapi::dpl::execution::device_policy function calls depend on TEST_EXPLICIT_KERNEL_NAMES macro state.
 * 
 * ATTENTION: Please avoid using oneapi::dpl::execution::device_policy directly in the tests.
 */
template <typename KernelName = oneapi::dpl::execution::DefaultKernelName, typename Arg>
inline auto
make_device_policy(Arg&& arg)
{
#if TEST_EXPLICIT_KERNEL_NAMES
    return oneapi::dpl::execution::make_device_policy<KernelName>(::std::forward<Arg>(arg));
#else
    return oneapi::dpl::execution::make_device_policy(::std::forward<Arg>(arg));
#endif // TEST_EXPLICIT_KERNEL_NAMES
}

#if _ONEDPL_FPGA_DEVICE
/**
 * make_fpga_policy functions test wrappers
 * The main purpose of this function wrapper in TestUtils namespace - to cut template params from
 * oneapi::dpl::execution::device_policy function calls depend on TEST_EXPLICIT_KERNEL_NAMES macro state.
 * 
 * ATTENTION: Please avoid using oneapi::dpl::execution::make_fpga_policy directly in tests.
 */
template <unsigned int unroll_factor = 1, typename KernelName = oneapi::dpl::execution::DefaultKernelNameFPGA, typename Arg>
inline auto
make_fpga_policy(Arg&& arg)
{
#if TEST_EXPLICIT_KERNEL_NAMES
    return oneapi::dpl::execution::make_fpga_policy<unroll_factor, KernelName>(::std::forward<Arg>(arg));
#else
    return oneapi::dpl::execution::make_fpga_policy<unroll_factor>(::std::forward<Arg>(arg));
#endif // TEST_EXPLICIT_KERNEL_NAMES
}
#endif // _ONEDPL_FPGA_DEVICE

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
        invoke_on_all_iterator_types()(seq,       op, rest...);
        invoke_on_all_iterator_types()(unseq,     op, rest...);
        invoke_on_all_iterator_types()(par,       op, rest...);
        invoke_on_all_iterator_types()(par_unseq, op, ::std::forward<T>(rest)...);
#endif
    }
};

#if TEST_DPCPP_BACKEND_PRESENT

// Implemented in utils_sycl.h, required to include this file.
sycl::queue get_test_queue();

////////////////////////////////////////////////////////////////////////////////
// check fp16/fp64 support by a device
template<typename T>
bool has_type_support(const sycl::device&) { return true; }

template<>
inline bool has_type_support<double>(const sycl::device& device)
{
    return device.has(sycl::aspect::fp64);
}

template<>
inline bool has_type_support<sycl::half>(const sycl::device& device)
{
    return device.has(sycl::aspect::fp16);
}

template <typename T, typename = void>
struct value_type
{
    using type = T;
};

// TODO: add a specialization for zip_iterator
template <typename T>
struct value_type<T, ::std::void_t<typename ::std::iterator_traits<T>::iterator_category>>
{
    using type = typename ::std::iterator_traits<T>::value_type;
};

template<typename... Ts>
bool has_types_support(const sycl::device& device)
{
    return (... && has_type_support<typename value_type<Ts>::type>(device));
}

inline void unsupported_types_notifier(const sycl::device& device)
{
    static bool is_notified = false;
    if(!is_notified)
    {
        ::std::cout << device.template get_info<sycl::info::device::name>()
                    << " does not support fp64 (double) or fp16 (sycl::half) types,"
                    << " affected test cases have been skipped\n";
        is_notified = true;
    }
}

// Invoke test::operator()(policy,rest...) for each possible policy.
template <::std::size_t CallNumber = 0>
struct invoke_on_all_hetero_policies
{
    sycl::queue queue;

    invoke_on_all_hetero_policies(sycl::queue _queue = get_test_queue())
        : queue(_queue)
    {
    }

    template <typename Op, typename... Args>
    void
    operator()(Op op, Args&&... rest)
    {
        // Device may not support some types, e.g. double or sycl::half; test if they are supported or skip otherwise
        if (has_types_support<::std::decay_t<Args>...>(queue.get_device()))
        {
            // Since make_device_policy need only one parameter for instance, this alias is used to create unique type
            // of kernels from operator type and ::std::size_t
            // There may be an issue when there is a kernel parameter which has a pointer in its name.
            // For example, param<int*>. In this case the runtime interpreters it as a memory object and
            // performs some checks that fail. As a workaround, define for functors which have this issue
            // __functor_type(see kernel_type definition) type field which doesn't have any pointers in it's name.
            using kernel_name = unique_kernel_name<Op, CallNumber>;
            auto my_policy =
#if ONEDPL_FPGA_DEVICE
                TestUtils::make_fpga_policy</*unroll_factor = */ 1, kernel_name>(queue);
#else
                TestUtils::make_device_policy<kernel_name>(queue);
#endif
            iterator_invoker<::std::random_access_iterator_tag, /*IsReverse*/ ::std::false_type>()(
                my_policy, op, ::std::forward<Args>(rest)...);
        }
        else
        {
            unsupported_types_notifier(queue.get_device());
        }
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
#if TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_host_policies()(op, rest...);
        invoke_on_all_hetero_policies<CallNumber>()(op, ::std::forward<T>(rest)...);
#else
        invoke_on_all_host_policies()(op, ::std::forward<T>(rest)...);
#endif // TEST_DPCPP_BACKEND_PRESENT
    }
};

} /* namespace TestUtils */

#endif // _UTILS_INVOKE_H
