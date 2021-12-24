// -*- C++ -*-
//===-- utils_sycl.h ------------------------------------------------------===//
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
#ifndef UTILS_SYCL
#define UTILS_SYCL

// File contains common utilities for SYCL that tests rely on

// Do not #include <algorithm>, because if we do we will not detect accidental dependencies.

#include <iterator>
#include "oneapi/dpl/pstl/hetero/dpcpp/sycl_defs.h"
#if _ONEDPL_FPGA_DEVICE
#    if __LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif

#include "test_config.h"

#include _PSTL_TEST_HEADER(iterator)
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
#include "iterator_utils.h"

#include _PSTL_TEST_HEADER(execution)

namespace TestUtils
{

#define PRINT_DEBUG(message) ::TestUtils::print_debug(message)

inline void
print_debug(const char*
#if _ONEDPL_DEBUG_SYCL
                message
#endif
)
{
#if _ONEDPL_DEBUG_SYCL
    ::std::cout << message << ::std::endl;
#endif
}

// Check values in sequence
template <typename Iterator, typename T>
bool
check_values(Iterator first, Iterator last, const T& val)
{
    return ::std::all_of(first, last, [&val](const T& x) { return x == val; });
}

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

auto async_handler = [](sycl::exception_list ex_list) {
    for (auto& ex : ex_list)
    {
        try
        {
            ::std::rethrow_exception(ex);
        }
        catch (sycl::exception& ex)
        {
            ::std::cerr << ex.what() << ::std::endl;
            ::std::exit(EXIT_FAILURE);
        }
    }
};

//function is needed to wrap kernel name into another class
template <typename _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_Policy, int> = 0>
auto
make_new_policy(_Policy&& __policy)
    -> decltype(oneapi::dpl::execution::make_device_policy<_NewKernelName>(::std::forward<_Policy>(__policy)))
{
    return oneapi::dpl::execution::make_device_policy<_NewKernelName>(::std::forward<_Policy>(__policy));
}

#if ONEDPL_FPGA_DEVICE
template <typename _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_new_policy(_Policy&& __policy)
    -> decltype(oneapi::dpl::execution::make_fpga_policy<::std::decay<_Policy>::type::unroll_factor, _NewKernelName>(
        ::std::forward<_Policy>(__policy)))
{
    return oneapi::dpl::execution::make_fpga_policy<::std::decay<_Policy>::type::unroll_factor, _NewKernelName>(
        ::std::forward<_Policy>(__policy));
}
#endif

#if ONEDPL_FPGA_DEVICE
    auto default_selector =
#    if ONEDPL_FPGA_EMULATOR
        __dpl_sycl::__fpga_emulator_selector{};
#    else
        __dpl_sycl::__fpga_selector{};
#    endif // ONEDPL_FPGA_EMULATOR

    auto&& default_dpcpp_policy =
#    if ONEDPL_USE_PREDEFINED_POLICIES
        oneapi::dpl::execution::dpcpp_fpga;
#    else
        oneapi::dpl::execution::make_fpga_policy(sycl::queue{default_selector});
#    endif // ONEDPL_USE_PREDEFINED_POLICIES
#else
    auto default_selector = sycl::default_selector{};
    auto&& default_dpcpp_policy =
#    if ONEDPL_USE_PREDEFINED_POLICIES
        oneapi::dpl::execution::dpcpp_default;
#    else
        oneapi::dpl::execution::make_device_policy(sycl::queue{default_selector});
#    endif // ONEDPL_USE_PREDEFINED_POLICIES
#endif     // ONEDPL_FPGA_DEVICE

// create the queue with custom asynchronous exceptions handler
static auto my_queue = sycl::queue(default_selector, async_handler);

// Invoke op(policy,rest...) for each possible policy.
template <::std::size_t CallNumber = 0>
struct invoke_on_all_hetero_policies
{
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
            oneapi::dpl::execution::make_fpga_policy</*unroll_factor = */ 1, kernel_name>(my_queue), op,
            ::std::forward<T>(rest)...);
#else
            oneapi::dpl::execution::make_device_policy<kernel_name>(my_queue), op, ::std::forward<T>(rest)...);
#endif
    }
};

template <typename T, typename TestName>
void
test1buffer()
{
    const sycl::queue& queue = my_queue; // usm and allocator requires queue

#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        auto sycl_deleter = [queue](T* mem) { sycl::free(mem, queue.get_context()); };
        ::std::unique_ptr<T, decltype(sycl_deleter)> inout1_first(
            (T*)sycl::malloc_shared(sizeof(T) * (max_n + inout1_offset), queue.get_device(), queue.get_context()),
            sycl_deleter);

        // 2. create a pointer at first+offset
        T* inout1_offset_first = inout1_first.get() + inout1_offset;

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(TestName(), inout1_offset_first, inout1_offset_first + n, n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        sycl::buffer<T, 1> inout1{sycl::range<1>(max_n + inout1_offset)};

        // 2. create an iterator over buffer
        auto inout1_offset_first = oneapi::dpl::begin(inout1) + inout1_offset;

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(TestName(), inout1_offset_first, inout1_offset_first + n, n);
        }
    }
}

template <typename T, typename TestName>
void
test2buffers()
{
    const sycl::queue& queue = my_queue; // usm and allocator requires queue
#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        auto sycl_deleter = [queue](T* mem) { sycl::free(mem, queue.get_context()); };
        ::std::unique_ptr<T, decltype(sycl_deleter)> inout1_first(
            (T*)sycl::malloc_shared(sizeof(T) * (max_n + inout1_offset), queue.get_device(), queue.get_context()),
            sycl_deleter);
        ::std::unique_ptr<T, decltype(sycl_deleter)> inout2_first(
            (T*)sycl::malloc_shared(sizeof(T) * (max_n + inout2_offset), queue.get_device(), queue.get_context()),
            sycl_deleter);

        // 2. create pointers at first+offset
        T* inout1_offset_first = inout1_first.get() + inout1_offset;
        T* inout2_offset_first = inout2_first.get() + inout2_offset;

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(TestName(), inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n, n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        sycl::buffer<T, 1> inout1{sycl::range<1>(max_n + inout1_offset)};
        sycl::buffer<T, 1> inout2{sycl::range<1>(max_n + inout2_offset)};

        // 2. create iterators over buffers
        auto inout1_offset_first = oneapi::dpl::begin(inout1) + inout1_offset;
        auto inout2_offset_first = oneapi::dpl::begin(inout2) + inout2_offset;

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(TestName(), inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n, n);
        }
    }
}

template <typename T, typename TestName>
void
test3buffers(int mult = 1)
{
    const sycl::queue& queue = my_queue; // usm requires queue
#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        auto sycl_deleter = [queue](T* mem) { sycl::free(mem, queue.get_context()); };
        ::std::unique_ptr<T, decltype(sycl_deleter)> inout1_first(
            (T*)sycl::malloc_shared(sizeof(T) * (max_n + inout1_offset), queue.get_device(), queue.get_context()),
            sycl_deleter);
        ::std::unique_ptr<T, decltype(sycl_deleter)> inout2_first(
            (T*)sycl::malloc_shared(sizeof(T) * (max_n + inout2_offset), queue.get_device(), queue.get_context()),
            sycl_deleter);
        ::std::unique_ptr<T, decltype(sycl_deleter)> inout3_first(
            (T*)sycl::malloc_shared(mult * sizeof(T) * (max_n + inout3_offset), queue.get_device(),
                                    queue.get_context()),
            sycl_deleter);

        // 2. create pointers at first+offset
        T* inout1_offset_first = inout1_first.get() + inout1_offset;
        T* inout2_offset_first = inout2_first.get() + inout2_offset;
        T* inout3_offset_first = inout3_first.get() + inout3_offset;

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n)))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(TestName(), inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n, inout3_offset_first,
                                               inout3_offset_first + n, n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        sycl::buffer<T, 1> inout1{sycl::range<1>(max_n + inout1_offset)};
        sycl::buffer<T, 1> inout2{sycl::range<1>(max_n + inout2_offset)};
        sycl::buffer<T, 1> inout3{sycl::range<1>(mult * max_n + inout3_offset)};

        // 2. create iterators over buffers
        auto inout1_offset_first = oneapi::dpl::begin(inout1) + inout1_offset;
        auto inout2_offset_first = oneapi::dpl::begin(inout2) + inout2_offset;
        auto inout3_offset_first = oneapi::dpl::begin(inout3) + inout3_offset;

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n)))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(TestName(), inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n, inout3_offset_first,
                                               inout3_offset_first + n, n);
        }
    }
}

// use the function carefully due to temporary accessor creation.
// Race condition between host and device may be occurred
// if we work with the buffer host memory when kernel is invoked on device
template <typename Iter, sycl::access::mode mode = sycl::access::mode::read_write>
typename ::std::iterator_traits<Iter>::pointer
get_host_pointer(Iter it)
{
    auto temp_idx = it - oneapi::dpl::begin(it.get_buffer());
    return &it.get_buffer().template get_access<mode>()[0] + temp_idx;
}

template <typename T, int Dim, sycl::access::mode AccMode, sycl::access::target AccTarget,
          sycl::access::placeholder Placeholder>
T*
get_host_pointer(sycl::accessor<T, Dim, AccMode, AccTarget, Placeholder>& acc)
{
    return &acc[0];
}

// for USM pointers
template <typename T>
T*
get_host_pointer(T* data)
{
    return data;
}

template <typename Iter, sycl::access::mode mode = sycl::access::mode::read_write>
auto
get_host_access(Iter it)
    -> decltype(it.get_buffer().template get_access<mode>(__dpl_sycl::__get_buffer_size(it.get_buffer()) -
                                                              (it - oneapi::dpl::begin(it.get_buffer())),
                                                          it - oneapi::dpl::begin(it.get_buffer())))
{
    auto temp_buf = it.get_buffer();
    auto temp_idx = it - oneapi::dpl::begin(temp_buf);
    return temp_buf.template get_access<mode>(__dpl_sycl::__get_buffer_size(temp_buf) - temp_idx, temp_idx);
}

template <typename T>
T*
get_host_access(T* data)
{
    return data;
}
} /* namespace TestUtils */
#endif
