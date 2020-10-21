// -*- C++ -*-
//===-- utils_sycl.h ------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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
#include <CL/sycl.hpp>
#if _PSTL_FPGA_DEVICE
#    include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

#include "pstl_test_config.h"

#include "oneapi/dpl/iterator"
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
#include "iterator_utils.h"

#include _PSTL_TEST_HEADER(execution)

namespace TestUtils
{

#define PRINT_DEBUG(message) ::TestUtils::print_debug(message)

    inline void
    print_debug(const char* message)
    {
    }

    template <typename T>
    struct has_functor_type {
        struct __res { char p[2]; };

        template <typename P>
        static __res __test(...);

        template <typename P>
        static char __test(typename P::__functor_type*);

        static constexpr bool value = sizeof(__test<T>(0)) == sizeof(char);
    };

    // If type T has __functor_type field, use it as a kernel name
    // otherwise use the type itself.
    template <typename T, bool has_type = has_functor_type<T>::value>
    struct kernel_type {
        using type = T;
    };

    template <typename T>
    struct kernel_type<T, true> {
        using type = typename T::__functor_type;
    };

    // Check values in sequence

    template<typename Iterator, typename T>
    bool check_values(Iterator first, Iterator last, const T& val)
    {
        return ::std::all_of(first, last,
            [&val](const T& x) { return x == val; });
    }

    template<typename Op, ::std::size_t CallNumber>
    struct unique_kernel_name {};

    template<typename Policy, int idx>
    using new_kernel_name = unique_kernel_name<typename ::std::decay<Policy>::type, idx>;

    auto async_handler = [](cl::sycl::exception_list ex_list) {
        for (auto& ex : ex_list) {
            try {
                ::std::rethrow_exception(ex);
            }
            catch (cl::sycl::exception& ex) {
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

#if _PSTL_FPGA_DEVICE
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

#if _PSTL_FPGA_DEVICE
    auto& default_dpcpp_policy = oneapi::dpl::execution::dpcpp_fpga;
    auto default_selector =
#if _PSTL_FPGA_EMU
        cl::sycl::INTEL::fpga_emulator_selector{};
#else
        cl::sycl::INTEL::fpga_selector{};
#endif
#else
    auto& default_dpcpp_policy = oneapi::dpl::execution::dpcpp_default;
    auto default_selector = cl::sycl::default_selector{};
#endif

    // create the queue with custom asynchronous exceptions handler
    static auto my_queue = cl::sycl::queue(default_selector, async_handler);

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
            using kernel_name = unique_kernel_name<typename kernel_type<Op>::type, CallNumber>;
            iterator_invoker<::std::random_access_iterator_tag, /*IsReverse*/ ::std::false_type>()(
#if _PSTL_FPGA_DEVICE
                oneapi::dpl::execution::make_fpga_policy</*unroll_factor = */ 1, kernel_name>(my_queue), op, ::std::forward<T>(rest)...);
#else
                oneapi::dpl::execution::make_device_policy<kernel_name>(my_queue), op, ::std::forward<T>(rest)...);
#endif
        }
    };

    // Test buffers
    const int max_n = 100000;
    const int inout1_offset = 3;
    const int inout2_offset = 5;
    const int inout3_offset = 7;

    template <typename T, typename TestName>
    void
    test1buffer()
    {
        const cl::sycl::queue& queue = my_queue; // usm and allocator requires queue

#if _PSTL_SYCL_TEST_USM
        { // USM
            // 1. allocate usm memory
            auto sycl_deleter = [&queue](T* mem) { cl::sycl::free(mem, queue.get_context()); };
            ::std::unique_ptr<T, decltype(sycl_deleter)> inout1_first(
                (T*)cl::sycl::malloc_shared(sizeof(T)*(max_n + inout1_offset), queue.get_device(), queue.get_context()),
                sycl_deleter);

            // 2. create a pointer at first+offset
            T* inout1_offset_first = inout1_first.get() + inout1_offset;

            // 3. run algorithms
            for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
                invoke_on_all_hetero_policies<0>()(TestName(),
                    inout1_offset_first, inout1_offset_first + n, n);
            }
        }
#endif
        { // sycl::buffer
            // 1. create buffers
            cl::sycl::buffer<T, 1> inout1{ cl::sycl::range<1>(max_n + inout1_offset) };

            // 2. create an iterator over buffer
            auto inout1_offset_first = oneapi::dpl::begin(inout1) + inout1_offset;

            // 3. run algorithms
            for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
                invoke_on_all_hetero_policies<1>()(TestName(),
                    inout1_offset_first, inout1_offset_first + n, n);
            }
        }
    }

    template <typename T, typename TestName>
    void
    test2buffers()
    {
        const cl::sycl::queue& queue = my_queue; // usm and allocator requires queue
#if _PSTL_SYCL_TEST_USM
        { // USM
            // 1. allocate usm memory
            auto sycl_deleter = [&queue](T* mem) { cl::sycl::free(mem, queue.get_context()); };
            ::std::unique_ptr<T, decltype(sycl_deleter)> inout1_first(
                (T*)cl::sycl::malloc_shared(sizeof(T)*(max_n + inout1_offset), queue.get_device(), queue.get_context()),
                sycl_deleter);
            ::std::unique_ptr<T, decltype(sycl_deleter)> inout2_first(
                (T*)cl::sycl::malloc_shared(sizeof(T)*(max_n + inout2_offset), queue.get_device(), queue.get_context()),
                sycl_deleter);

            // 2. create pointers at first+offset
            T* inout1_offset_first = inout1_first.get() + inout1_offset;
            T* inout2_offset_first = inout2_first.get() + inout2_offset;

            // 3. run algorithms
            for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
                invoke_on_all_hetero_policies<0>()(TestName(),
                    inout1_offset_first, inout1_offset_first + n,
                    inout2_offset_first, inout2_offset_first + n, n);
            }
        }
#endif
        { // sycl::buffer
            // 1. create buffers
            cl::sycl::buffer<T, 1> inout1{ cl::sycl::range<1>(max_n + inout1_offset) };
            cl::sycl::buffer<T, 1> inout2{ cl::sycl::range<1>(max_n + inout2_offset) };

            // 2. create iterators over buffers
            auto inout1_offset_first = oneapi::dpl::begin(inout1) + inout1_offset;
            auto inout2_offset_first = oneapi::dpl::begin(inout2) + inout2_offset;

            // 3. run algorithms
            for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
                invoke_on_all_hetero_policies<1>()(TestName(),
                    inout1_offset_first, inout1_offset_first + n,
                    inout2_offset_first, inout2_offset_first + n, n);
            }
        }
    }

    template<typename T, typename TestName>
    void test3buffers(int mult = 1)
    {
        const cl::sycl::queue& queue = my_queue; // usm requires queue
#if _PSTL_SYCL_TEST_USM
        { // USM
            // 1. allocate usm memory
            auto sycl_deleter = [&queue](T* mem) { cl::sycl::free(mem, queue.get_context()); };
            ::std::unique_ptr<T, decltype(sycl_deleter)> inout1_first(
                (T*)cl::sycl::malloc_shared(sizeof(T)*(max_n + inout1_offset), queue.get_device(), queue.get_context()),
                sycl_deleter);
            ::std::unique_ptr<T, decltype(sycl_deleter)> inout2_first(
                (T*)cl::sycl::malloc_shared(sizeof(T)*(max_n + inout2_offset), queue.get_device(), queue.get_context()),
                sycl_deleter);
            ::std::unique_ptr<T, decltype(sycl_deleter)> inout3_first(
                (T*)cl::sycl::malloc_shared(mult*sizeof(T)*(max_n + inout3_offset), queue.get_device(), queue.get_context()),
                sycl_deleter);

            // 2. create pointers at first+offset
            T* inout1_offset_first = inout1_first.get() + inout1_offset;
            T* inout2_offset_first = inout2_first.get() + inout2_offset;
            T* inout3_offset_first = inout3_first.get() + inout3_offset;

            // 3. run algorithms
            for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n))) {
                invoke_on_all_hetero_policies<0>()(TestName(),
                    inout1_offset_first, inout1_offset_first + n,
                    inout2_offset_first, inout2_offset_first + n,
                    inout3_offset_first, inout3_offset_first + n, n);
            }
        }
#endif
        { // sycl::buffer
            // 1. create buffers
            cl::sycl::buffer<T, 1> inout1{ cl::sycl::range<1>(max_n + inout1_offset) };
            cl::sycl::buffer<T, 1> inout2{ cl::sycl::range<1>(max_n + inout2_offset) };
            cl::sycl::buffer<T, 1> inout3{ cl::sycl::range<1>(mult*max_n + inout3_offset) };

            // 2. create iterators over buffers
            auto inout1_offset_first = oneapi::dpl::begin(inout1) + inout1_offset;
            auto inout2_offset_first = oneapi::dpl::begin(inout2) + inout2_offset;
            auto inout3_offset_first = oneapi::dpl::begin(inout3) + inout3_offset;

            // 3. run algorithms
            for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n))) {
                invoke_on_all_hetero_policies<1>()(TestName(),
                    inout1_offset_first, inout1_offset_first + n,
                    inout2_offset_first, inout2_offset_first + n,
                    inout3_offset_first, inout3_offset_first + n, n);
            }
        }
    }

    // use the function carefully due to temporary accessor creation.
    // Race conditiion between host and device may be occured
    // if we work with the buffer host memory when kernel is invoked on device
    template <typename Iter, cl::sycl::access::mode mode = cl::sycl::access::mode::read_write>
    typename ::std::iterator_traits<Iter>::pointer
    get_host_pointer(Iter it)
    {
        auto temp_idx = it - oneapi::dpl::begin(it.get_buffer());
        return it.get_buffer().template get_access<mode>().get_pointer() + temp_idx;
    }


    template <typename T, int Dim, cl::sycl::access::mode AccMode, cl::sycl::access::target AccTarget,
              cl::sycl::access::placeholder Placeholder>
    T* get_host_pointer(cl::sycl::accessor<T, Dim, AccMode, AccTarget, Placeholder>& acc)
    {
        return acc.get_pointer();
    }

    // for USM pointers
    template<typename T>
    T* get_host_pointer(T* data)
    {
        return data;
    }

    template <typename Iter, cl::sycl::access::mode mode = cl::sycl::access::mode::read_write>
    auto
    get_host_access(Iter it)
        -> decltype(it.get_buffer().template get_access<mode>(it.get_buffer().get_count() - (it - oneapi::dpl::begin(it.get_buffer())),
                                                              it - oneapi::dpl::begin(it.get_buffer())))
    {
        auto temp_buf = it.get_buffer();
        auto temp_idx = it - oneapi::dpl::begin(temp_buf);
        return temp_buf.template get_access<mode>(temp_buf.get_count() - temp_idx, temp_idx);
    }

    template<typename T>
    T* get_host_access(T* data)
    {
        return data;
    }

} /* namespace TestUtils */
#endif
