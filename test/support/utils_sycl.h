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
#ifndef _UTILS_SYCL_H
#define _UTILS_SYCL_H

// File contains common utilities for SYCL that tests rely on

// Do not #include <algorithm>, because if we do we will not detect accidental dependencies.

#include <iterator>

#if TEST_DPCPP_BACKEND_PRESENT
#include "utils_sycl_defs.h"
#endif // TEST_DPCPP_BACKEND_PRESENT

#include "test_config.h"

#include _PSTL_TEST_HEADER(iterator)
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
#include "iterator_utils.h"
#include "utils_invoke.h"
#include "utils_test_base.h"

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
    -> decltype(TestUtils::make_device_policy<_NewKernelName>(::std::forward<_Policy>(__policy)))
{
    return TestUtils::make_device_policy<_NewKernelName>(::std::forward<_Policy>(__policy));
}

#if ONEDPL_FPGA_DEVICE
template <typename _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_new_policy(_Policy&& __policy)
    -> decltype(TestUtils::make_fpga_policy<::std::decay_t<_Policy>::unroll_factor, _NewKernelName>(
        ::std::forward<_Policy>(__policy)))
{
    return TestUtils::make_fpga_policy<::std::decay_t<_Policy>::unroll_factor, _NewKernelName>(
        ::std::forward<_Policy>(__policy));
}
#endif

#if ONEDPL_FPGA_DEVICE
inline auto default_selector =
#    if ONEDPL_FPGA_EMULATOR
        sycl::ext::intel::fpga_emulator_selector_v;
#    else
        sycl::ext::intel::fpga_selector{};
#    endif // ONEDPL_FPGA_EMULATOR

inline auto&& default_dpcpp_policy =
#    if ONEDPL_USE_PREDEFINED_POLICIES
        oneapi::dpl::execution::dpcpp_fpga;
#    else
        TestUtils::make_fpga_policy(sycl::queue{default_selector});
#    endif // ONEDPL_USE_PREDEFINED_POLICIES
#else
inline auto default_selector =
#    if TEST_LIBSYCL_VERSION >= 60000
        sycl::default_selector_v;
#    else
        sycl::default_selector{};
#    endif
inline auto&& default_dpcpp_policy =
#    if ONEDPL_USE_PREDEFINED_POLICIES
        oneapi::dpl::execution::dpcpp_default;
#    else
        oneapi::dpl::execution::make_device_policy(sycl::queue{default_selector});
#    endif // ONEDPL_USE_PREDEFINED_POLICIES
#endif     // ONEDPL_FPGA_DEVICE

inline
sycl::queue get_test_queue()
{
    // create the queue with custom asynchronous exceptions handler
    static sycl::queue my_queue(default_selector, async_handler);
    return my_queue;
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName>
void
test1buffer()
{
    sycl::queue queue = get_test_queue(); // usm and allocator requires queue

#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { max_n, inout1_offset } });

        // 2. create a pointer at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { max_n, inout1_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName>
void
test2buffers()
{
    sycl::queue queue = get_test_queue(); // usm and allocator requires queue

#if _PSTL_SYCL_TEST_USM
    { // USM
        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { max_n, inout1_offset },
                                             { max_n, inout2_offset } });

        // 2. create pointers at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { max_n, inout1_offset },
                                      { max_n, inout2_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName>
void
test3buffers(int mult = kDefaultMultValue)
{
    sycl::queue queue = get_test_queue(); // usm requires queue

#if _PSTL_SYCL_TEST_USM
    { // USM

        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { max_n,        inout1_offset },
                                             { max_n,        inout2_offset },
                                             { max_n * mult, inout3_offset } });

        // 2. create pointers at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n)))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { max_n,        inout1_offset },
                                      { max_n,        inout2_offset },
                                      { max_n * mult, inout3_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n)))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestValueType, typename TestName>
void
test4buffers(int mult = kDefaultMultValue)
{
    sycl::queue queue = get_test_queue(); // usm requires queue

#if _PSTL_SYCL_TEST_USM
    { // USM

        // 1. allocate usm memory
        using TestBaseData = test_base_data_usm<alloc_type, TestValueType>;
        TestBaseData test_base_data(queue, { { max_n,        inout1_offset },
                                             { max_n,        inout2_offset },
                                             { max_n * mult, inout3_offset },
                                             { max_n * mult, inout4_offset } });

        // 2. create pointers at first+offset
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);
        auto inout4_offset_first = test_base_data.get_start_from(UDTKind::eRes2);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n)))
        {
#    if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#    endif
            invoke_on_all_hetero_policies<0>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               inout4_offset_first, inout4_offset_first + n * mult,
                                               n);
        }
    }
#endif
    { // sycl::buffer
        // 1. create buffers
        using TestBaseData = test_base_data_buffer<TestValueType>;
        TestBaseData test_base_data({ { max_n,        inout1_offset },
                                      { max_n,        inout2_offset },
                                      { max_n * mult, inout3_offset },
                                      { max_n * mult, inout4_offset } });

        // 2. create iterators over buffers
        auto inout1_offset_first = test_base_data.get_start_from(UDTKind::eKeys);
        auto inout2_offset_first = test_base_data.get_start_from(UDTKind::eVals);
        auto inout3_offset_first = test_base_data.get_start_from(UDTKind::eRes);
        auto inout4_offset_first = test_base_data.get_start_from(UDTKind::eRes2);

        // 3. run algorithms
        for (size_t n = 1; n <= max_n; n = (n <= 16 ? n + 1 : size_t(3.1415 * n)))
        {
#if _ONEDPL_DEBUG_SYCL
            ::std::cout << "n = " << n << ::std::endl;
#endif
            invoke_on_all_hetero_policies<1>()(create_test_obj<TestValueType, TestName>(test_base_data),
                                               inout1_offset_first, inout1_offset_first + n,
                                               inout2_offset_first, inout2_offset_first + n,
                                               inout3_offset_first, inout3_offset_first + n * mult,
                                               inout4_offset_first, inout4_offset_first + n * mult,
                                               n);
        }
    }
}

template <sycl::usm::alloc alloc_type, typename TestName>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test1buffer()
{
    test1buffer<alloc_type, typename TestName::UsedValueType, TestName>();
}

template <sycl::usm::alloc alloc_type, typename TestName>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test2buffers()
{
    test2buffers<alloc_type, typename TestName::UsedValueType, TestName>();
}

template <sycl::usm::alloc alloc_type, typename TestName>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test3buffers(int mult = kDefaultMultValue)
{
    test3buffers<alloc_type, typename TestName::UsedValueType, TestName>(mult);
}

template <sycl::usm::alloc alloc_type, typename TestName>
::std::enable_if_t<::std::is_base_of_v<test_base<typename TestName::UsedValueType>, TestName>>
test4buffers(int mult = kDefaultMultValue)
{
    test4buffers<alloc_type, typename TestName::UsedValueType, TestName>(mult);
}


// Device copyable noop functor used in testing as surrogate for predicates, binary ops, unary functors
// Intentionally non-trivially copyable to test that device_copyable speciailzation works and we are not
// relying on trivial copyability
struct noop_device_copyable
{
    noop_device_copyable(const noop_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
};

struct noop_non_device_copyable
{
    noop_non_device_copyable(const noop_non_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
};

// Device copyable int wrapper struct used in testing as surrogate for values, value types, etc.
// Intentionally non-trivially copyable to test that device_copyable speciailzation works and we are not
// relying on trivial copyability
struct int_device_copyable
{
    int i;
    int_device_copyable(const int_device_copyable& other) : i(other.i) { std::cout << "non trivial copy ctor\n"; }
};

struct int_non_device_copyable
{
    int i;
    int_non_device_copyable(const int_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }
};

// Device copyable iterator used in testing as surrogate for iterators.
// Intentionally non-trivially copyable to test that device_copyable speciailzation works and we are not
// relying on trivial copyability
struct constant_iterator_device_copyable
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    int i;
    constant_iterator_device_copyable(int __i) : i(__i) {}

    constant_iterator_device_copyable(const constant_iterator_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }

    reference operator*() const { return i; }

    constant_iterator_device_copyable& operator++() { return *this; }
    constant_iterator_device_copyable operator++(int) {return *this; }

    constant_iterator_device_copyable& operator--() {  return *this; }
    constant_iterator_device_copyable operator--(int) { return *this; }

    constant_iterator_device_copyable& operator+=(difference_type n) {return *this; }
    constant_iterator_device_copyable operator+(difference_type n) const { return constant_iterator_device_copyable(i); }
    friend constant_iterator_device_copyable operator+(difference_type n, const constant_iterator_device_copyable& it) { return constant_iterator_device_copyable(it.i); }

    constant_iterator_device_copyable& operator-=(difference_type n) { return *this; }
    constant_iterator_device_copyable operator-(difference_type n) const { return constant_iterator_device_copyable(i); }
    difference_type operator-(const constant_iterator_device_copyable& other) const { return 0; }

    reference operator[](difference_type n) const { return i; }

    bool operator==(const constant_iterator_device_copyable& other) const { return true; }
    bool operator!=(const constant_iterator_device_copyable& other) const { return false; }
    bool operator<(const constant_iterator_device_copyable& other) const { return false; }
    bool operator>(const constant_iterator_device_copyable& other) const { return false; }
    bool operator<=(const constant_iterator_device_copyable& other) const { return true; }
    bool operator>=(const constant_iterator_device_copyable& other) const { return true; }
};

struct constant_iterator_non_device_copyable
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    int i;
    constant_iterator_non_device_copyable(int __i) : i(__i) {}

    constant_iterator_non_device_copyable(const constant_iterator_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }

    reference operator*() const { return i; }

    constant_iterator_non_device_copyable& operator++() { return *this; }
    constant_iterator_non_device_copyable operator++(int) {return *this; }

    constant_iterator_non_device_copyable& operator--() {  return *this; }
    constant_iterator_non_device_copyable operator--(int) { return *this; }

    constant_iterator_non_device_copyable& operator+=(difference_type n) {return *this; }
    constant_iterator_non_device_copyable operator+(difference_type n) const { return constant_iterator_non_device_copyable(i); }
    friend constant_iterator_non_device_copyable operator+(difference_type n, const constant_iterator_non_device_copyable& it) { return constant_iterator_non_device_copyable(it.i); }

    constant_iterator_non_device_copyable& operator-=(difference_type n) { return *this; }
    constant_iterator_non_device_copyable operator-(difference_type n) const { return constant_iterator_non_device_copyable(i); }
    difference_type operator-(const constant_iterator_non_device_copyable& other) const { return 0; }

    reference operator[](difference_type n) const { return i; }

    bool operator==(const constant_iterator_non_device_copyable& other) const { return true; }
    bool operator!=(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator<(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator>(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator<=(const constant_iterator_non_device_copyable& other) const { return true; }
    bool operator>=(const constant_iterator_non_device_copyable& other) const { return true; }
};

} /* namespace TestUtils */

template <>
struct sycl::is_device_copyable<TestUtils::noop_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::int_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::constant_iterator_device_copyable> : std::true_type
{
};


#endif // _UTILS_SYCL_H
