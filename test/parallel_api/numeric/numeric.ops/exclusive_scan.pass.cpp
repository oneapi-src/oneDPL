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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"

#include <iostream>
#include <vector>

#include "support/scan_serial_impl.h"

// This macro may be used to analyze source data and test results in exclusive_scan.pass
// WARNING: in the case of using this macro debug output is very large.
//#define DUMP_CHECK_RESULTS

using namespace TestUtils;

DEFINE_TEST_1(test_with_vector, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_with_vector)


#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        std::cout << msg;
        for (Size i = 0; i < n; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << it[i];
        }
        std::cout << std::endl;
    }
#endif // DUMP_CHECK_RESULTS

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 res_first, Iterator2 res_last,
               Iterator3 exp_first, Iterator3 exp_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;

        KeyT init{2};
        ::std::fill_n(keys_first, n, 1);
        oneapi::dpl::exclusive_scan(std::forward<Policy>(exec), keys_first, keys_last, res_first, init,
                                    BinaryOperation());
        exclusive_scan_serial(keys_first, keys_last, exp_first, init, BinaryOperation());

#ifdef DUMP_CHECK_RESULTS
        display_param("expected: ", exp_first, n);
        display_param("actual  : ", res_first, n);
#endif //DUMP_CHECK_RESULTS

        EXPECT_EQ_N(exp_first, res_first, n, "wrong effect from exclusive_scan");
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
                              void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
    }
};

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type, typename KernelName, typename _BinaryOp = oneapi::dpl::__internal::__pstl_plus>
void
test_with_usm(sycl::queue& q, const ::std::size_t count, _BinaryOp binary_op = _BinaryOp())
{
    // Prepare source data
    std::vector<int> h_idx(count, 1);

    int init = 2;
    
    // Copy source data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_idx(q, ::std::begin(h_idx), ::std::end(h_idx));
    auto d_idx = dt_helper_h_idx.get_data();

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_val(q, count);
    auto d_val = dt_helper_h_val.get_data();

    // Run dpl::exclusive_scan algorithm on USM shared-device memory
    auto myPolicy = oneapi::dpl::execution::make_device_policy<
        TestUtils::unique_kernel_name<KernelName, TestUtils::uniq_kernel_index<alloc_type>()>>(q);
    oneapi::dpl::exclusive_scan(myPolicy, d_idx, d_idx + count, d_val, init, binary_op);

    // Copy results from USM shared/device memory to host
    std::vector<int> h_val(count);
    dt_helper_h_val.retrieve_data(h_val.begin());

    // Check results
    std::vector<int> h_sval_expected(count);
    exclusive_scan_serial(h_idx.begin(), h_idx.begin() + count, h_sval_expected.begin(), init, binary_op);

    EXPECT_EQ_N(h_sval_expected.begin(), h_val.begin(), count, "wrong effect from exclusive_scan");
}

template <sycl::usm::alloc alloc_type, typename KernelName, typename _BinaryOp = oneapi::dpl::__internal::__pstl_plus>
void
test_with_usm(sycl::queue& q, _BinaryOp binary_op = _BinaryOp())
{
    for (::std::size_t n = 0; n <= TestUtils::max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_with_usm<alloc_type, KernelName, _BinaryOp>(q, n);
    }
}

#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename _Tp>
struct UserBinaryOperation
{
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        return __x * __y;
    }
};

template <typename T>
class MyType
{
  public:
    T value;
    MyType() : value() {}
    MyType(const T& a) : value(a) {}

    bool
    operator==(const MyType<T>& a) const
    {
        return value == a.value;
    }

    MyType<T>
    operator=(const T& a)
    {
        value = a;
        return *this;
    }

    MyType<T> operator*(const MyType<T>& a) const { return MyType<T>{value * a.value}; }
};

template <typename T>
::std::ostream&
operator<<(::std::ostream& os, const MyType<T>& val)
{
    return (os << val.value);
}

int
main()
{

    using BinaryOp = UserBinaryOperation<int>;
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q = TestUtils::get_test_queue();
#if _ONEDPL_DEBUG_SYCL
    std::cout << "    Device Name = " << q.get_device().get_info<sycl::info::device::name>().c_str() << "\n";
#endif // _ONEDPL_DEBUG_SYCL

    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared, class KernelName1>(q);
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device, class KernelName2>(q);
    test_with_usm<sycl::usm::alloc::device, class KernelName3, BinaryOp>(q);
#endif // TEST_DPCPP_BACKEND_PRESENT

    //test with custom operation and integer type
#if TEST_DPCPP_BACKEND_PRESENT
    test_algo_three_sequences<int, test_with_vector<int, BinaryOp>>();
#else
    test_algo_three_sequences<int, test_with_vector<BinaryOp>>();
#endif    

    //test with custom operation and custom (integer wrapper) type
    using ValType = MyType<int>;
    using BinaryOpCustType = UserBinaryOperation<ValType>;
#if TEST_DPCPP_BACKEND_PRESENT
    test_algo_three_sequences<ValType, test_with_vector<ValType, BinaryOpCustType>>();
#else
    test_algo_three_sequences<ValType, test_with_vector<BinaryOpCustType>>();
#endif

    return TestUtils::done();
}
