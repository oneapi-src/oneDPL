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

// We create this additional test for test functional of inclusive scan
// and exclusive scan for an in-place and non-in-place scan variants.

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"
#include "support/scan_serial_impl.h"

#include <vector>

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
static const char kMsgInclusiveScanNormal [] = "Wrong effect from inclusive scan (non-inplace)";
static const char kMsgInclusiveScanInplace[] = "Wrong effect from inclusive scan (inplace)";
static const char kMsgExclusiveScanNormal [] = "Wrong effect from exclusive scan (non-inplace)";
static const char kMsgExclusiveScanInplace[] = "Wrong effect from exclusive scan (inplace)";

// TODO: replace data generation with random data and update check to compare result to
// the result of a serial implementation of the algorithm
template <typename Iterator1, typename Size>
void
initialize_data(Iterator1 host_keys, Size n)
{
    const Size kStartVal = 1;
    for (Size i = 0; i != n; ++i)
        host_keys[i] = kStartVal + i;
}

struct TestingAlgoritmInclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::inclusive_scan(::std::forward<TArgs>(args)...);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        inclusive_scan_serial(::std::forward<TArgs>(args)...);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgInclusiveScanInplace : kMsgInclusiveScanNormal;
    }
};

template <typename BinaryOp, int InitValue>
struct TestingAlgoritmInclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::inclusive_scan(::std::forward<TArgs>(args)..., BinaryOp(), InitValue);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        inclusive_scan_serial(::std::forward<TArgs>(args)..., BinaryOp(), InitValue);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgInclusiveScanInplace : kMsgInclusiveScanNormal;
    }
};

template <int InitValue>
struct TestingAlgoritmExclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(::std::forward<TArgs>(args)..., InitValue);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(::std::forward<TArgs>(args)..., InitValue);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};

template <int InitValue, typename BinaryOp>
struct TestingAlgoritmExclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(::std::forward<TArgs>(args)..., InitValue, BinaryOp());
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(::std::forward<TArgs>(args)..., InitValue, BinaryOp());
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};


DEFINE_TEST_1(test_scan_non_inplace, TestingAlgoritm)
{
    DEFINE_TEST_CONSTRUCTOR(test_scan_non_inplace)

    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
        using ValT = typename ::std::iterator_traits<Iterator2>::value_type;

        TestingAlgoritm testingAlgo;

        // UDTKind::eKeys is assigned with iterators Iterator1 keys_first, Iterator1 keys_last
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        // UDTKind::eVals is assigned with iterators Iterator2 vals_first, Iterator2 vals_last
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        // Initialize source data in the buffer [keys_first, keys_last)
        // Iterators
        //      Iterator1 keys_first, Iterator1 keys_last
        // describe uninitialized memory so we should prepare test data on the host
        initialize_data(host_keys.get(), n);

        // Copy data from the buffer [keys_first, keys_last) to a device.
        // After test data is prepared we should copy data from the host into buffers
        // described by iterators
        //      Iterator1 keys_first, Iterator1 keys_last
        update_data(host_keys);

        // Now we are ready to call the tested algorithm
        testingAlgo.call_onedpl(make_new_policy<new_kernel_name<Policy, 0>>(exec), keys_first, keys_last, vals_first);

        // After the tested algorithm finished we should check the results.
        // For that, at first we copy data from the buffer described by iterators
        //      Iterator2 vals_first, Iterator2 vals_last
        // into the host memory.
        retrieve_data(host_vals);

        // ...and check results in the host memory.
        std::vector<ValT> expected(n);
        testingAlgo.call_serial(host_keys.get(), host_keys.get() + n, expected.data());
        EXPECT_EQ_N(expected.cbegin(), host_vals.get(), n, TestingAlgoritm().getMsg(false));
    }

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    typename ::std::enable_if<
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
        using ValT = typename ::std::iterator_traits<Iterator2>::value_type;

        TestingAlgoritm testingAlgo;

        // Initialize source data in the buffer [keys_first, keys_last)
        initialize_data(keys_first, n);

        testingAlgo.call_onedpl(exec, keys_first, keys_last, vals_first);

        std::vector<ValT> expected(n);
        testingAlgo.call_serial(keys_first, keys_last + n, expected.data());
        EXPECT_EQ_N(expected.cbegin(), vals_first, n, TestingAlgoritm().getMsg(false));
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value, void>::type
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
    }
};

DEFINE_TEST_1(test_scan_inplace, TestingAlgoritm)
{
    DEFINE_TEST_CONSTRUCTOR(test_scan_inplace)

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
        using KeyT = typename ::std::iterator_traits<Iterator1>::value_type;

        TestingAlgoritm testingAlgo;

        // Initialize source data in the buffer [keys_first, keys_last)
        initialize_data(keys_first, n);
        const std::vector<KeyT> source_host_keys_state(keys_first, keys_first + n);

        // Now we are ready to call the tested algorithm
        testingAlgo.call_onedpl(exec, keys_first, keys_last, keys_first);

        std::vector<KeyT> expected(n);
        testingAlgo.call_serial(source_host_keys_state.cbegin(), source_host_keys_state.cend(), expected.data());
        EXPECT_EQ_N(expected.cbegin(), keys_first, n, testingAlgo.getMsg(true));
    }

    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last,
               Size n)
    {
        using KeyT = typename ::std::iterator_traits<Iterator1>::value_type;

        TestingAlgoritm testingAlgo;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        // Initialize source data in the buffer [keys_first, keys_last)
        initialize_data(host_keys.get(), n);
        const std::vector<KeyT> source_host_keys_state(host_keys.get(), host_keys.get() + n);

        // Copy data from the buffer [keys_first, keys_last) to a device.
        update_data(host_keys);

        // Now we are ready to call tested algorithm
        testingAlgo.call_onedpl(make_new_policy<new_kernel_name<Policy, 0>>(exec), keys_first, keys_last, keys_first);

        retrieve_data(host_keys);

        std::vector<KeyT> expected(n);
        testingAlgo.call_serial(source_host_keys_state.cbegin(), source_host_keys_state.cend(), expected.data());
        EXPECT_EQ_N(expected.cbegin(), host_keys.get(), n, testingAlgo.getMsg(true));
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator1>::value, void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
    }
};

template <sycl::usm::alloc alloc_type, typename ValueType, typename BinaryOperation>
void
run_test()
{
    // Inclusive scan
    {
        // Non inplace
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmInclusiveScan> >();
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmInclusiveScanExt<BinaryOperation, 2 > > >();

        // Inplace
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmInclusiveScan>>();
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmInclusiveScanExt<BinaryOperation, 2 > > >();
    }

    // Exclusive scan
    {
        // Non inplace
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScan<2>>>();
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation > > >();

        // Inplace
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScan<2>>>();
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation>>>();
    }
}

namespace oneapi::dpl::__internal
{
    // Check the correctness of oneapi::dpl::__internal::__check_if_iterator_equality_is_possible
    void
    test_internals()
    {
        // Check some internals from oneapi::dpl::__internal
        using namespace oneapi::dpl::__internal;

        constexpr size_t count = 0;
        sycl::buffer<int> buf1(count);
        sycl::buffer<int> buf2(count);

        auto it1 = oneapi::dpl::begin(buf1);
        auto it2 = oneapi::dpl::begin(buf2);
        auto& it1Ref = it1;
        auto& it2Ref = it2;

        EXPECT_TRUE(__check_if_iterator_equality_is_possible(it1, it1),
                    "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_TRUE(__check_if_iterator_equality_is_possible(it1, it1Ref),
                    "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_TRUE(__check_if_iterator_equality_is_possible(it1Ref, it1),
                    "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_TRUE(__check_if_iterator_equality_is_possible(it1Ref, it1Ref),
                    "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(it1, it2),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(it1Ref, it2),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(it1, it2Ref),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(it1Ref, it2Ref),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(oneapi::dpl::begin(buf1), it2),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(oneapi::dpl::begin(buf1), it2Ref),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(oneapi::dpl::begin(buf1), oneapi::dpl::begin(buf2)),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(it1, oneapi::dpl::begin(buf2)),
                     "wrong __check_if_iterator_equality_is_possible result");

        EXPECT_FALSE(__check_if_iterator_equality_is_possible(oneapi::dpl::begin(buf1), nullptr),
                     "wrong __check_if_iterator_equality_is_possible result");
        EXPECT_FALSE(__check_if_iterator_equality_is_possible(nullptr, oneapi::dpl::begin(buf2)),
                     "wrong __check_if_iterator_equality_is_possible result");

        {
            float floatData = .0;

            ::std::vector<int> dataVec{1, 2, 3};
            const auto intConstData = dataVec.data();
            auto intData = dataVec.data();

            // check pointer + pointer
            EXPECT_TRUE(__check_if_iterator_equality_is_possible(intData, intData),
                        "wrong __check_if_iterator_equality_is_possible result");
            // check const pointer + pointer
            EXPECT_TRUE(__check_if_iterator_equality_is_possible(intConstData, intData),
                        "wrong __check_if_iterator_equality_is_possible result");
            // check pointer + const pointer
            EXPECT_TRUE(__check_if_iterator_equality_is_possible(intData, intConstData),
                        "wrong __check_if_iterator_equality_is_possible result");
            // check pointer + pointer to other type
            EXPECT_FALSE(__check_if_iterator_equality_is_possible(intData, floatData),
                         "wrong __check_if_iterator_equality_is_possible result");
        }

        {
            int srcIntData = 0;
            const auto& intConstData = srcIntData;
            auto& intData = srcIntData;
            float floatData = .0;

            //intConstData = 1; // OK: error: cannot assign to variable 'intConstData' with const-qualified type 'const int &'
            intData = 0;

            // Check pointer to const data + pointer to data
            EXPECT_TRUE(__check_if_iterator_equality_is_possible(&intConstData, &intData),
                        "wrong __check_if_iterator_equality_is_possible result");
            // Check pointer to data + pointer to const data
            EXPECT_TRUE(__check_if_iterator_equality_is_possible(&intData, &intConstData),
                        "wrong __check_if_iterator_equality_is_possible result");
            // Check pointer to const data + pointer to const data
            EXPECT_TRUE(__check_if_iterator_equality_is_possible(&intConstData, &intConstData),
                        "wrong __check_if_iterator_equality_is_possible result");
        }
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    using ValueType = int;
    using BinaryOperation = ::std::plus<ValueType>;

    // Check the correctness of oneapi::dpl::__internal::__check_if_iterator_equality_is_possible
    oneapi::dpl::__internal::test_internals();

    // Run tests for USM shared memory
    run_test<sycl::usm::alloc::shared, ValueType, BinaryOperation>();
    // Run tests for USM device memory
    run_test<sycl::usm::alloc::device, ValueType, BinaryOperation>();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
