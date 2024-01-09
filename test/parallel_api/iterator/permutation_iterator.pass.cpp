// -*- C++ -*-
//===-- permutation_iterator.pass.cpp -------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/utils.h"
#include "support/utils_test_base.h"

#include <iostream>

#include "permutation_iterator.h"

#define ASSERT_ON_NOT_IMPLEMENTED_CHECK 0

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;
#endif
using namespace TestUtils;

////////////////////////////////////////////////////////////////////////////////
// 
// Table: Current test state
// 
// +------------------------+-----------------------+-----------+--------------------------------------+-------------------------------------------------+---------------+
// +       Test name        |     Algorithm         + Is modify +         Pattern                      +                Host policy                      + Hetero policy +
// +------------------------+-----------------------+-----------+--------------------------------------+-------------------------------------------------+---------------+
// | test_transform         | dpl::transform        |     N     | __parallel_for                       |                     +                           |       +       |
// | test_transform_reduce  | dpl::transform_reduce |     N     | __parallel_transform_reduce          |                     +                           |       +       |
// | test_find              | dpl::find             |     N     | __parallel_find -> _parallel_find_or |                     +                           |       +       |
// | test_is_heap           | dpl::is_heap          |     N     | __parallel_or -> _parallel_find_or   |                     +                           |       +       |
// | test_merge             | dpl::merge            |     N     | __parallel_merge                     | exc. perm_it_index_tags::transform_iterator(1)  |       +       |
// | test_sort              | dpl::sort             |     Y     | __parallel_stable_sort               | exc. perm_it_index_tags::transform_iterator     |       -       |
// | test_partial_sort      | dpl::partial_sort     |     Y     | __parallel_partial_sort              | exc. perm_it_index_tags::transform_iterator     |       -       |
// | test_remove_if         | dpl::remove_if        |     Y     | __parallel_transform_scan            | exc. perm_it_index_tags::transform_iterator     |       -       |
// +------------------------+-----------------------+-----------+--------------------------------------+-------------------------------------------------+---------------+
// TODO: compile error in case(1) looks like an error in host implementation of dpl::merge

namespace
{
template <typename ExecutionPolicy, typename PermItIndexTag>
struct is_able_to_to_modify_src_data_in_test : ::std::true_type { };

#if TEST_DPCPP_BACKEND_PRESENT
template <typename ExecutionPolicy>
struct is_able_to_to_modify_src_data_in_test<ExecutionPolicy, perm_it_index_tags::counting>
    : ::std::negation<oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<ExecutionPolicy>>>
{
};

template <typename ExecutionPolicy>
struct is_able_to_to_modify_src_data_in_test<ExecutionPolicy, perm_it_index_tags::host>
    : ::std::negation<oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<ExecutionPolicy>>>
{
};

template <typename ExecutionPolicy>
struct is_able_to_to_modify_src_data_in_test<ExecutionPolicy, perm_it_index_tags::usm_shared>
    : ::std::negation<oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<ExecutionPolicy>>>
{
};

#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename ExecutionPolicy>
struct is_able_to_to_modify_src_data_in_test<ExecutionPolicy, perm_it_index_tags::transform_iterator>
    : std::false_type
{
};
};

// Check ability to run non-modifying source data test
template <typename Iterator>
constexpr bool
can_run_nonmodify_test()
{
    return is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator>::value;
}

// Check ability to run modifying source data test
template <typename Policy, typename Iterator, typename PermItIndexTag>
constexpr bool
can_run_modify_test()
{
    return is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator>::value &&
           is_able_to_to_modify_src_data_in_test<Policy, PermItIndexTag>::value;
}

//----------------------------------------------------------------------------//
template <typename ExecutionPolicy>
void
wait_and_throw(ExecutionPolicy&& exec)
{
#if TEST_DPCPP_BACKEND_PRESENT
    if constexpr (oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<ExecutionPolicy>>::value)
    {
        exec.queue().wait_and_throw();
    }
#endif // _PSTL_SYCL_TEST_USM
}

// DEFINE_TEST_PERM_IT should be used to declare permutation iterator tests
#define DEFINE_TEST_PERM_IT(TestClassName, TemplateParams)                                                        \
    template <typename TestValueType, typename TemplateParams>                                                    \
    struct TestClassName : TestUtils::test_base<TestValueType>

// DEFINE_TEST_PERM_IT_CONSTRUCTOR should be used to declare permutation iterator tests constructor
#define DEFINE_TEST_PERM_IT_CONSTRUCTOR(TestClassName)                                                            \
    TestClassName(test_base_data<TestValueType>& _test_base_data)                                                 \
        : TestUtils::test_base<TestValueType>(_test_base_data)                                                    \
    {                                                                                                             \
    }                                                                                                             \
                                                                                                                  \
    template <UDTKind kind, typename Size>                                                                        \
    using TestDataTransfer = typename TestUtils::test_base<TestValueType>::template TestDataTransfer<kind, Size>; \
                                                                                                                  \
    using UsedValueType = TestValueType;

#if TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
void
test_counting_iterator(const std::size_t num_elelemts)
{
    sycl::queue q = TestUtils::get_test_queue();

    std::vector<float> result(num_elelemts, 1);
    oneapi::dpl::counting_iterator<int> first(0);
    oneapi::dpl::counting_iterator<int> last(20);

    // first and last are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the first and last that are accessed
    // by the permutation iterator
    size_t num_elements = std::distance(first, last) / 2 + std::distance(first, last) % 2;
    auto permutation_first = dpl::make_permutation_iterator(first, multiply_index_by_two());
    auto permutation_last = permutation_first + num_elements;

    auto it = std::copy(TestUtils::default_dpcpp_policy, permutation_first, permutation_last, result.begin());
    auto count = ::std::distance(result.begin(), it);

    for (int i = 0; i < count; i++)
        ::std::cout << result[i] << " ";

    ::std::cout << ::std::endl;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

// test_sort : dpl::sort -> __parallel_stable_sort
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_sort, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_sort)

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = n - index;
    }

    template <typename TIterator>
    void check_results(TIterator itBegin, TIterator itEnd)
    {
        const auto result = std::is_sorted(itBegin, itEnd);
        EXPECT_TRUE(result, "Wrong sort data results");
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (can_run_modify_test<Policy, Iterator1, PermItIndexTag>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // sorting data

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    // Fill full source data set (not only values iterated by permutation iterator)
                    generate_data(host_keys.get(), host_keys.get() + n, n);
                    host_keys.update_data();

                    dpl::sort(exec, permItBegin, permItEnd);
                    wait_and_throw(exec);

                    host_keys.retrieve_data();

                    // KSATODO required to implement results check
                    //check_results(permItBegin, permItEnd);
    #if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    assert(false);
    #endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                });
        }
    }
};

// test_partial_sort : dpl::partial_sort -> __parallel_partial_sort
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_partial_sort, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_partial_sort)

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = n - index;
    }

    template <typename TIterator>
    void check_results(TIterator itBegin, TIterator itEnd)
    {
        const auto result = std::is_sorted(itBegin, itEnd);
        EXPECT_TRUE(result, "Wrong partial_sort data results");
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (can_run_modify_test<Policy, Iterator1, PermItIndexTag>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // sorting data

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    const auto testing_n = ::std::distance(permItBegin, permItEnd);

                    for (::std::size_t p = 0; p < testing_n; p = p <= 16 ? p + 1 : ::std::size_t(31.415 * p))
                    {
                        // Fill full source data set (not only values iterated by permutation iterator)
                        generate_data(host_keys.get(), host_keys.get() + n, n);
                        host_keys.update_data();

                        dpl::partial_sort(exec, permItBegin, permItBegin + p, permItEnd);
                        wait_and_throw(exec);

                        host_keys.retrieve_data();

                        // KSATODO required to implement results check
                        //check_results(permItBegin, permItBegin + p);
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                        assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    }
                });
        }
    }
};

// dpl::transform -> __parallel_for
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_transform, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_transform)

    struct TransformOp
    {
        TestValueType operator()(TestValueType arg)
        {
            return arg * arg / 2;
        }
    };

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = n - index;
    }

    template <typename TIterator>
    void clear_output_data(TIterator itBegin, TIterator itEnd)
    {
        ::std::fill(itBegin, itEnd, TestValueType{});
    }

    template <typename TIterator, typename TResultIterator, typename UnaryOperation>
    void check_results(TIterator itBegin, TIterator itEnd, TResultIterator itResult, UnaryOperation op)
    {
        for (auto it = itBegin; it != itEnd; ++it, ++itResult)
        {
            const auto expected = op(*it);
            const auto result = *itResult;
            EXPECT_EQ(expected, result, "Wrong result of dpl::transform call");
        }
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        if constexpr (can_run_nonmodify_test<Iterator1>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for transform
            TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);     // result data of transform

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    // Fill full source data set (not only values iterated by permutation iterator)
                    generate_data(host_keys.get(), host_keys.get() + n, n);
                    host_keys.update_data();

                    clear_output_data(host_vals.get(), host_vals.get() + n);
                    host_vals.update_data();

                    dpl::transform(exec, permItBegin, permItEnd, first2, TransformOp{});
                    wait_and_throw(exec);

                    host_vals.retrieve_data();

                    // KSATODO required to implement results check
                    //check_results(permItBegin, permItEnd, first2, TransformOp{});
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                });
        }
    }
};

// dpl::remove_if -> __parallel_transform_scan
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_remove_if, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_remove_if)

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = (n - index) % 2 ? 0 : 1;
    }

    template <typename TIteratorExpected, typename TIteratorResult>
    void
    check_results(TIteratorExpected itBeginExpected, TIteratorExpected itEndExpected,
                  TIteratorResult itBeginResult, TIteratorResult itEndResult)
    {
        const auto sizeExpected = ::std::distance(itBeginExpected, itEndExpected);
        const auto sizeResult = ::std::distance(itBeginExpected, itEndExpected);
        EXPECT_EQ(sizeExpected, sizeResult, "Wrong result size after dpl::remove_if");

        EXPECT_EQ_N(itBeginExpected, itBeginResult, sizeExpected, "Wrong result of dpl::remove_if");
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (can_run_modify_test<Policy, Iterator1, PermItIndexTag>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for remove_if

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    // Fill full source data set (not only values iterated by permutation iterator)
                    generate_data(host_keys.get(), host_keys.get() + n, n);
                    host_keys.update_data();

                    const auto op = [](TestValueType val) { return val > 0; };

                    auto itEndNewRes = dpl::remove_if(exec, permItBegin, permItEnd, op);
                    wait_and_throw(exec);

                    // KSATODO required to implement results check
                    //check_results(srcDataCopy.begin(), itEndNewExp, permItBegin, itEndNewRes);
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                });
        }
    }
};

// dpl::reduce, dpl::transform_reduce -> __parallel_transform_reduce
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_transform_reduce, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_transform_reduce)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (can_run_nonmodify_test<Iterator1>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for transform_reduce

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    // Fill full source data set (not only values iterated by permutation iterator)
                    generate_data(host_keys.get(), host_keys.get() + n, TestValueType{});
                    host_keys.update_data();

                    const auto result = dpl::transform_reduce(exec, permItBegin, permItEnd, TestValueType{},
                                                              ::std::plus<TestValueType>(), ::std::negate<TestValueType>());
                    wait_and_throw(exec);

                    // KSATODO required to implement results check
                    //const auto expected =
                    //    ::std::transform_reduce(permItBegin, permItEnd, TestValueType{}, ::std::plus<TestValueType>(),
                    //                            ::std::negate<TestValueType>());
                    //
                    //EXPECT_EQ(expected, result, "Wrong result of dpl::transform_reduce");
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                });
        }
    }
};

// dpl::merge, dpl::inplace_merge -> __parallel_merge
DEFINE_TEST_PERM_IT(test_merge, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_merge)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3, Iterator3 last3, Size n)
    {
        if constexpr (can_run_nonmodify_test<Iterator1>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data(1) for merge
            TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);     // source data(2) for merge
            TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);     // merge results

            assert(::std::distance(first3, last3) >= ::std::distance(first1, last1) + ::std::distance(first2, last2));

            // Fill full source data set
            generate_data(host_keys.get(), host_keys.get() + n, TestValueType{});
            generate_data(host_vals.get(), host_vals.get() + n, TestValueType{} + n / 2);

            host_keys.update_data();
            host_vals.update_data();

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin1, auto permItEnd1)
                {
                    const auto size1 = ::std::distance(permItBegin1, permItEnd1);

                    test_through_permutation_iterator_mibt<Iterator2, Size, PermItIndexTag>{first2, n}(
                        [&](auto permItBegin2, auto permItEnd2)
                        {
                            const auto size2 = ::std::distance(permItBegin1, permItEnd1);

                            const auto resultEnd = dpl::merge(exec, permItBegin1, permItEnd1, permItBegin2, permItEnd2, first3);
                            wait_and_throw(exec);

                            host_res.retrieve_data();

                            // KSATODO required to implement results check
                            //std::vector<TestValueType> expected(3 * n);
                            //
                            //auto expectedEnd = std::merge(permItBegin1, permItEnd1, permItBegin2, permItEnd2, expected.begin());
                            //
                            //const auto expectedSize = ::std::distance(expected.begin(), expectedEnd);
                            //const auto resultSize = ::std::distance(resultBuf.begin(), resultEnd);
                            //EXPECT_EQ(expectedSize, resultSize, "Wrong size from dpl::merge");
                            //EXPECT_EQ_N(expected.begin(), resultBuf.begin(), expectedSize, "Wrong result of dpl::merge");
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                            assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                        });
                });
        }
    }
};

// dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
DEFINE_TEST_PERM_IT(test_find, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_find)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (can_run_nonmodify_test<Iterator1>())
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for find

            // Fill full source data set
            generate_data(host_keys.get(), host_keys.get() + n, TestValueType{});
            host_keys.update_data();

            assert(n > 0);
            const auto valueToFind = *host_keys.get();

            test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    if (2 <= ::std::distance(permItBegin, permItEnd))
                    {
                        const auto result = dpl::find(exec, permItBegin, permItEnd, valueToFind);
                        wait_and_throw(exec);

                        // KSA//TODO: required to implement results check
                        //EXPECT_TRUE(result != permItEnd, "Wrong result of dpl::find");
                        //
                        //const auto foundedVal = *result;
                        //EXPECT_EQ(foundedVal, valueToFind, "Incorrect value was found in dpl::find");
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                        assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    }
                });
        }
    }
};

// dpl::is_heap, dpl::includes -> __parallel_or -> _parallel_find_or
DEFINE_TEST_PERM_IT(test_is_heap, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_is_heap)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (can_run_nonmodify_test<Iterator1>())
        {
            for (bool bCallMakeHeap : {false, true})
            {
                TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for find

                // Fill full source data set
                generate_data(host_keys.get(), host_keys.get() + n, TestValueType{});
                if (bCallMakeHeap)
                    ::std::make_heap(host_keys.get(), host_keys.get());
                host_keys.update_data();

                test_through_permutation_iterator_mibt<Iterator1, Size, PermItIndexTag>{first1, n}(
                    [&](auto permItBegin, auto permItEnd)
                    {
                        const auto result = dpl::is_heap(exec, permItBegin, permItEnd);
                        wait_and_throw(exec);

                        // KSA//TODO: required to implement results check
                        //const auto expected = std::is_heap(permItBegin, permItEnd);
                        //
                        //EXPECT_EQ(expected, result, "Wrong result of dpl::is_heap");
#if ASSERT_ON_NOT_IMPLEMENTED_CHECK
                        assert(false);
#endif // ASSERT_ON_NOT_IMPLEMENTED_CHECK
                    });
            }
        }
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type, typename ValueType, typename PermItIndexTag>
void
run_algo_tests_on_buffers()
{
    // dpl::transform -> __parallel_for (only for random_access_iterator)
    test2buffers<alloc_type, ValueType, test_transform<ValueType, PermItIndexTag>>();

    // dpl::reduce, dpl::transform_reduce -> __parallel_transform_reduce (only for random_access_iterator)
    test1buffer<alloc_type, ValueType, test_transform_reduce<ValueType, PermItIndexTag>>();

    // dpl::merge, dpl::inplace_merge -> __parallel_merge
    test3buffers<alloc_type, ValueType, test_merge<ValueType, PermItIndexTag>>(2);

    // dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
    test1buffer<alloc_type, ValueType, test_find<ValueType, PermItIndexTag>>();

    // dpl::is_heap, dpl::includes -> __parallel_or -> _parallel_find_or
    test1buffer<alloc_type, ValueType, test_is_heap<ValueType, PermItIndexTag>>();

    // dpl::remove_if -> __parallel_transform_scan (only for random_access_iterator)
    test1buffer<alloc_type, ValueType, test_remove_if<ValueType, PermItIndexTag>>();

    // test_sort : dpl::sort -> __parallel_stable_sort (only for random_access_iterator)
    test1buffer<alloc_type, ValueType, test_sort<ValueType, PermItIndexTag>>();

    // dpl::partial_sort -> __parallel_partial_sort (only for random_access_iterator)
    test1buffer<alloc_type, ValueType, test_partial_sort<ValueType, PermItIndexTag>>();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename ValueType, typename PermItIndexTag>
void
run_algo_tests_on_sequence()
{
    // dpl::transform -> __parallel_for (only for random_access_iterator)
    test_algo_two_sequences<ValueType, test_transform<ValueType, PermItIndexTag>>();

    // dpl::reduce, dpl::transform_reduce -> __parallel_transform_reduce (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_transform_reduce<ValueType, PermItIndexTag>>();

    // dpl::merge, dpl::inplace_merge -> __parallel_merge
    if constexpr (!::std::is_same_v<PermItIndexTag, perm_it_index_tags::transform_iterator>)
    {
        test_algo_three_sequences<ValueType, test_merge<ValueType, PermItIndexTag>>(2);
    }

    // dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
    test_algo_one_sequence<ValueType, test_find<ValueType, PermItIndexTag>>();

    // dpl::is_heap, dpl::includes -> __parallel_or -> _parallel_find_or
    test_algo_one_sequence<ValueType, test_is_heap<ValueType, PermItIndexTag>>();

    // dpl::remove_if -> __parallel_transform_scan (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_remove_if<ValueType, PermItIndexTag>>();

    // test_sort : dpl::sort -> __parallel_stable_sort (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_sort<ValueType, PermItIndexTag>>();

    // dpl::partial_sort -> __parallel_partial_sort (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_partial_sort<ValueType, PermItIndexTag>>();
}

template <typename ValueType, typename PermItIndexTag>
void
run_algo_tests()
{
#if TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <USM::shared, USM::device, sycl::buffer> + <all_hetero_policies>
    run_algo_tests_on_buffers<sycl::usm::alloc::shared, ValueType, PermItIndexTag>();
    run_algo_tests_on_buffers<sycl::usm::alloc::device, ValueType, PermItIndexTag>();

#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    run_algo_tests_on_sequence<ValueType, PermItIndexTag>();
}

int
main()
{
    using ValueType = ::std::uint32_t;

#if TEST_DPCPP_BACKEND_PRESENT
    test_counting_iterator(100);

    run_algo_tests<ValueType, perm_it_index_tags::usm_shared>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_tests<ValueType, perm_it_index_tags::counting>();
    run_algo_tests<ValueType, perm_it_index_tags::host>();
    run_algo_tests<ValueType, perm_it_index_tags::transform_iterator>();

    return TestUtils::done();
}
