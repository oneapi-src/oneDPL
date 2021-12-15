// -*- C++ -*-
//===-- merge.pass.cpp ----------------------------------------------------===//
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

#include "support/utils.h"

#include <functional>

using namespace TestUtils;

template <typename Type>
struct test_merge
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               OutputIterator out_first, OutputIterator out_last)
    {
        using namespace std;
        const auto res = merge(exec, first1, last1, first2, last2, out_first);
        EXPECT_TRUE(res == out_last, "wrong return result from merge");
        EXPECT_TRUE(is_sorted(out_first, res), "wrong result from merge");
    }

    // for reverse iterators
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    void
    operator()(Policy&& exec, ::std::reverse_iterator<InputIterator1> first1, ::std::reverse_iterator<InputIterator1> last1,
               ::std::reverse_iterator<InputIterator2> first2, ::std::reverse_iterator<InputIterator2> last2,
               ::std::reverse_iterator<OutputIterator> out_first, ::std::reverse_iterator<OutputIterator> out_last)
    {
        using namespace std;
        typedef typename ::std::iterator_traits<::std::reverse_iterator<InputIterator1>>::value_type T;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, ::std::greater<T>());

        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, ::std::greater<T>()), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, ::std::greater<T>()),
                    "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, ::std::greater<T>()),
                    "second sequence is not a part of result");
    }
};

template <typename Type>
struct test_merge_compare
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
    typename Compare>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               OutputIterator out_first, OutputIterator out_last, Compare comp)
    {
        using namespace std;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, comp);
        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, comp), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, comp), "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, comp), "second sequence is not a part of result");
    }

    // for reverse iterators
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
    typename Compare>
    void
    operator()(Policy&& exec, ::std::reverse_iterator<InputIterator1> first1, ::std::reverse_iterator<InputIterator1> last1,
               ::std::reverse_iterator<InputIterator2> first2, ::std::reverse_iterator<InputIterator2> last2,
               ::std::reverse_iterator<OutputIterator> out_first, ::std::reverse_iterator<OutputIterator> out_last,
               Compare /* comp */)
    {
        using namespace std;
        typedef typename ::std::iterator_traits<::std::reverse_iterator<InputIterator1>>::value_type T;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, ::std::greater<T>());

        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, ::std::greater<T>()), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, ::std::greater<T>()),
                    "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, ::std::greater<T>()),
                    "second sequence is not a part of result");
    }
};

template <typename T, typename Generator1, typename Generator2>
void
test_merge_by_type(Generator1 generator1, Generator2 generator2)
{
    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in1(max_size, generator1);
    Sequence<T> in2(max_size / 2, generator2);
    Sequence<T> out(in1.size() + in2.size());
    ::std::sort(in1.begin(), in1.end());
    ::std::sort(in2.begin(), in2.end());

    size_t start_size = 0;
#if TEST_DPCPP_BACKEND_PRESENT
    start_size = 2;
#endif

    for (size_t size = start_size; size <= max_size; size = size <= 16 ? size + 1 : size_t(3.1415 * size)) {
#if !TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_policies<0>()(test_merge<T>(),  in1.cbegin(), in1.cbegin() + size,  in2.data(),
                                    in2.data() + size / 2, out.begin(), out.begin() + 1.5 * size);
        invoke_on_all_policies<1>()(test_merge_compare<T>(), in1.cbegin(), in1.cbegin() + size, in2.data(),
                                    in2.data() + size / 2, out.begin(), out.begin() + 1.5 * size, ::std::less<T>());
#endif

        // Currently test harness doesn't execute the testcase for inputs with more than 1000 elements for const iterators to optimize execution time,
        // but merge's parallel version cut off size is equal to 2000. By using a non-const iterator the testcase can be executed for sizes > 1000 and therefore executing
        // the parallel version of merge.
        invoke_on_all_policies<2>()(test_merge<T>(),  in1.begin(), in1.begin() + size,  in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 1.5 * size);
        invoke_on_all_policies<3>()(test_merge_compare<T>(), in1.begin(), in1.begin() + size, in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 1.5 * size, ::std::less<T>());

#if !TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_policies<4>()(test_merge<T>(), in1.data(), in1.data() + size, in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 3 * size / 2);
        invoke_on_all_policies<5>()(test_merge_compare<T>(), in1.data(), in1.data() + size, in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 3 * size / 2, ::std::less<T>());
#endif
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputIterator out_iter)
    {
        merge(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(::std::less<T>()));
    }
};

int
main()
{
    test_merge_by_type<std::int32_t>([](size_t v) { return (v % 2 == 0 ? v : -v) * 3; }, [](size_t v) { return v * 2; });
#if !ONEDPL_FPGA_DEVICE
    test_merge_by_type<float64_t>([](size_t v) { return float64_t(v); }, [](size_t v) { return float64_t(v - 100); });
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    // Wrapper has atomic increment in ctor. It's not allowed in kernel
    test_merge_by_type<Wrapper<std::int16_t>>([](size_t v) { return Wrapper<std::int16_t>(v % 100); },
                                         [](size_t v) { return Wrapper<std::int16_t>(v % 10); });
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());
#endif

    return done();
}
