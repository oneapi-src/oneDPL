// -*- C++ -*-
//===-- transform_binary.pass.cpp -----------------------------------------===//
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

using namespace TestUtils;

struct test_one_policy
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
              typename BinaryOp>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 /* last2 */,
               OutputIterator out_first, OutputIterator /* out_last */, BinaryOp op)
    {
        ::std::transform(exec, first1, last1, first2, out_first, op);
        check_and_reset(first1, last1, first2, out_first, op);
    }

    template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Op>
    void
    check_and_reset(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator out_first, Op op)
    {
        typedef typename ::std::iterator_traits<OutputIterator>::value_type Out;
        typename ::std::iterator_traits<OutputIterator>::difference_type k = 0;
        for (; first1 != last1; ++first1, ++first2, ++out_first, ++k)
        {
            // check
            const auto expected = op.get_expected(*first1, *first2);
            auto& actual = op.get_actual(*out_first);
            EXPECT_EQ(expected, actual, "wrong value in output sequence");
            // reset
            actual = k % 7 != 4 ? 7 * k + 5 : 0;
        }
    }
};

template <typename In1, typename In2, typename Out, typename Predicate, typename _IteratorAdapter = _Identity>
void
test(Predicate pred, _IteratorAdapter adap = {})
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In1> in1(n, [](size_t k) { return k % 5 != 1 ? In1(3 * k + 7) : 0; });
        Sequence<In2> in2(n, [](size_t k) { return k % 7 != 2 ? In2(5 * k + 5) : 0; });

        Sequence<Out> out(n, [](size_t) { return -1; });

        invoke_on_all_policies<0>()(test_one_policy(), adap(in1.begin()), adap(in1.end()), adap(in2.begin()),
                                    adap(in2.end()), adap(out.begin()), adap(out.end()), pred);
        invoke_on_all_policies<1>()(test_one_policy(), adap(in1.cbegin()), adap(in1.cend()), adap(in2.cbegin()),
                                    adap(in2.cend()), adap(out.begin()), adap(out.end()), pred);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        invoke_if(exec, [&]() {
            InputIterator input_iter2 = input_iter;
            transform(exec, input_iter, input_iter, input_iter2, out_iter, non_const(::std::plus<T>()));
        });
    }
};

int
main()
{
    //const operator()
    test<std::int32_t, std::int32_t, std::int32_t>(TheOperation<std::int32_t, std::int32_t, std::int32_t>(1));
    test<float32_t, float32_t, float32_t>(TheOperation<float32_t, float32_t, float32_t>(1.5));
    //non-const operator()
#if !TEST_DPCPP_BACKEND_PRESENT
    test<std::int32_t, float32_t, float32_t>(non_const(TheOperation<std::int32_t, float32_t, float32_t>(1.5)));
    test<std::int64_t, float64_t, float32_t>(non_const(TheOperation<std::int64_t, float64_t, float32_t>(1.5)));
#endif

    test_algo_basic_double<std::int16_t>(run_for_rnd_fw<test_non_const<std::int16_t>>());

    //test case for zip iterator
    test<std::int32_t, std::int32_t, std::int32_t>(TheOperationZip<std::int32_t>(1), _ZipIteratorAdapter{});

    return done();
}
