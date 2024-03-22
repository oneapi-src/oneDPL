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

#include "permutation_iterator_common.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const int countingItIndexBegin = 0;
    const int countingItIndexEnd = 20;
    dpl::counting_iterator<int> countingItBegin(countingItIndexBegin);
    dpl::counting_iterator<int> countingItEnd(countingItIndexEnd);
    const auto countingItDistanceResult = ::std::distance(countingItBegin, countingItEnd);
    EXPECT_EQ(countingItIndexEnd - countingItIndexBegin, countingItDistanceResult,
              "Wrong result of std::distance<countingIterator1, countingIterator2)");

    // countingItBegin and countingItEnd are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the countingItBegin and countingItEnd that are accessed
    // by the permutation iterator
    const std::size_t perm_size_expected = kDefaultIndexStepOp.eval_items_count(countingItDistanceResult);
    auto permItBegin = dpl::make_permutation_iterator(countingItBegin, kDefaultIndexStepOp);
    auto permItEnd = permItBegin + perm_size_expected;

    //transform_iterator is not trivially_copyable, as it defines a copy assignment operator which does not copy
    // its unary functor.  permutation_iterator is based on transform_iterator and therefore is also not
    // trivially_copyable.  This makes permutation_iterator's device_copyable trait deprecated with sycl 2020.
    EXPECT_TRUE(check_if_device_copyable_by_sycl2020_or_by_old_definition <decltype(permItBegin)>,
                "permutation_iterator (counting_iterator) is not properly copyable");

    const std::size_t perm_size_result = std::distance(permItBegin, permItEnd);
    EXPECT_EQ(perm_size_expected, perm_size_result,
              "Wrong result of std::distance<permutationIterator1, permutationIterator2)");

    std::vector<int> resultCopy(perm_size_result);
    auto itCopiedDataEnd = dpl::copy(TestUtils::default_dpcpp_policy, permItBegin, permItEnd, resultCopy.begin());
    EXPECT_EQ(true, resultCopy.end() == itCopiedDataEnd, "Wrong result of dpl::copy");

    const std::vector<int> expectedCopy = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
    EXPECT_EQ_N(expectedCopy.begin(), resultCopy.begin(), perm_size_result, "Wrong state of dpl::copy data");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
