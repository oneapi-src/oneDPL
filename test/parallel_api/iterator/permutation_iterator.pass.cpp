//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/utils.h"

#include <iostream>

struct multiply_index_by_two
{
    template <typename Index>
    Index
    operator()(const Index& i) const
    {
        return i * 2;
    }
};

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const int num_elelemts = 100;
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

#endif // TEST_DPCPP_BACKEND_PRESENT

    return 0;
}
