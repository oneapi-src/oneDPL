/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

/*
  TODO: consider implementing the following tests for a better code coverage
  - correctness
  - bad input argument (if applicable)
  - data corruption around/of input and output
  - correctly work with nested parallelism
  - check that algorithm does not require anything more than is described in its requirements section
*/

using namespace Parallel_TestUtils;

struct test_any_of {
    template <typename ExecutionPolicy, typename Iterator, typename Predicate>
    void operator()(ExecutionPolicy&& exec, Iterator begin, Iterator end, Predicate pred, bool expected) {

        auto actualr = std::any_of(exec, begin, end, pred);
        EXPECT_EQ(expected, actualr, "result for any_of");
    }
};

template <typename T>
void test( size_t bits ) {
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {

        // Sequence of odd values
        Sequence<T> in(n, [n, bits](size_t k) {return T(2 * HashBits(n, bits - 1) ^ 1); });

        // Even value, or false when T is bool.
        T spike(2 * HashBits(n, bits - 1));
        Sequence<T> inCopy(in);

        invoke_on_all_policies(test_any_of(),  in.begin(),  in.end(), is_equal_to<T>(spike), false);
        invoke_on_all_policies(test_any_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), false);
        EXPECT_EQ(in, inCopy, "any_of modified input sequence");
        if (n > 0) {
            // Sprinkle in a hit
            in[2 * n / 3] = spike;
            invoke_on_all_policies(test_any_of(),  in.begin(),  in.end(), is_equal_to<T>(spike), true);
            invoke_on_all_policies(test_any_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), true);

            // Sprinkle in a few more hits
            in[n / 2] = spike;
            in[n / 3] = spike;
            invoke_on_all_policies(test_any_of(),  in.begin(),  in.end(), is_equal_to<T>(spike), true);
            invoke_on_all_policies(test_any_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), true);
        }
    }
}

int32_t main( ) {
    test<int32_t>(8*sizeof(int32_t));
    test<uint16_t>(8*sizeof(uint16_t));
    test<float64_t>(53);
#if !__PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>(1);
#endif
    std::cout << done() << std::endl;
    return 0;
}
