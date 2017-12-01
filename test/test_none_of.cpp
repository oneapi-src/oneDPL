/*
    Copyright (c) 2017 Intel Corporation

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

#include "test/pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

/*
  TODO: consider implementing the following tests for a better code coverage
  - correctness
  - bad input argument (if applicable)
  - data corruption around/of input and output
  - correctly work with nested parallelism
  - check that algorithm does not require anything more than is described in its requirements section
*/

using namespace TestUtils;

struct test_none_of {
    template <typename ExecutionPolicy, typename Iterator, typename Predicate>
    void operator()(ExecutionPolicy&& exec, Iterator begin, Iterator end, Predicate pred, bool expected) {

        auto actualr = std::none_of(exec, begin, end, pred);
        EXPECT_EQ( expected, actualr, "result for none_of" );
    }
};

template <typename T>
void test( size_t bits ) {
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {

        // Sequence of odd values 
        Sequence<T> in(n, [n, bits](size_t k) {return T(2 * HashBits(n, bits - 1) ^ 1); });

        // Even value, or false when T is bool.  
        T spike(2 * HashBits(n, bits - 1));
        
        invoke_on_all_policies(test_none_of(), in.begin(), in.end(), is_equal_to<T>(spike), true);
        invoke_on_all_policies(test_none_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), true);
        if( n>0 ) {
            // Sprinkle in a hit
            in[2*n/3] = spike;
            invoke_on_all_policies(test_none_of(), in.begin(), in.end(), is_equal_to<T>(spike), false);
            invoke_on_all_policies(test_none_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), false);

            // Sprinkle in a few more hits    
            in[n/3] = spike;
            in[n/2] = spike;
            invoke_on_all_policies(test_none_of(), in.begin(), in.end(), is_equal_to<T>(spike), false);
            invoke_on_all_policies(test_none_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), false);
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
    std::cout << "done" << std::endl;
    return 0;
}
