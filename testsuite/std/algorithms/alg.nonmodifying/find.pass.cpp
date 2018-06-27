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

// Tests for find
#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

using namespace Parallel_TestUtils;

struct test_find {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename Iterator, typename Value>
    void operator()(pstl::execution::unsequenced_policy, Iterator first, Iterator last, Value value) { }
    template <typename Iterator, typename Value>
    void operator()(pstl::execution::parallel_unsequenced_policy, Iterator first, Iterator last, Value value) { }
#endif

    template <typename Policy, typename Iterator, typename Value>
    void operator()( Policy&& exec, Iterator first, Iterator last, Value value ) {
        auto i = std::find(first, last, value);
        auto j = find( exec, first, last, value );
        EXPECT_TRUE( i == j,
                     "wrong return value from find" );
    }
};

template<typename T, typename Value, typename Hit, typename Miss>
void test(Value value, Hit hit, Miss miss) {
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        Sequence<T> in(n, [&](size_t k) -> T {
            return miss(n^k);
        });
        // Try different find positions, including not found.
        // By going backwards, we can add extra matches that are *not* supposed to be found.
        // The decreasing exponential gives us O(n) total work for the loop since each find takes O(m) time.
        for( size_t m=n; m>0 ; m *= 0.6 ) {
            if(m<n)
                in[m] = hit(n^m);
            invoke_on_all_policies(test_find(), in.begin(), in.end(), value);
            invoke_on_all_policies(test_find(), in.cbegin(), in.cend(), value);
        }
    }
}

// Type defined for sake of checking that std::find works with asymmetric ==.
class Weird {
    Number value;
public:
    friend bool operator==(Number x, Weird y) {return x==y.value;}
    Weird(int32_t val, OddTag ) : value(val,OddTag()) {}
};

int32_t main( ) {
    // Note that the "hit" and "miss" functions here avoid overflow issues.
    test<Number>( Weird(42,OddTag()),
                  [](int32_t j){return Number(42,OddTag());},                // hit
                  [](int32_t j){return Number(j==42 ? 0 : j,OddTag());});    // miss

    // Test with value that is equal to two different bit patterns (-0.0 and 0.0)
    test<float32_t>( -0.0,
                 [](int32_t j){return j&1 ? 0.0 : -0.0;},                    // hit
                 [](int32_t j){return j==0 ? ~j : j;});                      // miss


    std::cout << "done" << std::endl;
    return 0;
}
