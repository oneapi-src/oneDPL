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

// Tests for find_if and find_if_not

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

struct test_find_if {
    template <typename Policy, typename Iterator, typename Predicate, typename NotPredicate>
    void operator()( Policy&& exec, Iterator first, Iterator last, Predicate pred, NotPredicate not_pred ) {
        auto i = std::find_if(first, last, pred);
        auto j = find_if( exec, first, last, pred);
        EXPECT_TRUE( i == j, 
                     "wrong return value from find_if" );
        auto i_not = find_if_not( exec, first, last, not_pred );
        EXPECT_TRUE( i_not == i,
                     "wrong return value from find_if_not" );
    }
};

template<typename T, typename Predicate, typename Hit, typename Miss>
void test(Predicate pred, Hit hit, Miss miss) {
    auto not_pred = [pred](T x) {return !pred(x);};
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        Sequence<T> in(n, [&](size_t k) -> T {
            return miss(n^k);
        });
        // Try different find positions, including not found.
        // By going backwards, we can add extra matches that are *not* supposed to be found.
        // The decreasing exponential gives us O(n) total work for the loop since each find takes O(m) time.
        for( size_t m=n; m>0; m *= 0.6 ) {
            if(m<n)
                in[m] = hit(n^m);
            invoke_on_all_policies(test_find_if(), in.begin(), in.end(), pred, not_pred);
            invoke_on_all_policies(test_find_if(), in.cbegin(), in.cend(), pred, not_pred);
        }
    }
}

int32_t main( ) {
    // Note that the "hit" and "miss" functions here avoid overflow issues.
    test<Number>( IsMultiple(5,OddTag()),
                  [](int32_t j){return Number(j-j%5,OddTag());},             // hit
                  [](int32_t j){return Number(j%5==0 ? j^1 : j,OddTag());}); // miss

    // Try type for which algorithm can really be vectorized.
    test<float32_t>([](float32_t x) {return x>=0;},
                    [](float32_t j){return j*j;},
                    [](float32_t j){return -1-j*j;});

    std::cout << "done" << std::endl;
    return 0;
}
