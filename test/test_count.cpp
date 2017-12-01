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

// Tests for count and count_if
#include "test/pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

struct test_count {
    template <typename Policy, typename Iterator, typename T>
    void operator()( Policy&& exec, Iterator first, Iterator last, T needle ) {
        auto expected = std::count(first, last, needle);
        auto result = std::count( exec, first, last, needle );
        EXPECT_EQ( expected, result, "wrong count result" );
    }
};

struct test_count_if {
    template <typename Policy, typename Iterator, typename Predicate>
    void operator()( Policy&& exec, Iterator first, Iterator last, Predicate pred ) {
        auto expected = std::count_if(first, last, pred);
        auto result = std::count_if( exec, first, last, pred );
        EXPECT_EQ( expected, result, "wrong count_if result" );
    }
};

template<typename T>
class IsEqual {
    T value;
public:
    IsEqual( T value_, OddTag ) : value(value_) {}
    bool operator()( const T& x ) const {return x==value;}
};

template<typename In, typename T, typename Predicate, typename Convert>
void test(T needle, Predicate pred, Convert convert) {
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        Sequence<In> in(n, [=](size_t k) -> In {
            // Sprinkle "42" and "50" early, so that short sequences have non-zero count.
            return convert((n-k-1)%3==0 ? 42 : (n-k-2)%5==0 ? 50 : 3*(int(k)%1000-500));
        });
        invoke_on_all_policies(test_count(), in.begin(), in.end(), needle);
        invoke_on_all_policies(test_count_if(), in.begin(), in.end(), pred);

        invoke_on_all_policies(test_count(), in.cbegin(), in.cend(), needle);
        invoke_on_all_policies(test_count_if(), in.cbegin(), in.cend(), pred);
    }
}

int32_t main( ) {
    test<int32_t>(42, IsEqual<int32_t>(50,OddTag()), [](int j) {return j;});
#if !__PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test<int32_t>(42, [](const int32_t& x){return true;}, [](int32_t j) {return j;});
#endif
    test<float64_t>(42, IsEqual<float64_t>(50,OddTag()), [](int32_t j) {return float64_t(j);});
    test<Number>(Number(42,OddTag()),
                 IsEqual<Number>(Number(50,OddTag()),OddTag()),
                 [](int32_t j){return Number(j,OddTag());});
    std::cout << "done" << std::endl;
    return 0;
}
