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

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"
#include "pstl/internal/algorithm_impl.h" //for usage a serial pstl::internal::for_each_n function

using namespace TestUtils;

template<typename Type>
struct Gen {
    Type operator()(std::size_t k) {
        return Type(k%5!=1 ? 3*k-7 : 0);
    };
};

template<typename T>
struct Flip {
    int val;
    Flip(int y) : val(y) {}
    T operator()( T& x ) const {return x = val-x;}
};

struct test_one_policy {
    template <typename Policy, typename Iterator, typename Size>
    void operator()( Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n) {
        typedef typename std::iterator_traits<Iterator>::value_type T;

        // Try for_each
        std::for_each(expected_first, expected_last, Flip<T>(1));
        for_each( exec, first, last, Flip<T>(1) );
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_each");

        // Try for_each_n
        std::for_each(expected_first, expected_last, Flip<T>(1));
        for_each_n( exec, first, n, Flip<T>(1) );
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_each_n");
    }
};

template <typename T>
void test() {
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n) ) {
        Sequence<T> inout(n, Gen<T>());
        Sequence<T> expected(n, Gen<T>());
        invoke_on_all_policies(test_one_policy(), inout.begin(), inout.end(), expected.begin(), expected.end(), inout.size());
    }
}

int32_t main( ) {
    test<int32_t>();
    test<uint16_t>();
    test<float64_t>();
    std::cout << "done" << std::endl;
    return 0;
}
