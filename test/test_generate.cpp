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

// Tests for generate
#include <atomic>

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

template <typename T>
struct Generator_count {
    const T def_val = T(-1);
    T operator()() { 
        return def_val;
    }
    T default_value() const { return def_val; }
};

struct test_generate {
    template <typename Policy, typename Iterator, typename Size>
    void operator()(Policy&& exec, Iterator first, Iterator last, Size n) {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        // Try random-access iterator
        {
            Generator_count<T> g;
            generate(exec, first, last, g);
            EXPECT_TRUE(std::count(first, last, g.default_value()) == n,
                "generate wrong result for generate");
            std::fill(first, last, T(0));
        }

        {
            Generator_count<T> g;
            const auto m = n/2;
            auto last = generate_n(exec, first, m, g);
            EXPECT_TRUE(std::count(first, last, g.default_value()) == m && last == std::next(first, m),
                "generate_n wrong result for generate_n");
            std::fill(first, last, T(0));
        }
    }
};

template <typename T>
void test_generate_by_type() {
    for (size_t n = 0; n <= 100000; n = n < 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<T> in(n, [](size_t v)->T { return T(0); }); //fill by zero

        invoke_on_all_policies(test_generate(), in.begin(), in.end(), in.size());
    }
}

int32_t main( ) {

    test_generate_by_type<int32_t>();
    test_generate_by_type<float64_t>();

    std::cout << "done" << std::endl;
    return 0;
}
