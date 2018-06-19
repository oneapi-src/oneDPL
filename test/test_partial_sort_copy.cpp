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

// Tests for partial_sort_copy

#include <cmath>
#include "test/pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "test/utils.h"

using namespace TestUtils;

template <typename T>
struct Num {
    T val;

    Num() : val(0) { }
    Num(T v): val(v) { }
    Num(const Num<T>& v): val(v.val) { }
    Num(Num<T>&& v): val(v.val) { }
    Num<T>& operator=(const Num<T>& v) { val = v.val;  return *this; }
    operator T() const { return val; }
    bool operator<(const Num<T>& v) const {
        return val < v.val;
    }
};

template <typename RandomAccessIterator>
struct test_one_policy {
    RandomAccessIterator d_first;
    RandomAccessIterator d_last;
    RandomAccessIterator exp_first;
    RandomAccessIterator exp_last;
    // This ctor is needed because output shouldn't be transformed to any iterator type (only random access iterators are allowed)
    test_one_policy(RandomAccessIterator b1, RandomAccessIterator e1,
        RandomAccessIterator b2, RandomAccessIterator e2) :
        d_first(b1), d_last(e1), exp_first(b2), exp_last(e2) {}
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename Size, typename T, typename Compare>
    void operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash, Compare compare) {}

    template <typename InputIterator, typename Size, typename T, typename Compare>
    void operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash, Compare compare) {}

    template <typename InputIterator, typename Size, typename T>
    void operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash) {}

    template <typename InputIterator, typename Size, typename T>
    void operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash) {}
#endif

    template <typename Policy, typename InputIterator, typename Size, typename T, typename Compare>
    void operator()(Policy&& exec, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash, Compare compare) {
        prepare_data(first, last, n1, trash);
        RandomAccessIterator exp = std::partial_sort_copy(first, last, exp_first, exp_last, compare);
        RandomAccessIterator res = std::partial_sort_copy(exec, first, last, d_first, d_last, compare);

        EXPECT_TRUE((exp - exp_first) == (res - d_first), "wrong result from partial_sort_copy with predicate");
        EXPECT_EQ_N(exp_first, d_first, n2, "wrong effect from partial_sort_copy with predicate");
    }

    template <typename Policy, typename InputIterator, typename Size, typename T>
    void operator()(Policy&& exec, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash) {
        prepare_data(first, last, n1, trash);
        RandomAccessIterator exp = std::partial_sort_copy(first, last, exp_first, exp_last);
        RandomAccessIterator res = std::partial_sort_copy(exec, first, last, d_first, d_last);

        EXPECT_TRUE((exp - exp_first) == (res - d_first), "wrong result from partial_sort_copy without predicate");
        EXPECT_EQ_N(exp_first, d_first, n2, "wrong effect from partial_sort_copy without predicate");
    }
private:
    template <typename InputIterator, typename Size, typename T>
    void prepare_data(InputIterator first, InputIterator last, Size n1, const T& trash) {
        // The rand()%(2*n+1) encourages generation of some duplicates.
        std::srand(42);
        std::generate(first, last, [n1]() {return T(rand() % (2 * n1 + 1)); });

        std::fill(exp_first, exp_last, trash);
        std::fill(d_first, d_last, trash);
    }
};

template<typename T, typename Compare>
void test_partial_sort_copy(Compare compare) {

    typedef typename Sequence<T>::iterator iterator_type;
    const std::size_t n_max = 100000;
    Sequence<T> in(n_max);
    Sequence<T> out(2 * n_max);
    Sequence<T> exp(2 * n_max);
    std::size_t n1 = 0;
    std::size_t n2;
    T trash = T(-666);
    for (; n1 < n_max; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1)) {
        // If both sequences are equal
        n2 = n1;
        invoke_on_all_policies(test_one_policy<iterator_type>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
            in.begin(), in.begin() + n1, n1, n2, trash, compare);

        // If first sequence is greater than second
        n2 = n1 / 3;
        invoke_on_all_policies(test_one_policy<iterator_type>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
            in.begin(), in.begin() + n1, n1, n2, trash, compare);

        // If first sequence is less than second
        n2 = 2 * n1;
        invoke_on_all_policies(test_one_policy<iterator_type>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
            in.begin(), in.begin() + n1, n1, n2, trash, compare);
    }
    // Test partial_sort_copy without predicate
    n1 = n_max;
    n2 = 2 * n1;
    invoke_on_all_policies(test_one_policy<iterator_type>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
        in.begin(), in.begin() + n1, n1, n2, trash);
}

int32_t main() {
    test_partial_sort_copy<Num<float32_t>>([](Num<float32_t> x, Num<float32_t> y) {return x < y; });
    test_partial_sort_copy<int32_t>([](int32_t x, int32_t y) {return x > y; });

    std::cout << done() << std::endl;
    return 0;
}
