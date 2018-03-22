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

// Tests for the rest numeric algorithms:
// transform_exclusive_scan, exclusive_scan, transform_inclusive_scan, inclusive_scan,
// transform_reduce, reduce, adjacent_difference;

#include "test/pstl_test_config.h"

#include "pstl/execution"
#include "pstl/numeric"

#include "test/utils.h"

using namespace TestUtils;

template<typename T, typename U>
class UnaryOp {
public:
    U operator()(const T& x) const { return U(x); }
};

struct test_transform_scan {
    template <typename Policy, typename InputIterator, typename OutputIterator, typename U, typename BinaryOperation, typename UnaryOperation>
    void operator()(Policy&& exec, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin,
        BinaryOperation binary_op, UnaryOperation unary_op, U init) {

        std::transform_inclusive_scan(exec, inBegin, inEnd, outBegin, binary_op, unary_op, init);
        std::transform_inclusive_scan(exec, inBegin, inEnd, outBegin, binary_op, unary_op);
        std::transform_exclusive_scan(exec, inBegin, inEnd, outBegin, init, binary_op, unary_op);
    }
};

struct test_scan {
    template <typename Policy, typename InputIterator, typename OutputIterator, typename U, typename BinaryOperation>
    void operator()(Policy&& exec, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin,
        BinaryOperation binary_op, U init) {

        std::inclusive_scan(exec, inBegin, inEnd, outBegin, binary_op, init);
        std::inclusive_scan(exec, inBegin, inEnd, outBegin, binary_op);
        std::inclusive_scan(exec, inBegin, inEnd, outBegin);
        std::exclusive_scan(exec, inBegin, inEnd, outBegin, init, binary_op);
        std::exclusive_scan(exec, inBegin, inEnd, outBegin, init);
        std::exclusive_scan(exec, inBegin, inEnd, outBegin, init, non_const(binary_op));
    }
};

struct test_reduce {
    template <typename Policy, typename Iterator, typename UnaryOperation>
    void operator()(Policy&& exec, Iterator begin, Iterator end, UnaryOperation unary_op) {
        typedef typename Iterator::value_type T;
        auto zero = T();
        std::reduce(exec, begin, end);
        std::reduce(exec, begin, end, zero);
        std::reduce(exec, begin, end, zero, std::plus<T>());
        auto f = [](const T & a, const T &b) {return a + b; };
        std::reduce(exec, begin, end, zero, f);
        std::reduce(exec, begin, begin, zero, non_const(std::plus<T>()));

        std::transform_reduce(exec, begin, end, zero, std::plus<T>(), unary_op);
        std::transform_reduce(exec, begin, end, zero, f, unary_op);
        std::transform_reduce(exec, begin, begin, zero, non_const(f), non_const(unary_op));
    }
};

struct test_adjacent_difference {
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void operator()(Policy&& exec, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin) {
        typedef typename InputIterator::value_type T;

        std::adjacent_difference(exec, inBegin, inEnd, outBegin);
        std::adjacent_difference(exec, inBegin, inEnd, outBegin, std::plus<T>());
        auto f = [](const T & a, const T &b) {return a + b; };
        std::adjacent_difference(exec, inBegin, inEnd, outBegin, f);
        std::adjacent_difference(exec, inBegin, inEnd, outBegin, non_const(f));
    }
};

struct test_inner_product {
    template <typename Policy, typename InputIterator, typename OutputIterator, typename T, typename BinaryOperation1, typename BinaryOperation2>
    void operator()(Policy&& exec, InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin,
        T init, BinaryOperation1 op1, BinaryOperation2 op2) {

        std::transform_reduce(exec, inBegin, inEnd, outBegin, init);
        std::transform_reduce(exec, inBegin, inEnd, outBegin, init, op1, op2);

        //usage of "non_const" adapter below - we pass empty container due to just compilation checks
        std::transform_reduce(exec, inBegin, inEnd, outBegin, init, non_const(op1), non_const(op2));
    }
};

template <typename T>
void test_algo_by_type() {
    size_t N = 1000;
    for (size_t n = 0; n < N; n = n < 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<T> in(n, [](size_t v)->T { return T(v); }); //fill 0..n
        invoke_on_all_policies( test_reduce(), in.begin(), in.end(), [](const T& x) { return T(x); });
        invoke_on_all_policies( test_reduce(), in.begin(), in.end(), UnaryOp<T, T>());

        Sequence<T> out(n, [](size_t v)->T { return T(v); }); //fill 0..n
        invoke_on_all_policies( test_adjacent_difference(), in.begin(), in.end(), out.begin());

        typedef std::ptrdiff_t U;
        invoke_on_all_policies( test_transform_scan(), in.begin(), in.end(), out.begin(), std::plus<U>(), UnaryOp<T, U>(), U());
        invoke_on_all_policies( test_transform_scan(), in.begin(), in.end(), out.begin(), [](const U & a, const U &b) {return a + b; }, [](const T& x) { return U(x); }, U());

        invoke_on_all_policies( test_scan(), in.begin(), in.end(), out.begin(), std::plus<U>(), U());
        invoke_on_all_policies( test_scan(), in.begin(), in.end(), out.begin(), [](const U & a, const U &b) {return a + b; }, U());

        invoke_on_all_policies( test_inner_product(), in.begin(), in.end(), out.begin(), T(), std::plus<int32_t>(), std::plus<int32_t>());
        invoke_on_all_policies( test_inner_product(), in.begin(), in.end(), out.begin(), T(), [](const U & a, const U &b) {return a + b; }, std::multiplies<int32_t>());
        invoke_on_all_policies( test_inner_product(), in.begin(), in.end(), out.begin(), T(), std::plus<int32_t>(), [](const U & a, const U &b) {return a + b; });
        invoke_on_all_policies( test_inner_product(), in.begin(), in.end(), out.begin(), T(), [](const U & a, const U &b) {return a + b; }, [](const U & a, const U &b) {return a + b; });
    }
}

int32_t main() {

    test_algo_by_type<int32_t>();
    test_algo_by_type<float32_t>();

    std::cout << done() << std::endl;
    return 0;
}
