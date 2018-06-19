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

#ifndef __PSTL_glue_numeric_impl_H
#define __PSTL_glue_numeric_impl_H

#include <functional>

#include "utils.h"
#include "numeric_impl.h"

namespace std {

// [reduce]

template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,T>
reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T init, BinaryOperation binary_op) {
    return transform_reduce(std::forward<ExecutionPolicy>(exec), first, last, init, binary_op, pstl::internal::no_op());
}

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,T>
reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T init) {
    return transform_reduce(std::forward<ExecutionPolicy>(exec), first, last, init, std::plus<T>(), pstl::internal::no_op());
}

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,typename iterator_traits<ForwardIterator>::value_type>
reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type T;
    return transform_reduce(std::forward<ExecutionPolicy>(exec), first, last, T{}, std::plus<T>(), pstl::internal::no_op());
}

// [transform.reduce]

template <class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, T>
transform_reduce(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, T init) {
    typedef typename iterator_traits<ForwardIterator1>::value_type input_type;
    using namespace pstl::internal;
    return pattern_transform_reduce(first1, last1, first2, init, std::plus<T>(), std::multiplies<input_type>(),
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template <class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation1, class BinaryOperation2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, T>
transform_reduce(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2) {
    using namespace pstl::internal;
    return pattern_transform_reduce(first1, last1, first2, init, binary_op1, binary_op2,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation, class UnaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,T>
transform_reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T init, BinaryOperation binary_op, UnaryOperation unary_op) {
    using namespace pstl::internal;
    return pattern_transform_reduce(first, last, init, binary_op, unary_op,
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator>(exec));
}

// [exclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
exclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, T init) {
    return transform_exclusive_scan(std::forward<ExecutionPolicy>(exec), first, last, result, init, std::plus<T>(), pstl::internal::no_op());
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation>
ForwardIterator2 exclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, T init, BinaryOperation binary_op) {
    return transform_exclusive_scan(std::forward<ExecutionPolicy>(exec), first, last, result, init, binary_op, pstl::internal::no_op());
}

// [inclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result) {
    typedef typename iterator_traits<ForwardIterator1>::value_type input_type;
    return transform_inclusive_scan(std::forward<ExecutionPolicy>(exec), first, last, result, std::plus<input_type>(), pstl::internal::no_op());
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op) {
    return transform_inclusive_scan(std::forward<ExecutionPolicy>(exec), first, last, result, binary_op, pstl::internal::no_op());
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op, T init) {
    return transform_inclusive_scan(std::forward<ExecutionPolicy>(exec), first, last, result, binary_op, pstl::internal::no_op(), init);
}

// [transform.exclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation, class UnaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform_exclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, T init, BinaryOperation binary_op, UnaryOperation unary_op) {
    using namespace pstl::internal;
    return pattern_transform_scan(
        first, last, result, unary_op, init, binary_op,
        /*inclusive=*/std::false_type(),
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator1,ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator1,ForwardIterator2>(exec));
}

// [transform.inclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryOperation, class UnaryOperation, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform_inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op, UnaryOperation unary_op, T init) {
    using namespace pstl::internal;
    return pattern_transform_scan(
        first, last, result, unary_op, init, binary_op,
        /*inclusive=*/std::true_type(),
        is_vectorization_preferred<ExecutionPolicy,ForwardIterator1,ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy,ForwardIterator1,ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class UnaryOperation, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform_inclusive_scan(ExecutionPolicy&& exec,  ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op, UnaryOperation unary_op) {
    if( first!=last ) {
        auto tmp = unary_op(*first);
        *result = tmp;
        return transform_inclusive_scan(std::forward<ExecutionPolicy>(exec), ++first, last, ++result, binary_op, unary_op, tmp);
    } else {
        return result;
    }
}

// [adjacent.difference]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
adjacent_difference(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first, BinaryOperation op) {
    using namespace pstl::internal;

    if (first == last)
        return d_first;

    return pattern_adjacent_difference(first, last, d_first, op,
        is_vectorization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec),
        is_parallelization_preferred<ExecutionPolicy, ForwardIterator1, ForwardIterator2>(exec));
}

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
adjacent_difference(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first) {
    typedef typename iterator_traits<ForwardIterator1>::value_type value_type;
    return adjacent_difference(std::forward<ExecutionPolicy>(exec), first, last, d_first, std::minus<value_type>());
}

} // namespace std

#endif /* __PSTL_glue_numeric_impl_H_ */
