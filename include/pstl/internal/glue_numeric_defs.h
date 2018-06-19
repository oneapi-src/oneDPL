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

#ifndef __PSTL_glue_numeric_defs_H
#define __PSTL_glue_numeric_defs_H

#include "execution_defs.h"

namespace std {

// [reduce]

template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,T>
reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T init, BinaryOperation binary_op);

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,T>
reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T init);

template<class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,typename iterator_traits<ForwardIterator>::value_type>
reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template <class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, T>
transform_reduce(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, T init);

template <class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation1, class BinaryOperation2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, T>
transform_reduce(ExecutionPolicy&& exec, ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, T init, BinaryOperation1 binary_op1,
                 BinaryOperation2 binary_op2);

template<class ExecutionPolicy, class ForwardIterator, class T, class BinaryOperation, class UnaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,T>
transform_reduce(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, T init, BinaryOperation binary_op, UnaryOperation unary_op);

// [exclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
exclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, T init);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation>
ForwardIterator2 exclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, T init,
                                BinaryOperation binary_op);

// [inclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op, T init);

// [transform.exclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class T, class BinaryOperation, class UnaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform_exclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, T init,
                         BinaryOperation binary_op, UnaryOperation unary_op);

// [transform.inclusive.scan]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryOperation, class UnaryOperation, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform_inclusive_scan(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op,
                         UnaryOperation unary_op, T init);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class UnaryOperation, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy,ForwardIterator2>
transform_inclusive_scan(ExecutionPolicy&& exec,  ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 result, BinaryOperation binary_op,
                         UnaryOperation unary_op);

// [adjacent.difference]

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2, class BinaryOperation>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
adjacent_difference(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first, BinaryOperation op);

template<class ExecutionPolicy, class ForwardIterator1, class ForwardIterator2>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator2>
adjacent_difference(ExecutionPolicy&& exec, ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first);

} // namespace std

#endif /* __PSTL_glue_numeric_defs_H */
