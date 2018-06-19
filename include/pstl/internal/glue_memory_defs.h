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

#ifndef __PSTL_glue_memory_defs_H
#define __PSTL_glue_memory_defs_H

#include "execution_defs.h"

namespace std {

// [uninitialized.copy]

template<class ExecutionPolicy, class InputIterator, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_copy(ExecutionPolicy&& exec, InputIterator first, InputIterator last, ForwardIterator result);

template<class ExecutionPolicy, class InputIterator, class Size, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_copy_n(ExecutionPolicy&& exec, InputIterator first, Size n, ForwardIterator result);

// [uninitialized.move]

template<class ExecutionPolicy, class InputIterator, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_move(ExecutionPolicy&& exec, InputIterator first, InputIterator last, ForwardIterator result);

template<class ExecutionPolicy, class InputIterator, class Size, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_move_n(ExecutionPolicy&& exec, InputIterator first, Size n, ForwardIterator result);

// [uninitialized.fill]

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
uninitialized_fill(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value);

template<class ExecutionPolicy, class ForwardIterator, class Size, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_fill_n(ExecutionPolicy&& exec, ForwardIterator first, Size n, const T& value);

// [specialized.destroy]

template <class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
destroy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template <class ExecutionPolicy, class ForwardIterator, class Size>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
destroy_n(ExecutionPolicy&& exec, ForwardIterator first, Size n);

// [uninitialized.construct.default]

template <class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
uninitialized_default_construct(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template <class ExecutionPolicy, class ForwardIterator, class Size>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_default_construct_n(ExecutionPolicy&& exec, ForwardIterator first, Size n);

// [uninitialized.construct.value]

template <class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
uninitialized_value_construct(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last);

template <class ExecutionPolicy, class ForwardIterator, class Size>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_value_construct_n(ExecutionPolicy&& exec, ForwardIterator first, Size n);

} //  namespace std

#endif /* __PSTL_glue_memory_defs_H */
