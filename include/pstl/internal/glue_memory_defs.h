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

template<class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result);

template<class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result);

// [uninitialized.move]

template<class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result);

template<class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result);

// [uninitialized.fill]

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value);

template<class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_fill_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, const _Tp& __value);

// [specialized.destroy]

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
destroy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
destroy_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n);

// [uninitialized.construct.default]

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_default_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n);

// [uninitialized.construct.value]

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_value_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_value_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n);

} //  namespace std
#endif /* __PSTL_glue_memory_defs_H */
