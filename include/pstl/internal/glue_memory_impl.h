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

#ifndef __PSTL_glue_memory_impl_H
#define __PSTL_glue_memory_impl_H

#include "utils.h"
#include "algorithm_impl.h"

namespace std {

// [uninitialized.copy]

template<class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result) {
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::integral_constant<bool, std::is_trivial<_ValueType1>::value && std::is_trivial<_ValueType2>::value>(),
                                    [&]() { return internal::pattern_walk2_brick(__first, __last, __result,
                                                                                [__is_vector](_InputIterator __begin, _InputIterator __end, _ForwardIterator __res)
                                                                                { return internal::brick_copy(__begin, __end, __res, __is_vector); }, __is_parallel); },
                                    [&]() { return internal::pattern_it_walk2(__first, __last, __result, [](_InputIterator __it1, _ForwardIterator __it2)
                                                                              { ::new (internal::reduce_to_ptr(__it2)) _ValueType2(*__it1); }, __is_vector, __is_parallel); }
    );
}

template<class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result) {
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::integral_constant<bool, std::is_trivial<_ValueType1>::value && std::is_trivial<_ValueType2>::value>(),
                                    [&]() { return internal::pattern_walk2_brick_n(__first, __n, __result,
                                                                                   [__is_vector](_InputIterator __begin, _Size __sz, _ForwardIterator __res)
                                                                                   { return internal::brick_copy_n(__begin, __sz, __res, __is_vector); }, __is_parallel); },
                                    [&]() { return internal::pattern_it_walk2_n(__first, __n, __result, [](_InputIterator __it1, _ForwardIterator __it2)
                                                                                { ::new (internal::reduce_to_ptr(__it2)) _ValueType2(*__it1); }, __is_vector, __is_parallel); }
    );
}

// [uninitialized.move]

template<class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result) {
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::integral_constant<bool, std::is_trivial<_ValueType1>::value && std::is_trivial<_ValueType2>::value>(),
                                    [&]() { return internal::pattern_walk2_brick(__first, __last, __result,
                                                                                 [__is_vector](_InputIterator __begin, _InputIterator __end, _ForwardIterator __res)
                                                                                 { return internal::brick_copy(__begin, __end, __res, __is_vector);}, __is_parallel); },
                                    [&]() { return internal::pattern_it_walk2(__first, __last, __result, [](_InputIterator __it1, _ForwardIterator __it2)
                                                                              { ::new (internal::reduce_to_ptr(__it2)) _ValueType2(std::move(*__it1)); }, __is_vector, __is_parallel); }
        );
}

template<class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result) {
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::integral_constant<bool, std::is_trivial<_ValueType1>::value && std::is_trivial<_ValueType2>::value>(),
                                    [&]() { return internal::pattern_walk2_brick_n(__first, __n, __result,
                                                                                   [__is_vector](_InputIterator __begin, _Size __sz, _ForwardIterator __res)
                                                                                   { return internal::brick_copy_n(__begin, __sz, __res, __is_vector);}, __is_parallel); },
                                    [&]() { return internal::pattern_it_walk2_n(__first, __n, __result, [](_InputIterator __it1, _ForwardIterator __it2)
                                                                                { ::new (internal::reduce_to_ptr(__it2)) _ValueType2(std::move(*__it1)); }, __is_vector, __is_parallel); }
        );
}

// [uninitialized.fill]

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    internal::invoke_if_else(std::is_arithmetic<_ValueType>(),
                             [&]() { internal::pattern_walk_brick(__first, __last,
                                                                  [&__value, &__is_vector](_ForwardIterator __begin, _ForwardIterator __end)
                                                                  { internal::brick_fill(__begin, __end, _ValueType(__value), __is_vector);}, __is_parallel); },
                             [&]() { internal::pattern_it_walk1(__first, __last, [&__value](_ForwardIterator __it)
                                                                { ::new (internal::reduce_to_ptr(__it)) _ValueType(__value); }, __is_vector, __is_parallel); }
        );
}

template<class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_fill_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, const _Tp& __value) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::is_arithmetic<_ValueType>(),
                                    [&]() { return internal::pattern_walk_brick_n(__first, __n,
                                                                                  [&__value, &__is_vector](_ForwardIterator __begin, _Size __count)
                                                                                  { return internal::brick_fill_n(__begin, __count, _ValueType(__value), __is_vector);}, __is_parallel); },
                                    [&]() { return internal::pattern_it_walk1_n(__first, __n, [&__value](_ForwardIterator __it)
                                                                                { ::new (internal::reduce_to_ptr(__it)) _ValueType(__value); }, __is_vector, __is_parallel); }
        );
}

// [specialized.destroy]

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
destroy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    internal::invoke_if_not(std::is_trivially_destructible<_ValueType>(),
                            [&]() { internal::pattern_it_walk1(__first, __last,
                                                               [](_ForwardIterator __it){ (*__it).~_ValueType(); }, __is_vector, __is_parallel); }
        );
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
destroy_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::is_trivially_destructible<_ValueType>(),
        [&]() { return std::next(__first, __n);},
                          [&]() { return internal::pattern_it_walk1_n(__first, __n,
                                                                      [](_ForwardIterator __it){ (*__it).~_ValueType(); }, __is_vector, __is_parallel); }
        );
}

// [uninitialized.construct.default]

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_default_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    internal::invoke_if_not(std::is_trivial<_ValueType>(),
                            [&]() { internal::pattern_it_walk1(__first, __last, [](_ForwardIterator __it) { ::new (internal::reduce_to_ptr(__it)) _ValueType; },
                                 __is_vector, __is_parallel); });
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::is_trivial<_ValueType>(),
        [&]() { return std::next(__first, __n);},
                                    [&]() { return internal::pattern_it_walk1_n(__first, __n, [](_ForwardIterator __it)
                                                                                { ::new (internal::reduce_to_ptr(__it)) _ValueType; }, __is_vector, __is_parallel); }
        );
}

// [uninitialized.construct.value]

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_value_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    internal::invoke_if_else(std::is_trivial<_ValueType>(),
                             [&]() { internal::pattern_walk_brick(__first, __last, [__is_vector](_ForwardIterator __begin, _ForwardIterator __end)
                                                                  { internal::brick_fill(__begin, __end, _ValueType(), __is_vector);}, __is_parallel); },
                             [&]() { internal::pattern_it_walk1(__first, __last, [](_ForwardIterator it)
                                                                { ::new (internal::reduce_to_ptr(it)) _ValueType(); }, __is_vector, __is_parallel); }
        );
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_value_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n) {
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    using namespace pstl;

    const auto __is_parallel = internal::is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector = internal::is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return internal::invoke_if_else(std::is_trivial<_ValueType>(),
                                    [&]() { return internal::pattern_walk_brick_n(__first, __n, [__is_vector](_ForwardIterator __begin, _Size __count)
                                                                                  { return internal::brick_fill_n(__begin, __count, _ValueType(), __is_vector);}, __is_parallel); },
                                    [&]() { return internal::pattern_it_walk1_n(__first, __n, [](_ForwardIterator __it)
                                                                                { ::new (internal::reduce_to_ptr(__it)) _ValueType(); }, __is_vector, __is_parallel); }
        );
}

} // namespace std

#endif /* __PSTL_glue_memory_imple_H */
