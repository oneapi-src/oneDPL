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

template<class ExecutionPolicy, class InputIterator, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_copy(ExecutionPolicy&& exec, InputIterator first, InputIterator last, ForwardIterator result) {
    typedef typename iterator_traits<InputIterator>::value_type value_type1;
    typedef typename iterator_traits<ForwardIterator>::value_type value_type2;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);

    return invoke_if_else(std::integral_constant<bool, std::is_trivial<value_type1>::value && std::is_trivial<value_type2>::value>(),
        [&]() { return pattern_walk2_brick(first, last, result, [is_vector](InputIterator begin, InputIterator end, ForwardIterator res)
            { return brick_copy(begin, end, res, is_vector); }, is_parallel); },
        [&]() { return pattern_it_walk2(first, last, result, [](InputIterator it1, ForwardIterator it2)
            { ::new (reduce_to_ptr(it2)) value_type2(*it1); }, is_vector, is_parallel); }
    );
}

template<class ExecutionPolicy, class InputIterator, class Size, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_copy_n(ExecutionPolicy&& exec, InputIterator first, Size n, ForwardIterator result) {
    typedef typename iterator_traits<InputIterator>::value_type value_type1;
    typedef typename iterator_traits<ForwardIterator>::value_type value_type2;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);

    return invoke_if_else(std::integral_constant<bool, std::is_trivial<value_type1>::value && std::is_trivial<value_type2>::value>(),
        [&]() { return pattern_walk2_brick_n(first, n, result, [is_vector](InputIterator begin, Size sz, ForwardIterator res)
            { return brick_copy_n(begin, sz, res, is_vector); }, is_parallel); },
        [&]() { return pattern_it_walk2_n(first, n, result, [](InputIterator it1, ForwardIterator it2)
            { ::new (reduce_to_ptr(it2)) value_type2(*it1); }, is_vector, is_parallel); }
    );
}

// [uninitialized.move]

template<class ExecutionPolicy, class InputIterator, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_move(ExecutionPolicy&& exec, InputIterator first, InputIterator last, ForwardIterator result) {
    typedef typename iterator_traits<InputIterator>::value_type value_type1;
    typedef typename iterator_traits<ForwardIterator>::value_type value_type2;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);

    return invoke_if_else(std::integral_constant<bool, std::is_trivial<value_type1>::value && std::is_trivial<value_type2>::value>(),
        [&]() { return pattern_walk2_brick(first, last, result, [is_vector](InputIterator begin, InputIterator end, ForwardIterator res)
            { return brick_copy(begin, end, res, is_vector);}, is_parallel); },
        [&]() { return pattern_it_walk2(first, last, result, [](InputIterator it1, ForwardIterator it2)
            { ::new (reduce_to_ptr(it2)) value_type2(std::move(*it1)); }, is_vector, is_parallel); }
        );
}

template<class ExecutionPolicy, class InputIterator, class Size, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_move_n(ExecutionPolicy&& exec, InputIterator first, Size n, ForwardIterator result) {
    typedef typename iterator_traits<InputIterator>::value_type value_type1;
    typedef typename iterator_traits<ForwardIterator>::value_type value_type2;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, InputIterator, ForwardIterator>(exec);

    return invoke_if_else(std::integral_constant<bool, std::is_trivial<value_type1>::value && std::is_trivial<value_type2>::value>(),
        [&]() { return pattern_walk2_brick_n(first, n, result, [is_vector](InputIterator begin, Size sz, ForwardIterator res)
            { return brick_copy_n(begin, sz, res, is_vector);}, is_parallel); },
        [&]() { return pattern_it_walk2_n(first, n, result, [](InputIterator it1, ForwardIterator it2)
            { ::new (reduce_to_ptr(it2)) value_type2(std::move(*it1)); }, is_vector, is_parallel); }
        );
}

// [uninitialized.fill]

template<class ExecutionPolicy, class ForwardIterator, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
uninitialized_fill(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last, const T& value) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    invoke_if_else(std::is_arithmetic<value_type>(),
        [&]() { pattern_walk_brick(first, last, [&value, &is_vector](ForwardIterator begin, ForwardIterator end)
            { brick_fill(begin, end, value_type(value), is_vector);}, is_parallel); },
        [&]() { pattern_it_walk1(first, last, [&value](ForwardIterator it)
            { ::new (reduce_to_ptr(it)) value_type(value); }, is_vector, is_parallel); }
        );
}

template<class ExecutionPolicy, class ForwardIterator, class Size, class T>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_fill_n(ExecutionPolicy&& exec, ForwardIterator first, Size n, const T& value) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    return invoke_if_else(std::is_arithmetic<value_type>(),
        [&]() { return pattern_walk_brick_n(first, n, [&value, &is_vector](ForwardIterator begin, Size count)
            { return brick_fill_n(begin, count, value_type(value), is_vector);}, is_parallel); },
        [&]() { return pattern_it_walk1_n(first, n, [&value](ForwardIterator it)
            { ::new (reduce_to_ptr(it)) value_type(value); }, is_vector, is_parallel); }
        );
}

// [specialized.destroy]

template <class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
destroy(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    invoke_if_not(std::is_trivially_destructible<value_type>(),
        [&]() { pattern_it_walk1(first, last, [](ForwardIterator it){ (*it).~value_type(); }, is_vector, is_parallel); }
        );
}

template <class ExecutionPolicy, class ForwardIterator, class Size>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
destroy_n(ExecutionPolicy&& exec, ForwardIterator first, Size n) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    return invoke_if_else(std::is_trivially_destructible<value_type>(),
        [&]() { return std::next(first, n);},
        [&]() { return pattern_it_walk1_n(first, n, [](ForwardIterator it){ (*it).~value_type(); }, is_vector, is_parallel); }
        );
}

// [uninitialized.construct.default]

template <class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
uninitialized_default_construct(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    invoke_if_not(std::is_trivial<value_type>(),
        [&]() { pattern_it_walk1(first, last, [](ForwardIterator it) { ::new (reduce_to_ptr(it)) value_type; }, is_vector, is_parallel); });
}

template <class ExecutionPolicy, class ForwardIterator, class Size>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_default_construct_n(ExecutionPolicy&& exec, ForwardIterator first, Size n) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    return invoke_if_else(std::is_trivial<value_type>(),
        [&]() { return std::next(first, n);},
        [&]() { return pattern_it_walk1_n(first, n, [](ForwardIterator it)
            { ::new (reduce_to_ptr(it)) value_type; }, is_vector, is_parallel); }
        );
}

// [uninitialized.construct.value]

template <class ExecutionPolicy, class ForwardIterator>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, void>
uninitialized_value_construct(ExecutionPolicy&& exec, ForwardIterator first, ForwardIterator last) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    invoke_if_else(std::is_trivial<value_type>(),
        [&]() { pattern_walk_brick(first, last, [is_vector](ForwardIterator begin, ForwardIterator end)
            { brick_fill(begin, end, value_type(), is_vector);}, is_parallel); },
        [&]() { pattern_it_walk1(first, last, [](ForwardIterator it)
            { ::new (reduce_to_ptr(it)) value_type(); }, is_vector, is_parallel); }
        );
}

template <class ExecutionPolicy, class ForwardIterator, class Size>
pstl::internal::enable_if_execution_policy<ExecutionPolicy, ForwardIterator>
uninitialized_value_construct_n(ExecutionPolicy&& exec, ForwardIterator first, Size n) {
    typedef typename iterator_traits<ForwardIterator>::value_type value_type;
    using namespace pstl::internal;

    const auto is_parallel = is_parallelization_preferred<ExecutionPolicy, ForwardIterator>(exec);
    const auto is_vector = is_vectorization_preferred<ExecutionPolicy, ForwardIterator>(exec);

    return invoke_if_else(std::is_trivial<value_type>(),
        [&]() { return pattern_walk_brick_n(first, n, [is_vector](ForwardIterator begin, Size count)
            { return brick_fill_n(begin, count, value_type(), is_vector);}, is_parallel); },
        [&]() { return pattern_it_walk1_n(first, n, [](ForwardIterator it)
            { ::new (reduce_to_ptr(it)) value_type(); }, is_vector, is_parallel); }
        );
}

} // namespace std

#endif /* __PSTL_glue_memory_imple_H */
