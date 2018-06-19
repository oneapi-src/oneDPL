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

#ifndef __PSTL_numeric_impl_H
#define __PSTL_numeric_impl_H

#include <iterator>
#include <type_traits>
#include <numeric>

#include "pstl_config.h"
#include "execution_impl.h"
#include "unseq_backend_simd.h"
#include "bricks_impl.h"

#if __PSTL_USE_PAR_POLICIES
    #include "parallel_backend.h"
#endif

namespace pstl {
namespace internal {

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions, according to draft N4659)
//------------------------------------------------------------------------

template< class T, class BinaryOperation1, class IsArithmeticIsVector>
struct brick_transform_reduce_imp {

    template<class InputIterator1, class InputIterator2, class BinaryOperation2>
    T operator()(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2) noexcept {
        return std::inner_product(first1, last1, first2, init, binary_op1, binary_op2);
    }

    template< class InputIterator, class UnaryOperation>
    T operator()(InputIterator first, InputIterator last, T init, BinaryOperation1 binary_op, UnaryOperation unary_op) noexcept {
        for (; first != last; ++first) {
            init = binary_op(init, unary_op(*first));
        }
        return init;
    }
};

template< class T>
struct brick_transform_reduce_imp<T, std::plus<T>, /*IsArithmeticIsVector*/ std::true_type> {

    template<class InputIterator1, class InputIterator2, class BinaryOperation2>
    T operator()(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init, std::plus<T>, BinaryOperation2 binary_op2) noexcept {
        return unseq_backend::simd_transform_reduce(first1, last1-first1, first2, init, binary_op2);
    }

    template< class InputIterator, class UnaryOperation>
    T operator()(InputIterator first, InputIterator last, T init, std::plus<T>, UnaryOperation unary_op) noexcept {
        return unseq_backend::simd_transform_reduce(first, last-first, init, unary_op);
    }
};

template<class InputIterator1, class InputIterator2, class T, class BinaryOperation1, class BinaryOperation2>
T brick_transform_reduce(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2, /*is_vector=*/std::true_type) noexcept {

    return brick_transform_reduce_imp< T, BinaryOperation1, std::integral_constant<bool, std::is_arithmetic<T>::value> >()(first1, last1, first2, init, binary_op1, binary_op2);
}

template<class InputIterator1, class InputIterator2, class T, class BinaryOperation1, class BinaryOperation2>
T brick_transform_reduce(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2, /*is_vector=*/std::false_type) noexcept {

    return brick_transform_reduce_imp< T, BinaryOperation1, std::false_type >()(first1, last1, first2, init, binary_op1, binary_op2);
}

template<class InputIterator1, class InputIterator2, class T, class BinaryOperation1, class BinaryOperation2, class IsVector>
T pattern_transform_reduce(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_transform_reduce(first1, last1, first2, init, binary_op1, binary_op2, is_vector);
}

template<class InputIterator1, class InputIterator2, class T, class BinaryOperation1, class BinaryOperation2, class IsVector>
T pattern_transform_reduce(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2, IsVector is_vector,  /*is_parallel=*/std::true_type) noexcept {
    return except_handler([&]() {
        return par_backend::parallel_transform_reduce(first1, last1,
            [first1, first2, binary_op2](InputIterator1 i) mutable { return binary_op2(*i, *(first2 + (i - first1))); },
            init,
            binary_op1, // Combine
            [first1, first2, binary_op1, binary_op2, is_vector](InputIterator1 i, InputIterator1 j, T init) -> T {
            return brick_transform_reduce(i, j, first2 + (i - first1),
                init, binary_op1, binary_op2, is_vector);
        });
    });
}

//------------------------------------------------------------------------
// transform_reduce (version with unary and binary functions)
//------------------------------------------------------------------------

template< class InputIterator, class T, class UnaryOperation, class BinaryOperation >
T brick_transform_reduce(InputIterator first, InputIterator last, T init, BinaryOperation binary_op, UnaryOperation unary_op, /*is_vector=*/std::true_type) noexcept {
    return brick_transform_reduce_imp< T, BinaryOperation, std::integral_constant<bool, std::is_arithmetic<T>::value> >()(first, last, init, binary_op, unary_op);
}

template< class InputIterator, class T, class BinaryOperation, class UnaryOperation >
T brick_transform_reduce(InputIterator first, InputIterator last, T init, BinaryOperation binary_op, UnaryOperation unary_op, /*is_vector=*/std::false_type) noexcept {

    return brick_transform_reduce_imp< T, BinaryOperation, std::false_type >()(first, last, init, binary_op, unary_op);
}

template<class InputIterator, class T, class BinaryOperation, class UnaryOperation, class IsVector>
T pattern_transform_reduce(InputIterator first, InputIterator last, T init, BinaryOperation binary_op, UnaryOperation unary_op, IsVector is_vector, /*is_parallel=*/std::false_type ) noexcept {
    return brick_transform_reduce(first, last, init, binary_op, unary_op, is_vector);
}

template<class InputIterator, class T, class BinaryOperation, class UnaryOperation, class IsVector>
T pattern_transform_reduce(InputIterator first, InputIterator last, T init, BinaryOperation binary_op, UnaryOperation unary_op, IsVector is_vector, /*is_parallel=*/std::true_type) {
    return except_handler([&]() {
        return par_backend::parallel_transform_reduce(first, last,
            [unary_op](InputIterator i) mutable {return unary_op(*i); },
            init,
            binary_op,
            [unary_op, binary_op, is_vector](InputIterator i, InputIterator j, T init) {
            return brick_transform_reduce(i, j, init, binary_op, unary_op, is_vector);
        });
    });
}


//------------------------------------------------------------------------
// transform_exclusive_scan
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------

// Exclusive form
template<class InputIterator, class OutputIterator, class UnaryOperation, class T, class BinaryOperation>
std::pair<OutputIterator,T> brick_transform_scan(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op, T init, BinaryOperation binary_op, /*Inclusive*/ std::false_type) noexcept {
    for(; first!=last; ++first, ++result ) {
        *result = init;
__PSTL_PRAGMA_FORCEINLINE
        init = binary_op(init,unary_op(*first));
    }
    return std::make_pair(result,init);
}

// Inclusive form
template<class InputIterator, class OutputIterator, class UnaryOperation, class T, class BinaryOperation>
std::pair<OutputIterator,T> brick_transform_scan(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op, T init, BinaryOperation binary_op, /*Inclusive*/std::true_type) noexcept {
    for(; first!=last; ++first, ++result ) {
__PSTL_PRAGMA_FORCEINLINE
        init = binary_op(init,unary_op(*first));
        *result = init;
    }
    return std::make_pair(result,init);
}

template<class InputIterator, class OutputIterator, class UnaryOperation, class T, class BinaryOperation, class Inclusive, class IsVector>
OutputIterator pattern_transform_scan(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op, T init, BinaryOperation binary_op, Inclusive, IsVector is_vector, /*is_parallel=*/std::false_type ) noexcept {
    return brick_transform_scan(first, last, result, unary_op, init, binary_op, Inclusive()).first;
}

template<class InputIterator, class OutputIterator, class UnaryOperation, class T, class BinaryOperation, class Inclusive, class IsVector>
typename std::enable_if<!std::is_floating_point<T>::value, OutputIterator>::type
pattern_transform_scan(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op, T init, BinaryOperation binary_op, Inclusive, IsVector is_vector, /*is_parallel=*/std::true_type) {
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;

    return except_handler([=]() {
        par_backend::parallel_transform_scan(
            last - first,
            [first, unary_op](difference_type i) mutable {return unary_op(first[i]); },
            init,
            binary_op,
            [first, unary_op, binary_op, is_vector](difference_type i, difference_type j, T init) {
                return brick_transform_reduce(first + i, first + j, init, binary_op, unary_op, is_vector);
            },
            [first, unary_op, binary_op, result](difference_type i, difference_type j, T init) {
                return brick_transform_scan(first + i, first + j, result + i, unary_op, init, binary_op, Inclusive()).second;
            },
            result);
        return result + (last - first);
    });
}

template<class InputIterator, class OutputIterator, class UnaryOperation, class T, class BinaryOperation, class Inclusive, class IsVector>
typename std::enable_if<std::is_floating_point<T>::value, OutputIterator>::type
pattern_transform_scan(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op, T init, BinaryOperation binary_op, Inclusive, IsVector is_vector, /*is_parallel=*/std::true_type) {
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;
    difference_type n = last - first;

    if (n <= 0) {
        return result;
    }
    return except_handler([=, &binary_op]() {
        par_backend::parallel_strict_scan(n, init,
            [first, unary_op, binary_op, result](difference_type i, difference_type len) {
                return brick_transform_scan(first + i, first + (i + len), result + i, unary_op, T{}, binary_op, Inclusive()).second;
            },
            binary_op,
            [result, &binary_op](difference_type i, difference_type len, T initial) {
                return *(std::transform(result + i, result + i + len, result + i,
                    [&initial, &binary_op](const T& x) {
__PSTL_PRAGMA_FORCEINLINE
                        return binary_op(initial, x);
            }) - 1);
        },
        [](T res) { });
        return result + (last - first);
    });
}


//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

template<class ForwardIterator1, class ForwardIterator2, class BinaryOperation>
ForwardIterator2 brick_adjacent_difference(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first, BinaryOperation op, /*is_vector=*/std::false_type) noexcept {
    return std::adjacent_difference(first, last, d_first, op);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryOperation>
ForwardIterator2 brick_adjacent_difference(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first, BinaryOperation op, /*is_vector=*/std::true_type) noexcept {
    assert(first != last);
    typedef typename std::iterator_traits<ForwardIterator1>::value_type T;
    T val(*first);
    *d_first = val;
    unseq_backend::simd_it_walk_2(first, (last - first) - 1, d_first + 1,
        [&op](ForwardIterator1 in, ForwardIterator2 out) {
            T val(op(*(in + 1), *in));
            *out = val;
        });
    return d_first + (last - first);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryOperation, class IsVector>
ForwardIterator2 pattern_adjacent_difference(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first, BinaryOperation op, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_adjacent_difference(first, last, d_first, op, is_vector);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryOperation, class IsVector>
ForwardIterator2 pattern_adjacent_difference(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 d_first, BinaryOperation op, IsVector is_vector, /*is_parallel=*/std::true_type) {
    assert(first != last);
    typedef typename std::iterator_traits<ForwardIterator1>::value_type T;
    T val(*first);
    *d_first = val;
    par_backend::parallel_for(first, last - 1, [&op, is_vector, d_first, first](ForwardIterator1 b, ForwardIterator1 e) {
        ForwardIterator2 d_b = d_first + (b - first);
        brick_it_walk2(b, e, d_b + 1, [&op](ForwardIterator1 in, ForwardIterator2 out) {
            T val(op(*(in + 1), *in));
            *out = val;
        }, is_vector);
    });
    return d_first + (last - first);
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_numeric_impl_H */
