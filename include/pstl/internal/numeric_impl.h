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

#include "execution_impl.h"
#include "unseq_backend_simd.h"

#if __PSTL_USE_PAR_POLICIES
    #include "parallel_backend.h"
#endif

namespace __pstl {
namespace internal {
//------------------------------------------------------------------------
// transform_reduce (version with two binary functions, according to draft N4659)
//------------------------------------------------------------------------

template< class _Tp, class _BinaryOperation1, class _IsArithmeticIsVector>
struct brick_transform_reduce_imp {

    template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation2>
    _Tp operator()(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                 _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2) noexcept {
        return std::inner_product(__first1, __last1, __first2, __init, __binary_op1, __binary_op2);
    }

    template< class _ForwardIterator, class _UnaryOperation>
    _Tp operator()(_ForwardIterator __first, _ForwardIterator __last, _Tp __init,
                 _BinaryOperation1 __binary_op, _UnaryOperation __unary_op) noexcept {
        for (; __first != __last; ++__first) {
            __init = __binary_op(__init, __unary_op(*__first));
        }
        return __init;
    }
};

template< class _Tp>
struct brick_transform_reduce_imp<_Tp, std::plus<_Tp>, /*_IsArithmeticIsVector*/ std::true_type> {

    template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryOperation2>
    _Tp operator()(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Tp __init,
                   std::plus<_Tp>, _BinaryOperation2 __binary_op2) noexcept {
        return unseq_backend::simd_transform_reduce(__first1, __last1 - __first1, __first2, __init, __binary_op2);
    }

    template< class _RandomAccessIterator, class _UnaryOperation>
    _Tp operator()(_RandomAccessIterator __first, _RandomAccessIterator __last, _Tp __init,
                   std::plus<_Tp>, _UnaryOperation __unary_op) noexcept {
        return unseq_backend::simd_transform_reduce(__first, __last - __first, __init, __unary_op);
    }
};

template<class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp brick_transform_reduce(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Tp __init,
                           _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2, /*__is_vector=*/std::true_type) noexcept {
    return internal::brick_transform_reduce_imp< _Tp, _BinaryOperation1, std::integral_constant<bool, std::is_arithmetic<_Tp>::value>>()(__first1, __last1, __first2, __init, __binary_op1, __binary_op2);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp brick_transform_reduce(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Tp __init,
                           _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2, /*__is_vector=*/std::false_type) noexcept {
  return internal::brick_transform_reduce_imp< _Tp, _BinaryOperation1, std::false_type>()(__first1, __last1, __first2, __init, __binary_op1, __binary_op2);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2, class _IsVector>
_Tp pattern_transform_reduce(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Tp __init,
                             _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2, _IsVector __is_vector,
                             /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_transform_reduce(__first1, __last1, __first2, __init, __binary_op1, __binary_op2, __is_vector);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2,
         class _IsVector>
_Tp pattern_transform_reduce(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2, _IsVector __is_vector,  /*is_parallel=*/std::true_type) noexcept {
    return internal::except_handler([&]() {
        return par_backend::parallel_transform_reduce(__first1, __last1,
            [__first1, __first2, __binary_op2](_RandomAccessIterator1 __i) mutable
                                                      { return __binary_op2(*__i, *(__first2 + (__i - __first1))); },
            __init,
            __binary_op1, // Combine
            [__first1, __first2, __binary_op1,
             __binary_op2, __is_vector](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j, _Tp __init) -> _Tp {
                return internal::brick_transform_reduce(__i, __j, __first2 + (__i - __first1),
                                                        __init, __binary_op1, __binary_op2, __is_vector);
        });
    });
}

//------------------------------------------------------------------------
// transform_reduce (version with unary and binary functions)
//------------------------------------------------------------------------

template< class _ForwardIterator, class _Tp, class _UnaryOperation, class _BinaryOperation >
_Tp brick_transform_reduce(_ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op,
                           _UnaryOperation __unary_op, /*is_vector=*/std::true_type) noexcept {
    return internal::brick_transform_reduce_imp< _Tp, _BinaryOperation,
                                       std::integral_constant<bool, std::is_arithmetic<_Tp>::value> >()(__first, __last, __init,
                                                                                                        __binary_op, __unary_op);
}

template< class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation >
_Tp brick_transform_reduce(_ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op,
                           _UnaryOperation __unary_op, /*is_vector=*/std::false_type) noexcept {
    return internal::brick_transform_reduce_imp< _Tp, _BinaryOperation, std::false_type >()(__first, __last, __init, __binary_op, __unary_op);
}

template<class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation, class _IsVector>
_Tp pattern_transform_reduce(_ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op,
                             _UnaryOperation __unary_op, _IsVector __is_vector, /*is_parallel=*/std::false_type ) noexcept {
    return internal::brick_transform_reduce(__first, __last, __init, __binary_op, __unary_op, __is_vector);
}

template<class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation, class _IsVector>
_Tp pattern_transform_reduce(_ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op,
                             _UnaryOperation __unary_op, _IsVector __is_vector, /*is_parallel=*/std::true_type) {
    return internal::except_handler([&]() {
        return par_backend::parallel_transform_reduce(__first, __last,
            [__unary_op](_ForwardIterator __i) mutable {return __unary_op(*__i); },
            __init,
            __binary_op,
            [__unary_op, __binary_op, __is_vector](_ForwardIterator i, _ForwardIterator j, _Tp __init) {
              return internal::brick_transform_reduce(i, j, __init, __binary_op, __unary_op, __is_vector);
        });
    });
}


//------------------------------------------------------------------------
// transform_exclusive_scan
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------

// Exclusive form
template<class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
std::pair<_OutputIterator,_Tp> brick_transform_scan(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                                                    _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op,
                                                    /*Inclusive*/ std::false_type) noexcept {
    for(; __first!=__last; ++__first, ++__result ) {
        *__result = __init;
        __init = __binary_op(__init,__unary_op(*__first));
    }
    return std::make_pair(__result,__init);
}

// Inclusive form
template<class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
std::pair<_OutputIterator,_Tp> brick_transform_scan(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                                                    _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op,
                                                    /*Inclusive*/std::true_type) noexcept {
    for(; __first!=__last; ++__first, ++__result ) {
        __init = __binary_op(__init,__unary_op(*__first));
        *__result = __init;
    }
    return std::make_pair(__result,__init);
}

template<class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation,
         class _Inclusive, class _IsVector>
_OutputIterator pattern_transform_scan(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                                       _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op, _Inclusive,
                                       _IsVector, /*is_parallel=*/std::false_type ) noexcept {
    return internal::brick_transform_scan(__first, __last, __result, __unary_op, __init, __binary_op, _Inclusive()).first;
}

template<class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation,
         class _Inclusive, class _IsVector>
_OutputIterator pattern_transform_scan(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result,
                                       _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op, _Inclusive,
                                       _IsVector __is_vector, /*is_parallel=*/std::true_type ) {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _difference_type;

    return internal::except_handler([=]() {
        par_backend::parallel_transform_scan(
            __last-__first,
            [__first, __unary_op](size_t __i) mutable {return __unary_op(__first[__i]); },
            __init,
            __binary_op,
            [__first, __unary_op, __binary_op, __is_vector](_difference_type __i, _difference_type __j, _Tp __init) {
              return internal::brick_transform_reduce(__first + __i, __first + __j, __init, __binary_op, __unary_op, __is_vector);
        },
        [__first, __unary_op, __binary_op, __result](_difference_type __i, _difference_type __j, _Tp __init) {
          return internal::brick_transform_scan(__first + __i, __first + __j, __result + __i, __unary_op, __init, __binary_op, _Inclusive()).second;
        });
        return __result + (__last - __first);
    });
}


//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

template<class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator brick_adjacent_difference(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __d_first,
                                          _BinaryOperation __op, /*is_vector*/ std::false_type) noexcept {
    return std::adjacent_difference(__first, __last, __d_first, __op);
}

template<class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator brick_adjacent_difference(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __d_first,
                                          _BinaryOperation __op, /*is_vector*/ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, referenced to serial");
    return std::adjacent_difference(__first, __last, __d_first, __op);
}

template<class _ForwardIterator, class _OutputIterator, class _BinaryOperation, class _IsVector>
_OutputIterator pattern_adjacent_difference(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __d_first,
                                            _BinaryOperation __op, _IsVector __is_vector, /*is_parallel*/ std::false_type) noexcept {
    return internal::brick_adjacent_difference(__first, __last, __d_first, __op, __is_vector);
}

template<class _ForwardIterator, class _OutputIterator, class _BinaryOperation, class _IsVector>
_OutputIterator pattern_adjacent_difference(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __d_first,
                                            _BinaryOperation __op, _IsVector __is_vector, /*is_parallel*/ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, referenced to serial");
    return internal::brick_adjacent_difference(__first, __last, __d_first, __op, __is_vector);
}

} // namespace internal
} // namespace __pstl

#endif /* __PSTL_numeric_impl_H */
