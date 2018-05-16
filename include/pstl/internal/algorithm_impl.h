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

#ifndef __PSTL_algorithm_impl_H
#define __PSTL_algorithm_impl_H

#include <iterator>
#include <type_traits>
#include <utility>
#include <functional>
#include <algorithm>

#include "execution_impl.h"
#include "unseq_backend_simd.h"

#if __PSTL_USE_PAR_POLICIES
    #include "parallel_backend.h"
#endif
#include "parallel_impl.h"

namespace __pstl {
namespace internal {

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template<class _ForwardIterator, class _Pred>
bool brick_any_of( const _ForwardIterator __first, const _ForwardIterator __last, _Pred __pred, /*__is_vector=*/std::false_type ) noexcept {
    return std::any_of( __first, __last, __pred );
};

template<class _ForwardIterator, class _Pred>
bool brick_any_of( const _ForwardIterator __first, const _ForwardIterator __last, _Pred __pred, /*__is_vector=*/std::true_type ) noexcept {
    return unseq_backend::simd_or( __first, __last - __first, __pred );
};


template<class _ForwardIterator, class _Pred, class _IsVector>
bool pattern_any_of( _ForwardIterator __first, _ForwardIterator __last, _Pred __pred, _IsVector __is_vector, /*parallel=*/std::false_type ) noexcept {
  return internal::brick_any_of( __first, __last, __pred, __is_vector);
}

template<class _ForwardIterator, class _Pred, class _IsVector>
bool pattern_any_of( _ForwardIterator __first, _ForwardIterator __last, _Pred __pred, _IsVector __is_vector, /*parallel=*/std::true_type ) {
    return internal::except_handler([=]() {
        return internal::parallel_or( __first, __last,
            [__pred, __is_vector]( _ForwardIterator __i, _ForwardIterator __j)
                                 {
                                     return internal::brick_any_of(__i, __j, __pred, __is_vector);
                                 } );
    });
}


// [alg.foreach]
// for_each_n with no policy

template<class _ForwardIterator, class _Size, class _Function>
_ForwardIterator for_each_n_serial(_ForwardIterator __first, _Size __n, _Function __f) {
    for(; __n > 0; ++__first, --__n)
        __f(__first);
    return __first;
}

template<class _ForwardIterator, class _Size, class _Function>
_ForwardIterator for_each_n(_ForwardIterator __first, _Size __n, _Function __f) {
    return internal::for_each_n_serial(__first, __n, [&__f](_ForwardIterator __it) { __f(*__it); });
}

//------------------------------------------------------------------------
// walk1 (pseudo)
//
// walk1 evaluates f(x) for each dereferenced value x drawn from [first,last)
//------------------------------------------------------------------------
template<class _ForwardIterator, class _Function>
void brick_walk1( _ForwardIterator __first, _ForwardIterator __last, _Function __f, /*vector=*/std::false_type ) noexcept {
    for(; __first!=__last; ++__first )
        __f(*__first);
}

template<class _RandomAccessIterator, class _Function>
void brick_walk1( _RandomAccessIterator __first, _RandomAccessIterator __last, _Function __f, /*vector=*/std::true_type ) noexcept {
    unseq_backend::simd_walk_1( __first, __last - __first, __f );
}


template<class _ForwardIterator, class _Function, class _IsVector>
void pattern_walk1( _ForwardIterator __first, _ForwardIterator __last, _Function __f, _IsVector __is_vector,
                    /*parallel=*/std::false_type ) noexcept {
    internal::brick_walk1( __first, __last, __f, __is_vector );
}

template<class _ForwardIterator, class _Function, class _IsVector>
void pattern_walk1( _ForwardIterator __first, _ForwardIterator __last, _Function __f, _IsVector __is_vector,
                    /*parallel=*/std::true_type ) {
    internal::except_handler([=]() {
        par_backend::parallel_for( __first, __last, [__f,__is_vector](_ForwardIterator __i, _ForwardIterator __j) {
            internal::brick_walk1( __i, __j, __f, __is_vector );
        });
    });
}

template<class _ForwardIterator, class _Brick>
void pattern_walk_brick( _ForwardIterator __first, _ForwardIterator __last, _Brick __brick, /*parallel=*/std::false_type ) noexcept {
    __brick(__first, __last);
}

template<class _ForwardIterator, class _Brick>
void pattern_walk_brick( _ForwardIterator __first, _ForwardIterator __last, _Brick __brick, /*parallel=*/std::true_type ) {
    internal::except_handler([=]() {
        par_backend::parallel_for( __first, __last, [__brick](_ForwardIterator __i, _ForwardIterator __j) {
            __brick( __i, __j );
        });
    });
}


//------------------------------------------------------------------------
// it_walk1 (pseudo)
//
// it_walk1 evaluates f(it) for each iterator it drawn from [first,last)
//------------------------------------------------------------------------
template<class _ForwardIterator, class _Function>
void brick_it_walk1( _ForwardIterator __first, _ForwardIterator __last, _Function __f, /*vector=*/std::false_type ) noexcept {
    for(; __first != __last; ++__first )
        __f(__first);
}

template<class _RandomAccessIterator, class _Function>
void brick_it_walk1( _RandomAccessIterator __first, _RandomAccessIterator __last, _Function __f, /*vector=*/std::true_type ) noexcept {
    unseq_backend::simd_it_walk_1(__first, __last - __first, __f);
}

template<class __ForwardIterator, class _Function, class _IsVector>
void pattern_it_walk1( __ForwardIterator __first, __ForwardIterator __last, _Function __f, _IsVector __is_vector,
                       /*parallel=*/std::false_type ) noexcept {
    internal::brick_it_walk1( __first, __last, __f, __is_vector );
}

template<class __ForwardIterator, class _Function, class _IsVector>
void pattern_it_walk1( __ForwardIterator __first, __ForwardIterator __last, _Function __f, _IsVector __is_vector,
                       /*parallel=*/std::true_type ) {
    internal::except_handler([=]() {
        par_backend::parallel_for( __first, __last, [__f,__is_vector](__ForwardIterator __i, __ForwardIterator __j) {
            internal::brick_it_walk1( __i, __j, __f, __is_vector );
        });
    });
}

//------------------------------------------------------------------------
// walk1_n
//------------------------------------------------------------------------
template<class InputIterator, class _Size, class _Function>
InputIterator brick_walk1_n(InputIterator __first, _Size __n, _Function __f, /*_IsVectorTag=*/std::false_type ) {
    return internal::for_each_n( __first, __n, __f ); // calling serial version
}

template<class RandomAccessIterator, class _DifferenceType, class _Function>
RandomAccessIterator brick_walk1_n( RandomAccessIterator __first, _DifferenceType __n, _Function __f,
                                    /*vectorTag=*/std::true_type ) noexcept {
    return unseq_backend::simd_walk_1(__first, __n, __f);
}

template<class InputIterator, class _Size, class _Function, class _IsVector>
InputIterator pattern_walk1_n( InputIterator __first, _Size __n, _Function __f, _IsVector __is_vector,
                               /*is_parallel=*/std::false_type ) noexcept {
    return internal::brick_walk1_n(__first, __n, __f, __is_vector);
}

template<class RandomAccessIterator, class _Size, class _Function, class _IsVector>
RandomAccessIterator pattern_walk1_n( RandomAccessIterator __first, _Size __n, _Function __f, _IsVector __is_vector,
                                      /*is_parallel=*/std::true_type ) {
    internal::pattern_walk1(__first, __first + __n, __f, __is_vector, std::true_type());
    return __first + __n;
}

template<class InputIterator, class _Size, class _Brick>
InputIterator pattern_walk_brick_n( InputIterator __first, _Size __n, _Brick __brick, /*is_parallel=*/std::false_type ) noexcept {
    return __brick(__first, __n);
}

template<class RandomAccessIterator, class _Size, class _Brick>
RandomAccessIterator pattern_walk_brick_n( RandomAccessIterator __first, _Size __n, _Brick __brick, /*is_parallel=*/std::true_type ) {
    return internal::except_handler([=]() {
        par_backend::parallel_for(__first, __first + __n, [__brick](RandomAccessIterator __i, RandomAccessIterator __j) {
            __brick(__i, __j - __i);
        });
        return __first + __n;
    });
}

template<class InputIterator, class _Size, class _Function>
InputIterator brick_it_walk1_n(InputIterator __first, _Size __n, _Function __f, /*_IsVectorTag=*/std::false_type ) {
    return internal::for_each_n_serial(__first, __n, __f); // calling serial version
}

template<class RandomAccessIterator, class _DifferenceType, class _Function>
RandomAccessIterator brick_it_walk1_n( RandomAccessIterator __first, _DifferenceType __n, _Function __f,
                                       /*vectorTag=*/std::true_type ) noexcept {
    return unseq_backend::simd_it_walk_1(__first, __n, __f);
}

template<class InputIterator, class _Size, class _Function, class _IsVector>
InputIterator pattern_it_walk1_n( InputIterator __first, _Size __n, _Function __f, _IsVector __is_vector,
                                  /*is_parallel=*/std::false_type ) noexcept {
    return internal::brick_it_walk1_n(__first, __n, __f, __is_vector);
}

template<class RandomAccessIterator, class _Size, class _Function, class _IsVector>
RandomAccessIterator pattern_it_walk1_n( RandomAccessIterator __first, _Size __n, _Function __f,
                                         _IsVector __is_vector, /*is_parallel=*/std::true_type ) {
    internal::pattern_it_walk1(__first, __first + __n, __f, __is_vector, std::true_type());
    return __first + __n;
}

//------------------------------------------------------------------------
// walk2 (pseudo)
//
// walk2 evaluates f(x,y) for deferenced values (x,y) drawn from [first1,last1) and [first2,...)
//------------------------------------------------------------------------
template<class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_walk2( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f,
                               /*vector=*/std::false_type ) noexcept {
    for(; __first1 != __last1; ++__first1, ++__first2 )
        __f(*__first1, *__first2);
    return __first2;
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_walk2( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f,
                               /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_walk_2(__first1, __last1 - __first1, __first2, __f);
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_walk2_n( _ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Function __f,
                                 /*vector=*/std::false_type ) noexcept {
    for(; __n > 0; --__n, ++__first1, ++__first2 )
        __f(*__first1, *__first2);
    return __first2;
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_walk2_n(_ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Function __f,
                                /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_walk_2(__first1, __n, __first2, __f);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Function, class _IsVector>
_ForwardIterator2 pattern_walk2( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f,
                                 _IsVector __is_vector, /*parallel=*/std::false_type ) noexcept {
    return internal::brick_walk2( __first1,  __last1, __first2, __f, __is_vector );
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Function, class _IsVector>
_ForwardIterator2 pattern_walk2(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f,
                                _IsVector __is_vector, /*parallel=*/std::true_type ) {
    return internal::except_handler([=]() {
        par_backend::parallel_for(
            __first1, __last1,
            [__f, __first1, __first2, __is_vector](_ForwardIterator1 __i, _ForwardIterator1 __j) {
                internal::brick_walk2(__i,__j,__first2 + (__i - __first1), __f, __is_vector);
            }
        );
        return __first2 + (__last1-__first1);
    });
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class Function, class IsVector>
_ForwardIterator2 pattern_walk2_n( _ForwardIterator1 __first1, _Size n, _ForwardIterator2 __first2, Function f,
                                   IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return internal::brick_walk2_n(__first1, n, __first2, f, is_vector);
}

template<class _RandomAccessIterator1, class _Size, class _RandomAccessIterator2, class Function, class IsVector>
_RandomAccessIterator2 pattern_walk2_n(_RandomAccessIterator1 __first1, _Size n, _RandomAccessIterator2 __first2, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    return internal::pattern_walk2(__first1, __first1 + n, __first2, f, is_vector, std::true_type());
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Brick>
_ForwardIterator2 pattern_walk2_brick( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Brick __brick, /*parallel=*/std::false_type ) noexcept {
    return __brick(__first1,__last1,__first2);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _Brick>
_RandomAccessIterator2 pattern_walk2_brick(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Brick __brick, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            __first1, __last1,
            [__first1,__first2, __brick](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                __brick(__i, __j, __first2 + (__i - __first1));
            }
        );
        return __first2 + (__last1 - __first1);
    });
}

template<class _RandomAccessIterator1, class _Size, class _RandomAccessIterator2, class _Brick>
_RandomAccessIterator2 pattern_walk2_brick_n(_RandomAccessIterator1 __first1, _Size __n, _RandomAccessIterator2 __first2, _Brick __brick, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            __first1, __first1 + __n,
            [__first1,__first2, __brick](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                __brick( __i, __j - __i, __first2 + (__i - __first1));
            }
        );
        return __first2 + __n;
    });
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Brick>
_ForwardIterator2 pattern_walk2_brick_n( _ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Brick __brick, /*parallel=*/std::false_type ) noexcept {
    return __brick(__first1, __n, __first2);
}


//------------------------------------------------------------------------
// it_walk2 (pseudo)
//
// it_walk2 evaluates f(it1, it2) for iterators (it1, it2) drawn from [first1,last1) and [first2,...)
//------------------------------------------------------------------------
template<class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_it_walk2( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f, /*vector=*/std::false_type ) noexcept {
    for(; __first1!=__last1; ++__first1, ++__first2 )
        __f(__first1, __first2);
    return __first2;
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_it_walk2( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_it_walk_2(__first1, __last1-__first1, __first2, __f);
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_it_walk2_n( _ForwardIterator1 __first1, _Size n, _ForwardIterator2 __first2, _Function __f, /*vector=*/std::false_type ) noexcept {
    for(; n > 0; --n, ++__first1, ++__first2 )
        __f(__first1, __first2);
    return __first2;
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function>
_ForwardIterator2 brick_it_walk2_n(_ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Function __f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_it_walk_2(__first1, __n, __first2, __f);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Function, class _IsVector>
_ForwardIterator2 pattern_it_walk2( _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f, _IsVector __is_vector, /*parallel=*/std::false_type ) noexcept {
    return internal::brick_it_walk2(__first1,__last1,__first2,__f,__is_vector);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _Function, class _IsVector>
_RandomAccessIterator2 pattern_it_walk2(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Function __f, _IsVector __is_vector, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            __first1, __last1,
            [__f,__first1,__first2,__is_vector](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                internal::brick_it_walk2( __i, __j, __first2 +(__i - __first1), __f, __is_vector);
            }
        );
        return __first2 + (__last1 - __first1);
    });
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function, class _IsVector>
_ForwardIterator2 pattern_it_walk2_n( _ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Function __f, _IsVector __is_vector, /*parallel=*/std::false_type ) noexcept {
    return internal::brick_it_walk2_n(__first1, __n, __first2, __f, __is_vector);
}

template<class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function, class _IsVector>
_ForwardIterator2 pattern_it_walk2_n(_ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Function __f, _IsVector __is_vector, /*parallel=*/std::true_type ) {
    return internal::pattern_it_walk2(__first1, __first1 + __n, __first2, __f, __is_vector, std::true_type());
}

//------------------------------------------------------------------------
// walk3 (pseudo)
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------
template<class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator3, class _Function>
_ForwardIterator3 brick_walk3( _ForwardIterator1 __first1, _ForwardIterator1 last1, _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f, /*vector=*/std::false_type ) noexcept {
    for(; __first1!=last1; ++__first1, ++__first2, ++__first3 )
        __f(*__first1, *__first2, *__first3);
    return __first3;
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Function>
_RandomAccessIterator3 brick_walk3( _RandomAccessIterator1 __first1, _RandomAccessIterator1 last1, _RandomAccessIterator2 __first2, _RandomAccessIterator3 __first3, _Function __f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_walk_3(__first1, last1-__first1, __first2, __first3, __f);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator3, class _Function, class _IsVector>
_ForwardIterator3 pattern_walk3( _ForwardIterator1 __first1, _ForwardIterator1 last1, _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f, _IsVector __is_vector, /*parallel=*/std::false_type ) noexcept {
    return internal::brick_walk3(__first1, last1, __first2, __first3, __f, __is_vector);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Function, class _IsVector>
_RandomAccessIterator3 pattern_walk3(_RandomAccessIterator1 __first1, _RandomAccessIterator1 last1, _RandomAccessIterator2 __first2, _RandomAccessIterator3 __first3, _Function __f, _IsVector __is_vector, /*parallel=*/std::true_type ) {
    return internal::except_handler([=]() {
        par_backend::parallel_for(
            __first1, last1,
            [__f, __first1, __first2, __first3, __is_vector](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
            internal::brick_walk3(__i, __j, __first2 + (__i - __first1), __first3 + (__i - __first1), __f, __is_vector);
        });
        return __first3+(last1-__first1);
    });
}

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool brick_equal(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _BinaryPredicate __p, /* IsVector = */ std::false_type) noexcept {
    return std::equal(__first1, __last1, __first2, __p);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate>
bool brick_equal(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _BinaryPredicate __p, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_first(__first1, __last1 - __first1, __first2, not_pred<_BinaryPredicate>(__p)).first == __last1;
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
bool pattern_equal(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _BinaryPredicate __p, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_equal(__first1, __last1, __first2, __p, __is_vector);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate, class _IsVector>
bool pattern_equal(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _BinaryPredicate __p, _IsVector __is_vector, /*is_parallel=*/std::true_type) {
    return internal::except_handler([=]() {
        return !internal::parallel_or(__first1, __last1,
            [__first1, __first2, __p, __is_vector](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j)
                                      {
                                        return !brick_equal(__i, __j, __first2 + (__i - __first1), __p, __is_vector);
                                      });
    });
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------
template<class _ForwardIterator, class _Predicate>
_ForwardIterator brick_find_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, /*is_vector=*/std::false_type) noexcept {
    return std::find_if(__first, __last, __pred);
}

template<class _RandomAccessIterator, class _Predicate>
_RandomAccessIterator brick_find_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _Predicate __pred, /*is_vector=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _size_type;
    return unseq_backend::simd_first(__first, _size_type(0), __last - __first,
        [&__pred](_RandomAccessIterator __it, _size_type __i) {return __pred(__it[__i]); });
}

template<class _ForwardIterator, class _Predicate, class _IsVector>
_ForwardIterator pattern_find_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, _IsVector __is_vector,
                                 /*is_parallel=*/std::false_type ) noexcept {
    return internal::brick_find_if( __first, __last, __pred, __is_vector);
}

template<class _ForwardIterator, class _Predicate, class _IsVector>
_ForwardIterator pattern_find_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, _IsVector __is_vector,
                                 /*is_parallel=*/std::true_type) {
    return internal::except_handler([=]() {
        return internal::parallel_find(__first, __last, [__pred, __is_vector](_ForwardIterator __i, _ForwardIterator __j) {
            return internal::brick_find_if(__i, __j, __pred, __is_vector);
        },
        std::less<typename std::iterator_traits<_ForwardIterator>::difference_type>(), /*is___first=*/true);
    });
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

// find the first occurrence of the subsequence [s_first, s_last)
//   or the  last occurrence of the subsequence in the range [first, last)
// b_first determines what occurrence we want to find (first or last)
template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate, class _IsVector>
_RandomAccessIterator1 find_subrange(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
    _RandomAccessIterator1 __global_last, _RandomAccessIterator2 __s_first, _RandomAccessIterator2 __s_last,
    _BinaryPredicate __pred, bool __b_first, _IsVector __is_vector) noexcept {
    typedef typename std::iterator_traits<_RandomAccessIterator2>::value_type _value_type;
    auto  __n2 = __s_last - __s_first;
    if (__n2 < 1) {
        return __b_first ? __first : __last;
    }

    auto  __n1 = __global_last - __first;
    if (__n1 < __n2) {
        return __last;
    }

    auto __cur = __last;
    while (__first != __last && (__global_last - __first >= __n2)) {
        // find position of *s_first in [first, last) (it can be start of subsequence)
        __first = internal::brick_find_if(__first, __last,
                                          internal::equal_value_by_pred<_value_type, _BinaryPredicate>(*__s_first, __pred),
                                          __is_vector);

        // if position that was found previously is the start of subsequence
        // then we can exit the loop (b_first == true) or keep the position
        // (b_first == false)
        if (__first != __last && (__global_last - __first >= __n2) &&
            internal::brick_equal(__s_first + 1, __s_last, __first + 1, __pred, __is_vector)) {
            if (__b_first) {
                return __first;
            }
            else {
                __cur = __first;
            }
        }
        else if (__first == __last) {
            break;
        }
        else {}

        // in case of b_first == false we try to find new start position
        // for the next subsequence
        ++__first;
    }
    return __cur;
}

template<class _RandomAccessIterator, class _Size, class _Tp, class _BinaryPredicate, class _IsVector>
_RandomAccessIterator find_subrange(_RandomAccessIterator __first, _RandomAccessIterator __last,
    _RandomAccessIterator __global_last, _Size __count, const _Tp& __value,
    _BinaryPredicate __pred, _IsVector __is_vector) noexcept {
    if (__global_last - __first < __count || __count < 1) {
        return __last; // According to the standard last shall be returned when count < 1
    }

    auto __n = __global_last - __first;
    auto __unary_pred = internal::equal_value_by_pred<_Tp, _BinaryPredicate>(__value, __pred);
    while (__first != __last && (__global_last - __first >= __count)) {
        __first = brick_find_if(__first, __last, __unary_pred, __is_vector);

        // check that all of elements in [first+1, first+count) equal to value
        if (__first != __last && (__global_last - __first >= __count) &&
            !internal::brick_any_of(__first + 1, __first + __count,
                                    internal::not_pred<decltype(__unary_pred)>(__unary_pred), __is_vector)) {
                return __first;
        }
        else if (__first == __last) {
            break;
        }
        else {
            ++__first;
        }
    }
    return __last;
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 brick_find_end(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/std::false_type) noexcept {
    return std::find_end(__first, __last, __s_first, __s_last, __pred);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 brick_find_end(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/std::true_type) noexcept {
    return internal::find_subrange(__first, __last, __last, __s_first, __s_last, __pred, false, std::true_type());
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
_ForwardIterator1 pattern_find_end(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_find_end(__first, __last, __s_first, __s_last, __pred, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
_ForwardIterator1 pattern_find_end(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (__last - __first == __s_last - __s_first) {
        const bool __res = internal::pattern_equal(__first, __last, __s_first, __pred, __is_vector, std::true_type());
        return __res ? __first : __last;
    }
    else {
        return except_handler([=]() {
            return internal::parallel_find(__first, __last, [__first, __last, __s_first, __s_last, __pred, __is_vector](_ForwardIterator1 __i, _ForwardIterator1 __j) {
                return internal::find_subrange(__i, __j, __last, __s_first, __s_last, __pred, false, __is_vector);
            },
            std::greater<typename std::iterator_traits<_ForwardIterator1>::difference_type>(), /*is_first=*/false);
        });
    }
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------
template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 brick_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/std::false_type) noexcept {
    return std::find_first_of(__first, __last, __s_first, __s_last, __pred);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 brick_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/std::true_type) noexcept {
    return unseq_backend::simd_find_first_of(__first, __last, __s_first, __s_last, __pred);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
_ForwardIterator1 pattern_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_find_first_of(__first, __last, __s_first, __s_last, __pred, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
_ForwardIterator1 pattern_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    return except_handler([=]() {
        return internal::parallel_find(__first, __last, [__s_first, __s_last, __pred, __is_vector](_ForwardIterator1 __i, _ForwardIterator1 __j) {
            return internal::brick_find_first_of(__i, __j, __s_first, __s_last, __pred, __is_vector);
        },
        std::less<typename std::iterator_traits<_ForwardIterator1>::difference_type>(), /*is_first=*/true);
    });
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------
template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 brick_search(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*vector=*/std::false_type) noexcept {
    return std::search(__first, __last, __s_first, __s_last, __pred);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 brick_search(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*vector=*/std::true_type) noexcept {
    return internal::find_subrange(__first, __last, __last, __s_first, __s_last, __pred, true, std::true_type());
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
_ForwardIterator1 pattern_search(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_search(__first, __last, __s_first, __s_last, __pred, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate, class _IsVector>
_ForwardIterator1 pattern_search(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (__last - __first == __s_last - __s_first) {
        const bool __res = internal::pattern_equal(__first, __last, __s_first, __pred, __is_vector, std::true_type());
        return __res ? __first : __last;
    }
    else {
        return except_handler([=]() {
            return internal::parallel_find(__first, __last, [__last, __s_first, __s_last, __pred, __is_vector](_ForwardIterator1 __i, _ForwardIterator1 __j) {
                return internal::find_subrange(__i, __j, __last, __s_first, __s_last, __pred, true, __is_vector);
            },
            std::less<typename std::iterator_traits<_ForwardIterator1>::difference_type>(), /*is_first=*/true);
        });
    }
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------
template<class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
_ForwardIterator brick_search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value, _BinaryPredicate __pred, /*vector=*/std::false_type) noexcept {
    return std::search_n(__first, __last, __count, __value, __pred);
}

template<class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
_ForwardIterator brick_search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value, _BinaryPredicate __pred, /*vector=*/std::true_type) noexcept {
    return internal::find_subrange(__first, __last, __last, __count, __value, __pred, std::true_type());
}

template<class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate, class IsVector>
_ForwardIterator pattern_search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value, _BinaryPredicate __pred, IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_search_n(__first, __last, __count, __value, __pred, __is_vector);
}

template<class _RandomAccessIterator, class _Size, class _Tp, class _BinaryPredicate, class IsVector>
_RandomAccessIterator pattern_search_n(_RandomAccessIterator __first, _RandomAccessIterator __last, _Size __count, const _Tp& __value, _BinaryPredicate __pred, IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (__last - __first == __count) {
        const bool __result = !internal::pattern_any_of(__first, __last,
            [&__value, &__pred](const _Tp& __val) {return !__pred(__val, __value); },
            __is_vector, /*is_parallel*/ std::true_type());
        return __result ? __first : __last;
    }
    else {
        return except_handler([__first, __last, __count, &__value, __pred, __is_vector]() {
            return internal::parallel_find(__first, __last, [__last, __count, &__value, __pred, __is_vector](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                return internal::find_subrange(__i, __j, __last, __count, __value, __pred, __is_vector);
            },
            std::less<typename std::iterator_traits<_RandomAccessIterator>::difference_type>(), /*is_first=*/true);
        });
    }
}

//------------------------------------------------------------------------
// copy_n
//------------------------------------------------------------------------

template<class _ForwardIterator, class _Size, class _OutputIterator>
_OutputIterator brick_copy_n(_ForwardIterator __first, _Size __n, _OutputIterator __result, /*vector=*/std::false_type) noexcept {
    return std::copy_n(__first, __n, __result);
}

template<class _ForwardIterator, class _Size, class _OutputIterator>
_OutputIterator brick_copy_n(_ForwardIterator __first, _Size __n, _OutputIterator __result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_copy_move(__first, __n, __result,
        [](_ForwardIterator __first, _OutputIterator __result) {
            *__result = *__first;
    });
}

//------------------------------------------------------------------------
// copy
//------------------------------------------------------------------------
template<class _ForwardIterator, class _OutputIterator>
_OutputIterator brick_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, /*vector=*/std::false_type) noexcept {
    return std::copy(__first, __last, __result);
}

template<class _RandomAccessIterator, class _OutputIterator>
_OutputIterator brick_copy(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_copy_move(__first, __last - __first, __result,
        [](_RandomAccessIterator __first, _OutputIterator __result) {
            *__result = *__first;
    });
}

//------------------------------------------------------------------------
// move
//------------------------------------------------------------------------
template<class _ForwardIterator, class _OutputIterator>
_OutputIterator brick_move(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, /*vector=*/std::false_type) noexcept {
    return std::move(__first, __last, __result);
}

template<class _RandomAccessIterator, class _OutputIterator>
_OutputIterator brick_move(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_copy_move(__first, __last - __first, __result,
        [](_RandomAccessIterator __first, _OutputIterator __result) {
        *__result = std::move(*__first);
    });
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------
template<class _ForwardIterator, class OutputIterator, class _UnaryPredicate>
OutputIterator brick_copy_if(_ForwardIterator __first, _ForwardIterator __last, OutputIterator __result, _UnaryPredicate __pred, /*vector=*/std::false_type) noexcept {
    return std::copy_if(__first, __last, __result, __pred);
}

template<class _ForwardIterator, class OutputIterator, class _UnaryPredicate>
OutputIterator brick_copy_if(_ForwardIterator __first, _ForwardIterator __last, OutputIterator __result, _UnaryPredicate __pred, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    return unseq_backend::simd_copy_if(__first, __last - __first, __result, __pred);
#else
    return std::copy_if(__first, __last, __result, __pred);
#endif
}

// TODO: Try to use transform_reduce for combining brick_copy_if_phase1 on IsVector.
template<class _DifferenceType, class _ForwardIterator, class _UnaryPredicate>
std::pair<_DifferenceType, _DifferenceType> brick_calc_mask_1(
    _ForwardIterator __first, _ForwardIterator __last, bool* __restrict __mask, _UnaryPredicate __pred, /*vector=*/std::false_type) noexcept {
    auto __count_true  = _DifferenceType(0);
    auto __count_false = _DifferenceType(0);
    auto __size = std::distance(__first, __last);

    for (; __first != __last; ++__first, ++__mask) {
        *__mask = __pred(*__first);
        if (*__mask) {
            ++__count_true;
        }
    }
    return std::make_pair(__count_true, __size - __count_true);
}

template<class _DifferenceType, class _RandomAccessIterator, class _UnaryPredicate>
std::pair<_DifferenceType, _DifferenceType> brick_calc_mask_1(
    _RandomAccessIterator __first, _RandomAccessIterator __last, bool* __restrict __mask, _UnaryPredicate __pred, /*vector=*/std::true_type) noexcept {
    auto __result = unseq_backend::simd_calc_mask_1(__first, __last - __first, __mask, __pred);
    return std::make_pair(__result, (__last - __first) - __result);
}

template<class _ForwardIterator, class _OutputIterator>
void brick_copy_by_mask(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, bool* __mask, /*vector=*/std::false_type) noexcept {
    for (; __first != __last; ++__first, ++__mask) {
        if (*__mask) {
            *__result = *__first;
            ++__result;
        }
    }
}

template<class _ForwardIterator, class _OutputIterator>
void brick_copy_by_mask(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, bool* __restrict __mask, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    unseq_backend::simd_copy_by_mask(__first, __last - __first, __result, __mask);
#else
    internal::brick_copy_by_mask(__first, __last, __result, __mask, std::false_type());
#endif

}

template<class _ForwardIterator, class _OutputIterator1, class _OutputIterator2>
void brick_partition_by_mask(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator1 __out_true,
    _OutputIterator2 __out_false, bool* __mask, /*vector=*/std::false_type) noexcept {
    for (; __first != __last; ++__first, ++__mask) {
        if (*__mask) {
            *__out_true = *__first;
            ++__out_true;
        }
        else {
            *__out_false = *__first;
            ++__out_false;
        }
    }
}

template<class _RandomAccessIterator, class _OutputIterator1, class _OutputIterator2>
void brick_partition_by_mask(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator1 __out_true,
    _OutputIterator2 __out_false, bool* __mask, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    unseq_backend::simd_partition_by_mask(__first, __last - __first, __out_true, __out_false, __mask);
#else
    internal::brick_partition_by_mask(__first, __last, __out_true, __out_false, __mask, std::false_type());
#endif
}

template<class _ForwardIterator, class _OutputIterator, class _UnaryPredicate, class _IsVector>
_OutputIterator pattern_copy_if(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, _UnaryPredicate __pred, _IsVector __is_vector, /*parallel=*/std::false_type) noexcept {
    return internal::brick_copy_if(__first, __last, __result, __pred, __is_vector);
}

template<class _RandomAccessIterator, class _OutputIterator, class _UnaryPredicate, class _IsVector>
_OutputIterator pattern_copy_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result, _UnaryPredicate __pred, _IsVector __is_vector, /*parallel=*/std::true_type) {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _difference_type;
    const _difference_type __n = __last-__first;
    if( _difference_type(1) < __n ) {
        par_backend::buffer<bool> __mask_buf(__n);
        if( __mask_buf ) {
            return internal::except_handler([__n, __first, __last, __result, __is_vector, __pred, &__mask_buf]() {
                bool* __mask = __mask_buf.get();
                _difference_type __m;
                par_backend::parallel_strict_scan( __n, _difference_type(0),
                    [=](_difference_type __i, _difference_type __len) {                               // Reduce
                        return internal::brick_calc_mask_1<_difference_type>(__first + __i, __first + (__i + __len),
                                                                             __mask + __i,
                                                                             __pred,
                                                                             __is_vector).first;
                    },
                    std::plus<_difference_type>(),                                               // Combine
                    [=](_difference_type __i, _difference_type __len, _difference_type __initial) {      // Scan
                        internal::brick_copy_by_mask(__first + __i, __first + (__i + __len),
                                                     __result + __initial,
                                                     __mask + __i,
                                                     __is_vector);
                    },
                    [&__m](_difference_type __total) {__m = __total;});
                return __result + __m;
            });
        }
    }
    // _Out of memory or trivial sequence - use serial algorithm
    return brick_copy_if(__first, __last, __result, __pred, __is_vector);
}

//------------------------------------------------------------------------
// unique
//------------------------------------------------------------------------

template<class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator brick_unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, /*is_vector=*/std::false_type) noexcept {
    return std::unique(__first, __last, __pred);
}

template<class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator brick_unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::unique(__first, __last, __pred);
}

template<class _ForwardIterator, class _BinaryPredicate, class _IsVector>
_ForwardIterator pattern_unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_unique(__first, __last, __pred, __is_vector);
}

template<class _ForwardIterator, class _BinaryPredicate, class _IsVector>
_ForwardIterator pattern_unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_unique(__first, __last, __pred, __is_vector);
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template<class _ForwardIterator, class OutputIterator, class _BinaryPredicate>
OutputIterator brick_unique_copy(_ForwardIterator __first, _ForwardIterator __last, OutputIterator __result, _BinaryPredicate __pred, /*vector=*/std::false_type) noexcept {
    return std::unique_copy(__first, __last, __result, __pred);
}

template<class _RandomAccessIterator, class OutputIterator, class _BinaryPredicate>
OutputIterator brick_unique_copy(_RandomAccessIterator __first, _RandomAccessIterator __last, OutputIterator __result, _BinaryPredicate __pred, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    return unseq_backend::simd_unique_copy(__first, __last - __first, __result, __pred);
#else
    return std::unique_copy(__first, __last, __result, __pred);
#endif
}

template<class _ForwardIterator, class OutputIterator, class _BinaryPredicate, class _IsVector>
OutputIterator pattern_unique_copy(_ForwardIterator __first, _ForwardIterator __last, OutputIterator __result, _BinaryPredicate __pred, _IsVector __is_vector, /*parallel=*/std::false_type) noexcept {
    return internal::brick_unique_copy(__first, __last, __result, __pred, __is_vector);
}

template<class _DifferenceType, class _RandomAccessIterator, class _BinaryPredicate>
_DifferenceType brick_calc_mask_2(_RandomAccessIterator __first, _RandomAccessIterator __last, bool* __restrict __mask, _BinaryPredicate __pred, /*vector=*/std::false_type) noexcept {
    _DifferenceType count = 0;
    for (; __first != __last; ++__first, ++__mask) {
        *__mask = !__pred(*__first, *(__first - 1));
        count += *__mask;
    }
    return count;
}

template<class _DifferenceType, class _RandomAccessIterator, class _BinaryPredicate>
_DifferenceType brick_calc_mask_2(_RandomAccessIterator __first, _RandomAccessIterator __last, bool* __restrict __mask, _BinaryPredicate __pred, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_calc_mask_2(__first, __last - __first, __mask, __pred);
}

template<class _RandomAccessIterator, class _OutputIterator, class _BinaryPredicate, class _IsVector>
_OutputIterator pattern_unique_copy(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result, _BinaryPredicate __pred, _IsVector __is_vector, /*parallel=*/std::true_type) {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _difference_type;
    const _difference_type __n = __last - __first;
    if( _difference_type(2) < __n ) {
        par_backend::buffer<bool> __mask_buf(__n);
        if( _difference_type(2) < __n && __mask_buf ) {
          return internal::except_handler([__n, __first, __result, __pred, __is_vector, &__mask_buf]() {
                bool* __mask = __mask_buf.get();
                _difference_type __m;
                par_backend::parallel_strict_scan( __n, _difference_type(0),
                    [=](_difference_type __i, _difference_type __len) -> _difference_type {          // Reduce
                        _difference_type __extra = 0;
                        if( __i == 0 ) {
                            // Special boundary case
                            __mask[__i] = true;
                            if( --__len == 0 ) return 1;
                            ++__i;
                            ++__extra;
                        }
                        return brick_calc_mask_2<_difference_type>(__first + __i, __first + (__i + __len),
                                                                   __mask + __i,
                                                                   __pred,
                                                                   __is_vector) + __extra;
                    },
                    std::plus<_difference_type>(),                                               // Combine
                    [=](_difference_type __i, _difference_type __len, _difference_type __initial) {      // Scan
                        // Phase 2 is same as for pattern_copy_if
                        internal::brick_copy_by_mask(__first + __i, __first + (__i + __len),
                                                     __result + __initial,
                                                     __mask + __i,
                                                     __is_vector);
                    },
                    [&__m](_difference_type __total) {__m = __total;});
                return __result + __m;
            });
        }
    }
    // Out of memory or trivial sequence - use serial algorithm
    return brick_unique_copy(__first, __last, __result, __pred, __is_vector);
}

//------------------------------------------------------------------------
// swap_ranges
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 brick_swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, /*__is_vector=*/std::false_type) noexcept {
    return std::swap_ranges(__first1, __last1, __first2);
}

template<class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2 brick_swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::swap_ranges(__first1, __last1, __first2);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _IsVector>
_ForwardIterator2 pattern_swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_swap_ranges(__first1, __last1, __first2, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _IsVector>
_ForwardIterator2 pattern_swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_swap_ranges(__first1, __last1, __first2, __is_vector);
}

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------

template<class _BidirectionalIterator>
void brick_reverse(_BidirectionalIterator __first, _BidirectionalIterator __last,/*__is_vector=*/std::false_type) noexcept {
    std::reverse(__first, __last);
}

template<class _BidirectionalIterator>
void brick_reverse(_BidirectionalIterator __first, _BidirectionalIterator __last,/*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    internal::brick_reverse(__first, __last, std::false_type());
}

template<class _BidirectionalIterator, class _IsVector>
void pattern_reverse(_BidirectionalIterator __first, _BidirectionalIterator __last, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    internal::brick_reverse(__first, __last, __is_vector);
}

template<class _BidirectionalIterator, class _IsVector>
void pattern_reverse(_BidirectionalIterator __first, _BidirectionalIterator __last, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    internal::brick_reverse(__first, __last, __is_vector);
}

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------

template<class _BidirectionalIterator, class _OutputIterator>
_OutputIterator brick_reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __d_first, /*is_vector=*/std::false_type) noexcept {
    return std::reverse_copy(__first, __last, __d_first);
}

template<class _BidirectionalIterator, class _OutputIterator>
_OutputIterator brick_reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __d_first, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return internal::brick_reverse_copy(__first, __last, __d_first, std::false_type());
}

template<class _BidirectionalIterator, class _OutputIterator, class _IsVector>
_OutputIterator pattern_reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __d_first, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_reverse_copy(__first, __last, __d_first, __is_vector);
}

template<class _BidirectionalIterator, class _OutputIterator, class _IsVector>
_OutputIterator pattern_reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __d_first, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_reverse_copy(__first, __last, __d_first, __is_vector);
}

//------------------------------------------------------------------------
// rotate
//------------------------------------------------------------------------
template<class _ForwardIterator>
_ForwardIterator brick_rotate(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, /*is_vector=*/std::false_type) noexcept {
#if __PSTL_CPP11_STD_ROTATE_BROKEN
    std::rotate(__first, __middle, __last);
    return std::next(__first, std::distance(__middle, __last));
#else
    return std::rotate(__first, __middle, __last);
#endif
}

template<class _ForwardIterator>
_ForwardIterator brick_rotate(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return internal::brick_rotate(__first, __middle, __last, std::false_type());
}

template<class _ForwardIterator, class _IsVector>
_ForwardIterator pattern_rotate(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_rotate(__first, __middle, __last, __is_vector);
}

template<class _ForwardIterator, class _IsVector>
_ForwardIterator pattern_rotate(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_rotate(__first, __middle, __last, __is_vector);
}

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------

template<class _ForwardIterator, class _OutputIterator>
_OutputIterator brick_rotate_copy(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _OutputIterator __result, /*__is_vector=*/std::false_type) noexcept {
    return std::rotate_copy(__first, __middle, __last, __result);
}

template<class _ForwardIterator, class _OutputIterator>
_OutputIterator brick_rotate_copy(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _OutputIterator __result, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::rotate_copy(__first, __middle, __last, __result);
}

template<class _ForwardIterator, class _OutputIterator, class _IsVector>
_OutputIterator pattern_rotate_copy(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _OutputIterator __result, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_rotate_copy(__first, __middle, __last, __result, __is_vector);
}

template<class _ForwardIterator, class _OutputIterator, class _IsVector>
_OutputIterator pattern_rotate_copy(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _OutputIterator __result, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_rotate_copy(__first, __middle, __last, __result, __is_vector);
}

//------------------------------------------------------------------------
// is_partitioned
//------------------------------------------------------------------------

template<class _ForwardIterator, class _UnaryPredicate>
bool brick_is_partitioned(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, /*is_vector=*/std::false_type) noexcept
{
    return std::is_partitioned(__first, __last, __pred);
}

template<class _ForwardIterator, class _UnaryPredicate>
bool brick_is_partitioned(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return internal::brick_is_partitioned(__first, __last, __pred, std::false_type());
}

template<class _ForwardIterator, class _UnaryPredicate, class _IsVector>
bool pattern_is_partitioned(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_is_partitioned(__first, __last, __pred, __is_vector);
}


template<class _ForwardIterator, class _UnaryPredicate, class _IsVector>
bool pattern_is_partitioned(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_is_partitioned(__first, __last, __pred, __is_vector);
}

//------------------------------------------------------------------------
// partition
//------------------------------------------------------------------------

template<class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator brick_partition(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, /*is_vector=*/std::false_type) noexcept {
    return std::partition(__first, __last, __pred);
}

template<class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator brick_partition(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::partition(__first, __last, __pred);
}

template<class _ForwardIterator, class _UnaryPredicate, class _IsVector>
_ForwardIterator pattern_partition(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_partition(__first, __last, __pred, __is_vector);
}

template<class _ForwardIterator, class _UnaryPredicate, class _IsVector>
_ForwardIterator pattern_partition(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_partition(__first, __last, __pred, __is_vector);
}

//------------------------------------------------------------------------
// stable_partition
//------------------------------------------------------------------------

template<class _BidirectionalIterator, class _UnaryPredicate>
_BidirectionalIterator brick_stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _UnaryPredicate __pred, /*__is_vector=*/std::false_type) noexcept {
    return std::stable_partition(__first, __last, __pred);
}

template<class _BidirectionalIterator, class _UnaryPredicate>
_BidirectionalIterator brick_stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _UnaryPredicate __pred, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::stable_partition(__first, __last, __pred);
}

template<class _BidirectionalIterator, class _UnaryPredicate, class _IsVector>
_BidirectionalIterator pattern_stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallelization=*/std::false_type) noexcept {
    return internal::brick_stable_partition(__first, __last, __pred, __is_vector);
}

template<class _BidirectionalIterator, class _UnaryPredicate, class _IsVector>
_BidirectionalIterator pattern_stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallelization=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_stable_partition(__first, __last, __pred, __is_vector);
}

//------------------------------------------------------------------------
// partition_copy
//------------------------------------------------------------------------

template<class _ForwardIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPredicate>
std::pair<_OutputIterator1, _OutputIterator2>
brick_partition_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator1 __out_true, _OutputIterator2 __out_false, _UnaryPredicate __pred, /*is_vector=*/std::false_type) noexcept {
    return std::partition_copy(__first, __last, __out_true, __out_false, __pred);
}

template<class _ForwardIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPredicate>
std::pair<_OutputIterator1, _OutputIterator2>
brick_partition_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator1 __out_true, _OutputIterator2 __out_false, _UnaryPredicate __pred, /*is_vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    return unseq_backend::simd_partition_copy(__first, __last - __first, __out_true, __out_false, __pred);
#else
    return std::partition_copy(__first, __last, __out_true, __out_false, __pred);
#endif
}

template<class _ForwardIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPredicate, class _IsVector>
std::pair<_OutputIterator1, _OutputIterator2>
pattern_partition_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator1 __out_true, _OutputIterator2 __out_false, _UnaryPredicate __pred, _IsVector __is_vector,/*is_parallelization=*/std::false_type) noexcept {
    return internal::brick_partition_copy(__first, __last, __out_true, __out_false, __pred, __is_vector);
}

template<class _RandomAccessIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPredicate, class _IsVector>
std::pair<_OutputIterator1, _OutputIterator2>
pattern_partition_copy(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator1 __out_true, _OutputIterator2 __out_false, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallelization=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _difference_type;
    typedef std::pair<_difference_type, _difference_type> _return_type;
    const _difference_type __n = __last - __first;
    if (_difference_type(1) < __n) {
        par_backend::buffer<bool> __mask_buf(__n);
        if (__mask_buf) {
            return internal::except_handler([__n, __first, __last, __out_true, __out_false, __is_vector, __pred, &__mask_buf]() {
                bool* __mask = __mask_buf.get();
                _return_type __m;
                par_backend::parallel_strict_scan(__n, std::make_pair(_difference_type(0), _difference_type(0)),
                    [=](_difference_type __i, _difference_type __len) {                             // Reduce
                        return internal::brick_calc_mask_1<_difference_type>(__first + __i, __first + (__i + __len),
                        __mask + __i,
                        __pred,
                        __is_vector);
                },
                    [](const _return_type& __x, const _return_type& __y)-> _return_type {
                    return std::make_pair(__x.first + __y.first, __x.second + __y.second);
                },                                                                            // Combine
                    [=](_difference_type __i, _difference_type __len, _return_type __initial) {        // Scan
                        internal::brick_partition_by_mask(__first + __i, __first + (__i + __len),
                        __out_true + __initial.first,
                        __out_false + __initial.second,
                        __mask + __i,
                        __is_vector);
                },
                [&__m](_return_type __total) {__m = __total; });
                return std::make_pair(__out_true + __m.first, __out_false + __m.second);
            });
        }
    }
    // Out of memory or trivial sequence - use serial algorithm
    return internal::brick_partition_copy(__first, __last, __out_true, __out_false, __pred, __is_vector);
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------

template<class _RandomAccessIterator, class _Compare, class _IsVector, class _IsMoveConstructible>
void pattern_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _IsVector /*is_vector*/, /*is_parallel=*/std::false_type, _IsMoveConstructible) noexcept {
    std::sort(__first, __last, __comp);
}


template<class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _IsVector /*is_vector*/, /*is_parallel=*/std::true_type, /*is_move_constructible=*/std::true_type ) {
    internal::except_handler([=]() {
        par_backend::parallel_stable_sort(__first, __last, __comp,
            [](_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) { std::sort(__first, __last, __comp); });
    });
}

//------------------------------------------------------------------------
// stable_sort
//------------------------------------------------------------------------

template<class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _IsVector /*is_vector*/, /*is_parallel=*/std::false_type) noexcept {
    std::stable_sort(__first, __last, __comp);
}

template<class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _IsVector /*is_vector*/, /*is_parallel=*/std::true_type) {
    internal::except_handler([=]() {
        par_backend::parallel_stable_sort(__first, __last, __comp,
            [](_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
            std::stable_sort(__first, __last, __comp);
        });
    });
}

//------------------------------------------------------------------------
// partial_sort
//------------------------------------------------------------------------

template<class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp, _IsVector, /*is_parallel=*/std::false_type) noexcept {
    std::partial_sort(__first, __middle, __last, __comp);
}

template <class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp, _IsVector, /*is_parallel=*/std::true_type) noexcept {
    par_backend::parallel_partial_sort(__first, __middle, __last, __comp);
}

//------------------------------------------------------------------------
// partial_sort_copy
//------------------------------------------------------------------------

template<class _ForwardIterator, class _RandomAccessIterator, class _Compare>
_RandomAccessIterator brick_partial_sort_copy(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp, /*__is_vector*/std::false_type) noexcept {
    return std::partial_sort_copy(__first, __last, __d_first, __d_last, __comp);
}

template<class _ForwardIterator, class _RandomAccessIterator, class _Compare>
_RandomAccessIterator brick_partial_sort_copy(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp, /*__is_vector*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::partial_sort_copy(__first, __last, __d_first, __d_last, __comp);
}

template<class _ForwardIterator, class _RandomAccessIterator, class _Compare, class _IsVector>
_RandomAccessIterator pattern_partial_sort_copy(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_partial_sort_copy(__first, __last, __d_first, __d_last, __comp, __is_vector);
}

template<class _ForwardIterator, class _RandomAccessIterator, class _Compare, class _IsVector>
_RandomAccessIterator pattern_partial_sort_copy(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_partial_sort_copy(__first, __last, __d_first, __d_last, __comp, __is_vector);
}

//------------------------------------------------------------------------
// count
//------------------------------------------------------------------------
template<class _ForwardIterator, class _Predicate>
typename std::iterator_traits<_ForwardIterator>::difference_type
brick_count(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_count(__first, __last-__first, __pred);
}

template<class _ForwardIterator, class _Predicate>
typename std::iterator_traits<_ForwardIterator>::difference_type
brick_count(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, /* is_vector = */ std::false_type) noexcept {
    return std::count_if(__first, __last, __pred);
}

template<class _ForwardIterator, class _Predicate, class _IsVector>
typename std::iterator_traits<_ForwardIterator>::difference_type
pattern_count(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, /* is_parallel */ std::false_type, _IsVector __is_vector) noexcept {
    return internal::brick_count(__first, __last, __pred, __is_vector);
}

template<class _ForwardIterator, class _Predicate, class _IsVector>
typename std::iterator_traits<_ForwardIterator>::difference_type
pattern_count(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, /* is_parallel */ std::true_type, _IsVector __is_vector) {
    typedef typename std::iterator_traits<_ForwardIterator>::difference_type _size_type;
    return internal::except_handler([=]() {
        return par_backend::parallel_reduce(__first, __last, _size_type(0),
            [__pred, __is_vector](_ForwardIterator __begin, _ForwardIterator __end, _size_type __value) -> _size_type {
                return __value + internal::brick_count(__begin, __end, __pred, __is_vector);
            },
            std::plus<_size_type>()
        );
    });
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------
template<class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator brick_adjacent_find(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, /* IsVector = */ std::true_type, bool __or_semantic) noexcept {
    return unseq_backend::simd_adjacent_find(__first, __last, __pred, __or_semantic);
}

template<class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator brick_adjacent_find(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, /* IsVector = */ std::false_type, bool __or_semantic) noexcept {
    return std::adjacent_find(__first, __last, __pred);
}

template<class _ForwardIterator, class _BinaryPredicate, class _IsVector>
_ForwardIterator pattern_adjacent_find(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred, /* is_parallel */ std::false_type, _IsVector __is_vector, bool __or_semantic) noexcept {
    return internal::brick_adjacent_find(__first, __last, __pred, __is_vector, __or_semantic);
}

template<class _RandomAccessIterator, class _BinaryPredicate, class _IsVector>
_RandomAccessIterator pattern_adjacent_find(_RandomAccessIterator __first, _RandomAccessIterator __last, _BinaryPredicate __pred, /* is_parallel */ std::true_type, _IsVector __is_vector, bool __or_semantic) {
    if (__last - __first < 2)
        return __last;

    return internal::except_handler([=]() {
        return par_backend::parallel_reduce(__first, __last, __last,
            [__last, __pred, __is_vector, __or_semantic](_RandomAccessIterator __begin, _RandomAccessIterator __end,
                                                         _RandomAccessIterator __value) -> _RandomAccessIterator {

            // TODO: investigate performance benefits from the use of shared variable for the result,
            // checking (compare_and_swap idiom) its __value at __first.
            if (__or_semantic && __value < __last) {//found
                par_backend::cancel_execution();
                return __value;
            }

            if (__value > __begin) {
                // modify __end to check the predicate on the boundary __values;
                // TODO: to use a custom range with boundaries overlapping
                // TODO: investigate what if we remove "if" below and run algorithm on range [__first, __last-1)
                // then check the pair [__last-1, __last)
                if (__end != __last)
                    ++__end;

                //correct the global result iterator if the "brick" returns a local "__last"
                const _RandomAccessIterator __res = internal::brick_adjacent_find(__begin, __end, __pred, __is_vector, __or_semantic);
                if (__res < __end)
                    __value = __res;
            }
            return __value;
        },
            [](_RandomAccessIterator __x, _RandomAccessIterator __y) -> _RandomAccessIterator { return __x < __y ? __x : __y; } //reduce a __value
        );
    });
}

//------------------------------------------------------------------------
// nth_element
//------------------------------------------------------------------------

template<class _RandomAccessIterator, class _Compare>
void brick_nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp, /* __is_vector = */ std::false_type) noexcept {
    std::nth_element(__first, __nth, __last, __comp);
}

template<class _RandomAccessIterator, class _Compare>
void brick_nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp, /* __is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    std::nth_element(__first, __nth, __last, __comp);
}

template<class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    internal::brick_nth_element(__first, __nth, __last, __comp, __is_vector);
}

template<class _RandomAccessIterator, class _Compare, class _IsVector>
void pattern_nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    internal::brick_nth_element(__first, __nth, __last, __comp, __is_vector);
}

//------------------------------------------------------------------------
// fill, fill_n
//------------------------------------------------------------------------
template<class _ForwardIterator, class _Tp>
void brick_fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, /* __is_vector = */ std::true_type) noexcept {
    unseq_backend::simd_fill_n(__first, __last - __first, __value);
}

template<class _ForwardIterator, class _Tp>
void brick_fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, /* __is_vector = */std::false_type) noexcept {
    std::fill(__first, __last, __value);
}

template<class _ForwardIterator, class _Tp, class _IsVector>
void pattern_fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, /*is_parallel=*/std::false_type, _IsVector __is_vector) noexcept {
    internal::brick_fill(__first, __last, __value, __is_vector);
}

template<class _ForwardIterator, class T, class _IsVector>
_ForwardIterator pattern_fill(_ForwardIterator __first, _ForwardIterator __last, const T& __value, /*is_parallel=*/std::true_type, _IsVector __is_vector) {
    return except_handler([__first, __last, &__value, __is_vector]() {
        par_backend::parallel_for(__first, __last, [&__value, __is_vector](_ForwardIterator __begin, _ForwardIterator __end) {
            internal::brick_fill(__begin, __end, __value, __is_vector); });
        return __last;
    });
}

template<class _OutputIterator, class _Size, class _Tp>
_OutputIterator brick_fill_n(_OutputIterator __first, _Size __count, const _Tp& __value, /* __is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_fill_n(__first, __count, __value);
}

template<class _OutputIterator, class _Size, class _Tp>
_OutputIterator brick_fill_n(_OutputIterator __first, _Size __count, const _Tp& __value, /* __is_vector = */ std::false_type) noexcept {
    return std::fill_n(__first, __count, __value);
}

template<class _OutputIterator, class _Size, class _Tp, class _IsVector>
_OutputIterator pattern_fill_n(_OutputIterator __first, _Size __count, const _Tp& __value, /*is_parallel=*/std::false_type, _IsVector __is_vector) noexcept {
    return internal::brick_fill_n(__first, __count, __value, __is_vector);
}

template<class _OutputIterator, class _Size, class _Tp, class _IsVector>
_OutputIterator pattern_fill_n(_OutputIterator __first, _Size __count, const _Tp& __value, /*is_parallel=*/std::true_type, _IsVector __is_vector) {
    return internal::pattern_fill(__first, __first + __count, __value, std::true_type(), __is_vector);
}

//------------------------------------------------------------------------
// generate, generate_n
//------------------------------------------------------------------------
template<class _RandomAccessIterator, class _Generator>
void brick_generate(_RandomAccessIterator __first, _RandomAccessIterator __last, _Generator __g, /* is_vector = */ std::true_type) noexcept {
    unseq_backend::simd_generate_n(__first, __last-__first, __g);
}

template<class _ForwardIterator, class _Generator>
void brick_generate(_ForwardIterator __first, _ForwardIterator __last, _Generator __g, /* is_vector = */std::false_type) noexcept {
    std::generate(__first, __last, __g);
}

template<class _ForwardIterator, class _Generator, class _IsVector>
void pattern_generate(_ForwardIterator __first, _ForwardIterator __last, _Generator __g, /*is_parallel=*/std::false_type, _IsVector __is_vector) noexcept {
    internal::brick_generate(__first, __last, __g, __is_vector);
}

template<class _ForwardIterator, class _Generator, class _IsVector>
_ForwardIterator pattern_generate(_ForwardIterator __first, _ForwardIterator __last, _Generator __g, /*is_parallel=*/std::true_type, _IsVector __is_vector) {
    return internal::except_handler([=]() {
        par_backend::parallel_for(__first, __last, [__g, __is_vector](_ForwardIterator __begin, _ForwardIterator __end) {
            internal::brick_generate(__begin, __end, __g, __is_vector); });
        return __last;
    });
}

template<class OutputIterator, class Size, class _Generator>
OutputIterator brick_generate_n(OutputIterator __first, Size __count, _Generator __g, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_generate_n(__first, __count, __g);
}

template<class OutputIterator, class Size, class _Generator>
OutputIterator brick_generate_n(OutputIterator __first, Size __count, _Generator __g, /* is_vector = */ std::false_type) noexcept {
    return std::generate_n(__first, __count, __g);
}

template<class OutputIterator, class Size, class _Generator, class _IsVector>
OutputIterator pattern_generate_n(OutputIterator __first, Size __count, _Generator __g, /*is_parallel=*/std::false_type, _IsVector __is_vector) noexcept {
    return internal::brick_generate_n(__first, __count, __g, __is_vector);
}

template<class OutputIterator, class Size, class _Generator, class _IsVector>
OutputIterator pattern_generate_n(OutputIterator __first, Size __count, _Generator __g, /*is_parallel=*/std::true_type, _IsVector __is_vector) {
  // BUG? - This assumes that an OutputIterator supports operator+()
    return internal::pattern_generate(__first, __first + __count, __g, std::true_type(), __is_vector);
}

//------------------------------------------------------------------------
// remove
//------------------------------------------------------------------------

template<class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator brick_remove_if(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, /* __is_vector = */ std::false_type) noexcept {
    return std::remove_if(__first, __last, __pred);
}

template<class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator brick_remove_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _UnaryPredicate __pred, /* __is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_remove_if(__first, __last - __first, __pred);
}

template<class _ForwardIterator, class _UnaryPredicate, class _IsVector>
_ForwardIterator pattern_remove_if(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallel*/ std::false_type) noexcept {
    return internal::brick_remove_if(__first, __last, __pred, __is_vector);
}

template<class _ForwardIterator, class _UnaryPredicate, class _IsVector>
_ForwardIterator pattern_remove_if(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred, _IsVector __is_vector, /*is_parallel*/ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_remove_if(__first, __last, __pred, __is_vector);
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_merge(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __d_first, _Compare __comp, /* __is_vector = */ std::false_type) noexcept {
    return std::merge(__first1, __last1, __first2, __last2, __d_first, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_merge(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __d_first, _Compare __comp, /* __is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::merge(__first1, __last1, __first2, __last2, __d_first, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_merge(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __d_first, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_merge(__first1, __last1, __first2, __last2, __d_first, __comp, __is_vector);
}

template<class _RandomAccessIterator1, class _RandomAccessIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_merge(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2, _OutputIterator __d_first, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::true_type) noexcept {
    par_backend::parallel_merge(__first1, __last1, __first2, __last2, __d_first, __comp,
        [__is_vector](_RandomAccessIterator1 __f1, _RandomAccessIterator1 __l1, _RandomAccessIterator2 __f2, _RandomAccessIterator2 __l2, _OutputIterator __f3, _Compare __comp) {return brick_merge(__f1, __l1, __f2, __l2, __f3, __comp, __is_vector); });
    return __d_first + (__last1 - __first1) + (__last2 - __first2);
}

//------------------------------------------------------------------------
// inplace_merge
//------------------------------------------------------------------------
template<class _BidirectionalIterator, class _Compare>
void brick_inplace_merge(_BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last, _Compare __comp, /* __is_vector = */ std::false_type) noexcept {
    std::inplace_merge(__first, __middle, __last, __comp);
}

template<class _BidirectionalIterator, class _Compare>
void brick_inplace_merge(_BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last, _Compare __comp, /* __is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial")
        std::inplace_merge(__first, __middle, __last, __comp);
}

template<class _BidirectionalIterator, class _Compare, class _IsVector>
void pattern_inplace_merge(_BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    internal::brick_inplace_merge(__first, __middle, __last, __comp, __is_vector);
}

template<class _BidirectionalIterator, class _Compare, class _IsVector>
void pattern_inplace_merge(_BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    internal::brick_inplace_merge(__first, __middle, __last, __comp, __is_vector);
}

//------------------------------------------------------------------------
// includes
//------------------------------------------------------------------------

template<class ForwardIterator1, class ForwardIterator2, class _Compare>
bool brick_includes(ForwardIterator1 __first1, ForwardIterator1 __last1, ForwardIterator2 __first2, ForwardIterator2 __last2, _Compare __comp, /* _IsVector = */ std::false_type) noexcept {
    return std::includes(__first1, __last1, __first2, __last2, __comp);
}

template<class ForwardIterator1, class ForwardIterator2, class _Compare>
bool brick_includes(ForwardIterator1 __first1, ForwardIterator1 __last1, ForwardIterator2 __first2, ForwardIterator2 __last2, _Compare __comp, /* _IsVector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial")
        return std::includes(__first1, __last1, __first2, __last2, __comp);
}

template<class ForwardIterator1, class ForwardIterator2, class _Compare, class _IsVector>
bool pattern_includes(ForwardIterator1 __first1, ForwardIterator1 __last1, ForwardIterator2 __first2, ForwardIterator2 __last2, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_includes(__first1, __last1, __first2, __last2, __comp, __is_vector);
}

template<class ForwardIterator1, class ForwardIterator2, class _Compare, class _IsVector>
bool pattern_includes(ForwardIterator1 __first1, ForwardIterator1 __last1, ForwardIterator2 __first2, ForwardIterator2 __last2, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_includes(__first1, __last1, __first2, __last2, __comp, __is_vector);
}

//------------------------------------------------------------------------
// set_union
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_union(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::false_type) noexcept {
    return std::set_union(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_union(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
    _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_union(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_union(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_set_union(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_union(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_set_union(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

//------------------------------------------------------------------------
// set_intersection
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_intersection(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::false_type) noexcept {
    return std::set_intersection(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_intersection(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_intersection(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_intersection(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_set_intersection(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_intersection(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_set_intersection(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

//------------------------------------------------------------------------
// set_difference
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::false_type) noexcept {
    return std::set_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_set_difference(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_set_difference(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

//------------------------------------------------------------------------
// set_symmetric_difference
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_symmetric_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,/*__is_vector=*/std::false_type) noexcept {
    return std::set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator brick_set_symmetric_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, /*__is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_symmetric_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::false_type) noexcept {
    return internal::brick_set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare, class _IsVector>
_OutputIterator pattern_set_symmetric_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _IsVector __is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp, __is_vector);
}

//------------------------------------------------------------------------
// is_heap_until
//------------------------------------------------------------------------

template<class _RandomAccessIterator, class _Compare>
_RandomAccessIterator brick_is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
    /* __is_vector = */ std::false_type) noexcept {
    return std::is_heap_until(__first, __last, __comp);
}

template<class _RandomAccessIterator, class _Compare>
_RandomAccessIterator brick_is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
    /* __is_vector = */ std::true_type) noexcept {
    if (__last - __first < 2)
        return __last;
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type _size_type;
    return unseq_backend::simd_first(__first, _size_type(0), __last - __first,
        [&__comp](_RandomAccessIterator __it, _size_type __i) {return __comp(__it[(__i - 1) / 2], __it[__i]); });
}

template<class _RandomAccessIterator, class _Compare, class _IsVector>
_RandomAccessIterator pattern_is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
    _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_is_heap_until(__first, __last, __comp, __is_vector);
}

template<class _RandomAccessIterator, class _DifferenceType, class _Compare>
_RandomAccessIterator is_heap_until_local(_RandomAccessIterator __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp,
    /* __is_vector = */ std::false_type) noexcept {
    _DifferenceType __i = __begin;
    for (; __i < __end; ++__i) {
        if (__comp(__first[(__i - 1) / 2], __first[__i])) {
            break;
        }
    }
    return __first + __i;
}

template<class _RandomAccessIterator, class _DifferenceType, class _Compare>
_RandomAccessIterator is_heap_until_local(_RandomAccessIterator __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp,
    /* __is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_first(__first, __begin, __end,
        [&__comp](_RandomAccessIterator __it, _DifferenceType __i) {return __comp(__it[(__i - 1) / 2], __it[__i]); });
}

template<class _RandomAccessIterator, class _Compare, class _IsVector>
_RandomAccessIterator pattern_is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
    _IsVector __is_vector, /* is_parallel = */ std::true_type) noexcept {
    if (__last - __first < 2)
        return __last;

    return internal::except_handler([=]() {
        return internal::parallel_find(__first, __last, [__first, __last, __comp, __is_vector](_RandomAccessIterator __i, _RandomAccessIterator __j) {
            return internal::is_heap_until_local(__first, __i - __first, __j - __first, __comp, __is_vector);
        },
        std::less<typename std::iterator_traits<_RandomAccessIterator>::difference_type>(), /*is___first=*/true);
    });

}

//------------------------------------------------------------------------
// min_element
//------------------------------------------------------------------------

template <typename _ForwardIterator, typename _Compare>
_ForwardIterator brick_min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, /* __is_vector = */ std::false_type) noexcept {
    return std::min_element(__first, __last, __comp);
}

template <typename _ForwardIterator, typename _Compare>
_ForwardIterator brick_min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, /* __is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::min_element(__first, __last, __comp);
}

template <typename _ForwardIterator, typename _Compare, typename _IsVector>
_ForwardIterator pattern_min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_min_element(__first, __last, __comp, __is_vector);
}

template <typename _RandomAccessIterator, typename _Compare, typename _IsVector>
_RandomAccessIterator pattern_min_element(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::true_type) noexcept {
    if(__first == __last)
        return __last;

    return internal::except_handler([=]() {
        return par_backend::parallel_reduce(
            __first + 1, __last, __first,
            [=](_RandomAccessIterator __begin, _RandomAccessIterator __end, _RandomAccessIterator __init) -> _RandomAccessIterator {
                const _RandomAccessIterator subresult = brick_min_element(__begin, __end, __comp, __is_vector);
                return internal::cmp_iterators_by_values(__init, subresult, __comp);
            },
            [=](_RandomAccessIterator __it1, _RandomAccessIterator __it2) -> _RandomAccessIterator {
                return internal::cmp_iterators_by_values(__it1, __it2, __comp);
            }
        );
    });
}

//------------------------------------------------------------------------
// minmax_element
//------------------------------------------------------------------------

template <typename _ForwardIterator, typename _Compare>
std::pair<_ForwardIterator, _ForwardIterator> brick_minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, /* __is_vector = */ std::false_type) noexcept {
    return std::minmax_element(__first, __last, __comp);
}

template <typename _ForwardIterator, typename _Compare>
std::pair<_ForwardIterator, _ForwardIterator> brick_minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, /* __is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::minmax_element(__first, __last, __comp);
}

template <typename _ForwardIterator, typename _Compare, typename _IsVector>
std::pair<_ForwardIterator, _ForwardIterator> pattern_minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_minmax_element(__first, __last, __comp, __is_vector);
}

template <typename _ForwardIterator, typename _Compare, typename _IsVector>
std::pair<_ForwardIterator, _ForwardIterator> pattern_minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::true_type) noexcept {
    if(__first == __last)
        return std::make_pair(__first, __first);

    return internal::except_handler([=]() {
        typedef std::pair<_ForwardIterator, _ForwardIterator> _result_t;

        return par_backend::parallel_reduce(
            __first + 1, __last, std::make_pair(__first, __first),
            [=](_ForwardIterator __begin, _ForwardIterator __end, _result_t __init) -> _result_t {
                const _result_t __subresult = brick_minmax_element(__begin, __end, __comp, __is_vector);
                return std::make_pair(internal::cmp_iterators_by_values(__subresult.first, __init.first, __comp),
                                      internal::cmp_iterators_by_values(__init.second, __subresult.second,
                                                                        internal::not_pred<_Compare>(__comp)));
            },
            [=](_result_t __p1, _result_t __p2) -> _result_t {
                return std::make_pair(internal::cmp_iterators_by_values(__p1.first, __p2.first, __comp),
                                      internal::cmp_iterators_by_values(__p2.second, __p1.second, internal::not_pred<_Compare>(__comp)));
            }
        );
    });
}

//------------------------------------------------------------------------
// mismatch
//------------------------------------------------------------------------
template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
std::pair<_ForwardIterator1, _ForwardIterator2> mismatch_serial(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __pred) {
#if __PSTL_CPP14_2RANGE_MISMATCH_EQUAL_PRESENT
    return std::mismatch(__first1, __last1, __first2, __last2, __pred);
#else
    for (; __first1 != __last1 && __first2 != __last2 && __pred(*__first1, *__first2); ++__first1,++__first2){ }
    return std::make_pair(__first1, __first2);
#endif
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
std::pair<_ForwardIterator1, _ForwardIterator2> brick_mismatch(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Predicate __pred, /* __is_vector = */ std::false_type) noexcept {
    return internal::mismatch_serial(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
std::pair<_ForwardIterator1, _ForwardIterator2> brick_mismatch(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Predicate __pred, /* __is_vector = */ std::true_type) noexcept {
    auto n = std::min(__last1 - __first1, __last2 - __first2);
    return unseq_backend::simd_first(__first1, n, __first2, not_pred<_Predicate>(__pred));
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Predicate, class _IsVector>
std::pair<_ForwardIterator1, _ForwardIterator2> pattern_mismatch(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Predicate __pred, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_mismatch(__first1, __last1, __first2, __last2, __pred, __is_vector);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Predicate, class _IsVector>
std::pair<_RandomAccessIterator1, _RandomAccessIterator2> pattern_mismatch(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2, _Predicate __pred, _IsVector __is_vector, /* is_parallel = */ std::true_type) noexcept {
    return internal::except_handler([=]() {
        auto n = std::min(__last1 - __first1, __last2 - __first2);
        auto result = internal::parallel_find(__first1, __first1 + n,
                                              [__first1, __first2, __pred, __is_vector](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
           return internal::brick_mismatch(__i, __j, __first2 + (__i - __first1), __first2 + (__j - __first1), __pred, __is_vector).first;
        },
        std::less<typename std::iterator_traits<_RandomAccessIterator1>::difference_type>(), /*is___first=*/true);
        return std::make_pair(result, __first2 + (result - __first1));
    });
}

//------------------------------------------------------------------------
// lexicographical_compare
//------------------------------------------------------------------------

template<class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool brick_lexicographical_compare(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp, /* __is_vector = */ std::false_type) noexcept {
    return std::lexicographical_compare(__first1, __last1, __first2, __last2, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool brick_lexicographical_compare(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp, /* __is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::lexicographical_compare(__first1, __last1, __first2, __last2, __comp);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Compare, class _IsVector>
bool pattern_lexicographical_compare(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::false_type) noexcept {
    return internal::brick_lexicographical_compare(__first1, __last1, __first2, __last2, __comp, __is_vector);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _Compare, class _IsVector>
bool pattern_lexicographical_compare(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp, _IsVector __is_vector, /* is_parallel = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return internal::brick_lexicographical_compare(__first1, __last1, __first2, __last2, __comp, __is_vector);
}

} // namespace internal
} // namespace __pstl

#endif /* __PSTL_algorithm_impl_H */
