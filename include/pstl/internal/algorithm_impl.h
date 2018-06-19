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
#include "memory_impl.h"
#include "unseq_backend_simd.h"
#include "bricks_impl.h"

#if __PSTL_USE_PAR_POLICIES
    #include "parallel_backend.h"
#endif
#include "parallel_impl.h"

namespace pstl {
namespace internal {

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template<class InputIterator, class Pred>
bool brick_any_of( const InputIterator first, const InputIterator last, Pred pred, /*is_vector=*/std::false_type ) noexcept {
    return std::any_of(first,last,pred);
};

template<class InputIterator, class Pred>
bool brick_any_of( const InputIterator first, const InputIterator last, Pred pred, /*is_vector=*/std::true_type ) noexcept {
    return unseq_backend::simd_or( first, last-first, pred );
};


template<class InputIterator, class Pred, class IsVector>
bool pattern_any_of( InputIterator first, InputIterator last, Pred pred, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return brick_any_of(first,last,pred,is_vector);
}

template<class InputIterator, class Pred, class IsVector>
bool pattern_any_of( InputIterator first, InputIterator last, Pred pred, IsVector is_vector, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        return internal::parallel_or( first, last,
            [pred, is_vector](InputIterator i, InputIterator j) {return brick_any_of(i, j, pred, is_vector);} );
    });
}


// [alg.foreach]
// for_each_n with no policy

template<class InputIterator, class Size, class Function>
InputIterator for_each_n_it_serial(InputIterator first, Size n, Function f) {
    for(; n > 0; ++first, --n)
        f(first);
    return first;
}

//------------------------------------------------------------------------
// walk1 (pseudo)
//
// walk1 evaluates f(x) for each dereferenced value x drawn from [first,last)
//------------------------------------------------------------------------
template<class Iterator, class Function>
void brick_walk1( Iterator first, Iterator last, Function f, /*vector=*/std::false_type ) noexcept {
    std::for_each(first, last, f);
}

template<class Iterator, class Function>
void brick_walk1( Iterator first, Iterator last, Function f, /*vector=*/std::true_type ) noexcept {
    unseq_backend::simd_walk_1(first, last-first, f);
}


template<class Iterator, class Function, class IsVector>
void pattern_walk1( Iterator first, Iterator last, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    brick_walk1( first, last, f, is_vector );
}

template<class Iterator, class Function, class IsVector>
void pattern_walk1( Iterator first, Iterator last, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    except_handler([=]() {
        par_backend::parallel_for( first, last, [f,is_vector](Iterator i, Iterator j) {
            brick_walk1(i,j,f,is_vector);
        });
    });
}

template<class Iterator, class Brick>
void pattern_walk_brick( Iterator first, Iterator last, Brick brick, /*parallel=*/std::false_type ) noexcept {
    brick(first, last);
}

template<class Iterator, class Brick>
void pattern_walk_brick( Iterator first, Iterator last, Brick brick, /*parallel=*/std::true_type ) {
    except_handler([=]() {
        par_backend::parallel_for( first, last, [brick](Iterator i, Iterator j) {
            brick(i,j);
        });
    });
}


//------------------------------------------------------------------------
// it_walk1 (pseudo)
//
// it_walk1 evaluates f(it) for each iterator it drawn from [first,last)
//------------------------------------------------------------------------
template<class Iterator, class Function>
void brick_it_walk1( Iterator first, Iterator last, Function f, /*vector=*/std::false_type ) noexcept {
    for(; first!=last; ++first )
        f(first);
}

template<class Iterator, class Function>
void brick_it_walk1( Iterator first, Iterator last, Function f, /*vector=*/std::true_type ) noexcept {
    unseq_backend::simd_it_walk_1(first, last-first, f);
}

template<class Iterator, class Function, class IsVector>
void pattern_it_walk1( Iterator first, Iterator last, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    brick_it_walk1( first, last, f, is_vector );
}

template<class Iterator, class Function, class IsVector>
void pattern_it_walk1( Iterator first, Iterator last, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    except_handler([=]() {
        par_backend::parallel_for( first, last, [f,is_vector](Iterator i, Iterator j) {
            brick_it_walk1(i,j,f,is_vector);
        });
    });
}

//------------------------------------------------------------------------
// walk1_n
//------------------------------------------------------------------------
template<class InputIterator, class Size, class Function>
InputIterator brick_walk1_n(InputIterator first, Size n, Function f, /*IsVectorTag=*/std::false_type ) {
    return for_each_n_it_serial(first, n, [&f](InputIterator it) { f(*it); }); // calling serial version
}

template<class RandomAccessIterator, class DifferenceType, class Function>
RandomAccessIterator brick_walk1_n( RandomAccessIterator first, DifferenceType n, Function f, /*vectorTag=*/std::true_type ) noexcept {
    return unseq_backend::simd_walk_1(first, n, f);
}

template<class InputIterator, class Size, class Function, class IsVector>
InputIterator pattern_walk1_n( InputIterator first, Size n, Function f, IsVector is_vector, /*is_parallel=*/std::false_type ) noexcept {
    return brick_walk1_n(first, n, f, is_vector);
}

template<class RandomAccessIterator, class Size, class Function, class IsVector>
RandomAccessIterator pattern_walk1_n( RandomAccessIterator first, Size n, Function f, IsVector is_vector, /*is_parallel=*/std::true_type ) {
    pattern_walk1(first, first + n, f, is_vector, std::true_type());
    return first + n;
}

template<class InputIterator, class Size, class Brick>
InputIterator pattern_walk_brick_n( InputIterator first, Size n, Brick brick, /*is_parallel=*/std::false_type ) noexcept {
    return brick(first, n);
}

template<class RandomAccessIterator, class Size, class Brick>
RandomAccessIterator pattern_walk_brick_n( RandomAccessIterator first, Size n, Brick brick, /*is_parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(first, first + n, [brick](RandomAccessIterator i, RandomAccessIterator j) {
            brick(i, j-i);
        });
        return first + n;
    });
}



template<class InputIterator, class Size, class Function>
InputIterator brick_it_walk1_n(InputIterator first, Size n, Function f, /*IsVectorTag=*/std::false_type ) {
    return for_each_n_it_serial(first, n, f); // calling serial version
}

template<class RandomAccessIterator, class DifferenceType, class Function>
RandomAccessIterator brick_it_walk1_n( RandomAccessIterator first, DifferenceType n, Function f, /*vectorTag=*/std::true_type ) noexcept {
    return unseq_backend::simd_it_walk_1(first, n, f);
}

template<class InputIterator, class Size, class Function, class IsVector>
InputIterator pattern_it_walk1_n( InputIterator first, Size n, Function f, IsVector is_vector, /*is_parallel=*/std::false_type ) noexcept {
    return brick_it_walk1_n(first, n, f, is_vector);
}

template<class RandomAccessIterator, class Size, class Function, class IsVector>
RandomAccessIterator pattern_it_walk1_n( RandomAccessIterator first, Size n, Function f, IsVector is_vector, /*is_parallel=*/std::true_type ) {
    pattern_it_walk1(first, first + n, f, is_vector, std::true_type());
    return first + n;
}

//------------------------------------------------------------------------
// walk2 (pseudo)
//
// walk2 evaluates f(x,y) for deferenced values (x,y) drawn from [first1,last1) and [first2,...)
//------------------------------------------------------------------------
template<class Iterator1, class Iterator2, class Function>
Iterator2 brick_walk2( Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, /*vector=*/std::false_type ) noexcept {
    for(; first1!=last1; ++first1, ++first2 )
        f(*first1, *first2);
    return first2;
}

template<class Iterator1, class Iterator2, class Function>
Iterator2 brick_walk2( Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_walk_2(first1, last1-first1, first2, f);
}

template<class Iterator1, class Size, class Iterator2, class Function>
Iterator2 brick_walk2_n( Iterator1 first1, Size n, Iterator2 first2, Function f, /*vector=*/std::false_type ) noexcept {
    for(; n > 0; --n, ++first1, ++first2 )
        f(*first1, *first2);
    return first2;
}

template<class Iterator1, class Size, class Iterator2, class Function>
Iterator2 brick_walk2_n(Iterator1 first1, Size n, Iterator2 first2, Function f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_walk_2(first1, n, first2, f);
}



template<class Iterator1, class Iterator2, class Function, class IsVector>
Iterator2 pattern_walk2( Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return brick_walk2(first1,last1,first2,f,is_vector);
}

template<class Iterator1, class Iterator2, class Function, class IsVector>
Iterator2 pattern_walk2(Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            first1, last1,
            [f,first1,first2,is_vector](Iterator1 i, Iterator1 j) {
                brick_walk2(i,j,first2+(i-first1),f,is_vector);
            }
        );
        return first2+(last1-first1);
    });
}

template<class Iterator1, class Size, class Iterator2, class Function, class IsVector>
Iterator2 pattern_walk2_n( Iterator1 first1, Size n, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return brick_walk2_n(first1, n, first2, f, is_vector);
}

template<class Iterator1, class Size, class Iterator2, class Function, class IsVector>
Iterator2 pattern_walk2_n(Iterator1 first1, Size n, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    return pattern_walk2(first1, first1 + n, first2, f, is_vector, std::true_type());
}

template<class Iterator1, class Iterator2, class Brick>
Iterator2 pattern_walk2_brick( Iterator1 first1, Iterator1 last1, Iterator2 first2, Brick brick, /*parallel=*/std::false_type ) noexcept {
    return brick(first1,last1,first2);
}

template<class Iterator1, class Iterator2, class Brick>
Iterator2 pattern_walk2_brick(Iterator1 first1, Iterator1 last1, Iterator2 first2, Brick brick, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            first1, last1,
            [first1,first2, brick](Iterator1 i, Iterator1 j) {
                brick(i,j,first2+(i-first1));
            }
        );
        return first2+(last1-first1);
    });
}

template<class Iterator1, class Size, class Iterator2, class Brick>
Iterator2 pattern_walk2_brick_n(Iterator1 first1, Size n, Iterator2 first2, Brick brick, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            first1, first1+n,
            [first1,first2, brick](Iterator1 i, Iterator1 j) {
                brick(i, j-i, first2+(i-first1));
            }
        );
        return first2 + n;
    });
}

template<class Iterator1, class Size, class Iterator2, class Brick>
Iterator2 pattern_walk2_brick_n( Iterator1 first1, Size n, Iterator2 first2, Brick brick, /*parallel=*/std::false_type ) noexcept {
    return brick(first1, n, first2);
}


//------------------------------------------------------------------------
// it_walk2 (pseudo)
//
// it_walk2 evaluates f(it1, it2) for iterators (it1, it2) drawn from [first1,last1) and [first2,...)
//------------------------------------------------------------------------
template<class Iterator1, class Iterator2, class Function, class IsVector>
Iterator2 pattern_it_walk2( Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return brick_it_walk2(first1,last1,first2,f,is_vector);
}

template<class Iterator1, class Iterator2, class Function, class IsVector>
Iterator2 pattern_it_walk2(Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            first1, last1,
            [f,first1,first2,is_vector](Iterator1 i, Iterator1 j) {
                brick_it_walk2(i,j,first2+(i-first1),f,is_vector);
            }
        );
        return first2+(last1-first1);
    });
}

template<class Iterator1, class Size, class Iterator2, class Function, class IsVector>
Iterator2 pattern_it_walk2_n( Iterator1 first1, Size n, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return brick_it_walk2_n(first1, n, first2, f, is_vector);
}

template<class Iterator1, class Size, class Iterator2, class Function, class IsVector>
Iterator2 pattern_it_walk2_n(Iterator1 first1, Size n, Iterator2 first2, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    return pattern_it_walk2(first1, first1 + n, first2, f, is_vector, std::true_type());
}

//------------------------------------------------------------------------
// walk3 (pseudo)
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------
template<class Iterator1, class Iterator2, class Iterator3, class Function>
Iterator3 brick_walk3( Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 first3, Function f, /*vector=*/std::false_type ) noexcept {
    for(; first1!=last1; ++first1, ++first2, ++first3 )
        f(*first1, *first2, *first3);
    return first3;
}

template<class Iterator1, class Iterator2, class Iterator3, class Function>
Iterator3 brick_walk3( Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 first3, Function f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_walk_3(first1, last1-first1, first2, first3, f);
}


template<class Iterator1, class Iterator2, class Iterator3, class Function, class IsVector>
Iterator3 pattern_walk3( Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 first3, Function f, IsVector is_vector, /*parallel=*/std::false_type ) noexcept {
    return brick_walk3(first1, last1, first2, first3, f, is_vector);
}

template<class Iterator1, class Iterator2, class Iterator3, class Function, class IsVector>
Iterator3 pattern_walk3(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator3 first3, Function f, IsVector is_vector, /*parallel=*/std::true_type ) {
    return except_handler([=]() {
        par_backend::parallel_for(
            first1, last1,
            [f, first1, first2, first3, is_vector](Iterator1 i, Iterator1 j) {
            brick_walk3(i, j, first2+(i-first1), first3+(i-first1), f, is_vector);
        });
        return first3+(last1-first1);
    });
}


//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class BinaryPredicate>
bool brick_equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate p, /* IsVector = */ std::false_type) noexcept {
    return std::equal(first1, last1, first2, p);
}

template<class InputIterator1, class InputIterator2, class BinaryPredicate>
bool brick_equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate p, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_first(first1, last1 - first1, first2, not_pred<BinaryPredicate>(p)).first == last1;
}

template<class InputIterator1, class InputIterator2, class BinaryPredicate, class IsVector>
bool pattern_equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate p, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    return brick_equal(first1, last1, first2, p, is_vector);
}

template<class InputIterator1, class InputIterator2, class BinaryPredicate, class IsVector>
bool pattern_equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate p, IsVector vec, /*is_parallel=*/std::true_type) {
    return except_handler([=]() {
        return !internal::parallel_or(first1, last1,
            [first1, first2, p, vec](InputIterator1 i, InputIterator1 j) {return !brick_equal(i, j,
                first2 + (i - first1), p, vec); });
    });
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------
template<class ForwardIterator, class Predicate>
ForwardIterator brick_find_if(ForwardIterator first, ForwardIterator last, Predicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::find_if(first, last, pred);
}

template<class ForwardIterator, class Predicate>
ForwardIterator brick_find_if(ForwardIterator first, ForwardIterator last, Predicate pred, /*is_vector=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::difference_type size_type;
    return unseq_backend::simd_first(first, size_type(0), last - first,
        [&pred](ForwardIterator it, size_type i) {return pred(it[i]); });
}

template<class ForwardIterator, class Predicate, class IsVector>
ForwardIterator pattern_find_if(ForwardIterator first, ForwardIterator last, Predicate pred, IsVector is_vector, /*is_parallel=*/std::false_type ) noexcept {
    return brick_find_if(first,last,pred,is_vector);
}

template<class ForwardIterator, class Predicate, class IsVector>
ForwardIterator pattern_find_if(ForwardIterator first, ForwardIterator last, Predicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) {
    return except_handler([=]() {
        return internal::parallel_find(first, last, [pred, is_vector](ForwardIterator i, ForwardIterator j) {
            return brick_find_if(i, j, pred, is_vector);
        },
        std::less<typename std::iterator_traits<ForwardIterator>::difference_type>(), /*is_first=*/true);
    });
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

// find the first occurrence of the subsequence [s_first, s_last)
//   or the  last occurrence of the subsequence in the range [first, last)
// b_first determines what occurrence we want to find (first or last)
template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 find_subrange(ForwardIterator1 first, ForwardIterator1 last,
    ForwardIterator1 global_last, ForwardIterator2 s_first, ForwardIterator2 s_last,
    BinaryPredicate pred, bool b_first, IsVector is_vector) noexcept {
    typedef typename std::iterator_traits<ForwardIterator2>::value_type value_type;
    auto  n2 = s_last - s_first;
    if (n2 < 1) {
        return b_first ? first : last;
    }

    auto  n1 = global_last - first;
    if (n1 < n2) {
        return last;
    }

    auto cur = last;
    while (first != last && (global_last - first >= n2)) {
        // find position of *s_first in [first, last) (it can be start of subsequence)
        first = brick_find_if(first, last,
            equal_value_by_pred<value_type, BinaryPredicate>(*s_first, pred), is_vector);

        // if position that was found previously is the start of subsequence
        // then we can exit the loop (b_first == true) or keep the position
        // (b_first == false)
        if (first != last && (global_last - first >= n2) &&
            brick_equal(s_first + 1, s_last, first + 1, pred, is_vector)) {
            if (b_first) {
                return first;
            }
            else {
                cur = first;
            }
        }
        else if (first == last) {
            break;
        }
        else {}

        // in case of b_first == false we try to find new start position
        // for the next subsequence
        ++first;
    }
    return cur;
}

template<class ForwardIterator, class Size, class T, class BinaryPredicate, class IsVector>
ForwardIterator find_subrange(ForwardIterator first, ForwardIterator last,
    ForwardIterator global_last, Size count, const T& value,
    BinaryPredicate pred, IsVector is_vector) noexcept {
    if (global_last - first < count || count < 1) {
        return last; // According to the standard last shall be returned when count < 1
    }

    auto n = global_last - first;
    auto unary_pred = equal_value_by_pred<T, BinaryPredicate>(value, pred);
    while (first != last && (global_last - first >= count)) {
        first = brick_find_if(first, last, unary_pred, is_vector);

        // check that all of elements in [first+1, first+count) equal to value
        if (first != last && (global_last - first >= count) &&
            !brick_any_of(first + 1, first + count, not_pred<decltype(unary_pred)>(unary_pred), is_vector)) {
                return first;
        }
        else if (first == last) {
            break;
        }
        else {
            ++first;
        }
    }
    return last;
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 brick_find_end(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::find_end(first, last, s_first, s_last, pred);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 brick_find_end(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
    return find_subrange(first, last, last, s_first, s_last, pred, false, std::true_type());
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 pattern_find_end(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_find_end(first, last, s_first, s_last, pred, is_vector);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 pattern_find_end(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (last - first == s_last - s_first) {
        const bool res = pattern_equal(first, last, s_first, pred, is_vector, std::true_type());
        return res ? first : last;
    }
    else {
        return except_handler([=]() {
            return internal::parallel_find(first, last, [first, last, s_first, s_last, pred, is_vector](ForwardIterator1 i, ForwardIterator1 j) {
                return find_subrange(i, j, last, s_first, s_last, pred, false, is_vector);
            },
            std::greater<typename std::iterator_traits<ForwardIterator1>::difference_type>(), /*is_first=*/false);
        });
    }
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------
template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 brick_find_first_of(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::find_first_of(first, last, s_first, s_last, pred);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 brick_find_first_of(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
    return unseq_backend::simd_find_first_of(first, last, s_first, s_last, pred);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 pattern_find_first_of(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_find_first_of(first, last, s_first, s_last, pred, is_vector);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 pattern_find_first_of(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    return except_handler([=]() {
        return internal::parallel_find(first, last, [s_first, s_last, pred, is_vector](ForwardIterator1 i, ForwardIterator1 j) {
            return brick_find_first_of(i, j, s_first, s_last, pred, is_vector);
        },
        std::less<typename std::iterator_traits<ForwardIterator1>::difference_type>(), /*is_first=*/true);
    });
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------
template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 brick_search(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, /*vector=*/std::false_type) noexcept {
    return std::search(first, last, s_first, s_last, pred);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 brick_search(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, /*vector=*/std::true_type) noexcept {
    return find_subrange(first, last, last, s_first, s_last, pred, true, std::true_type());
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 pattern_search(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_search(first, last, s_first, s_last, pred, is_vector);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate, class IsVector>
ForwardIterator1 pattern_search(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (last - first == s_last - s_first) {
        const bool res = pattern_equal(first, last, s_first, pred, is_vector, std::true_type());
        return res ? first : last;
    }
    else {
        return except_handler([=]() {
            return internal::parallel_find(first, last, [last, s_first, s_last, pred, is_vector](ForwardIterator1 i, ForwardIterator1 j) {
                return find_subrange(i, j, last, s_first, s_last, pred, true, is_vector);
            },
            std::less<typename std::iterator_traits<ForwardIterator1>::difference_type>(), /*is_first=*/true);
        });
    }
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------
template<class ForwardIterator, class Size, class T, class BinaryPredicate>
ForwardIterator brick_search_n(ForwardIterator first, ForwardIterator last, Size count, const T& value, BinaryPredicate pred, /*vector=*/std::false_type) noexcept {
    return std::search_n(first, last, count, value, pred);
}

template<class ForwardIterator, class Size, class T, class BinaryPredicate>
ForwardIterator brick_search_n(ForwardIterator first, ForwardIterator last, Size count, const T& value, BinaryPredicate pred, /*vector=*/std::true_type) noexcept {
    return find_subrange(first, last, last, count, value, pred, std::true_type());
}

template<class ForwardIterator, class Size, class T, class BinaryPredicate, class IsVector>
ForwardIterator pattern_search_n(ForwardIterator first, ForwardIterator last, Size count, const T& value, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_search_n(first, last, count, value, pred, is_vector);
}

template<class ForwardIterator, class Size, class T, class BinaryPredicate, class IsVector>
ForwardIterator pattern_search_n(ForwardIterator first, ForwardIterator last, Size count, const T& value, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (last - first == count) {
        const bool result = !pattern_any_of(first, last,
            [&value, &pred](const T& val) {return !pred(val, value); },
            is_vector, /*is_parallel*/ std::true_type());
        return result ? first : last;
    }
    else {
        return except_handler([first, last, count, &value, pred, is_vector]() {
            return internal::parallel_find(first, last, [last, count, &value, pred, is_vector](ForwardIterator i, ForwardIterator j) {
                return find_subrange(i, j, last, count, value, pred, is_vector);
            },
            std::less<typename std::iterator_traits<ForwardIterator>::difference_type>(), /*is_first=*/true);
        });
    }
}


//------------------------------------------------------------------------
// copy_n
//------------------------------------------------------------------------

template<class InputIterator, class Size, class OutputIterator>
OutputIterator brick_copy_n(InputIterator first, Size n, OutputIterator result, /*vector=*/std::false_type) noexcept {
    return std::copy_n(first, n, result);
}

template<class InputIterator, class Size, class OutputIterator>
OutputIterator brick_copy_n(InputIterator first, Size n, OutputIterator result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_assign(first, n, result,
        [](InputIterator first, OutputIterator result) {
            *result = *first;
    });
}

//------------------------------------------------------------------------
// copy
//------------------------------------------------------------------------
template<class InputIterator, class OutputIterator>
OutputIterator brick_copy(InputIterator first, InputIterator last, OutputIterator result, /*vector=*/std::false_type) noexcept {
    return std::copy(first, last, result);
}

template<class InputIterator, class OutputIterator>
OutputIterator brick_copy(InputIterator first, InputIterator last, OutputIterator result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_assign(first, last - first, result,
        [](InputIterator first, OutputIterator result) {
            *result = *first;
    });
}

//------------------------------------------------------------------------
// move
//------------------------------------------------------------------------
template<class InputIterator, class OutputIterator>
OutputIterator brick_move(InputIterator first, InputIterator last, OutputIterator result, /*vector=*/std::false_type) noexcept {
    return std::move(first, last, result);
}

template<class InputIterator, class OutputIterator>
OutputIterator brick_move(InputIterator first, InputIterator last, OutputIterator result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_assign(first, last - first, result,
        [](InputIterator first, OutputIterator result) {
            *result = std::move(*first);
        });
}

//------------------------------------------------------------------------
// swap_ranges
//------------------------------------------------------------------------
template<class InputIterator, class OutputIterator>
OutputIterator brick_swap_ranges(InputIterator first, InputIterator last, OutputIterator result, /*vector=*/std::false_type) noexcept {
    return std::swap_ranges(first, last, result);
}

template<class InputIterator, class OutputIterator>
OutputIterator brick_swap_ranges(InputIterator first, InputIterator last, OutputIterator result, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_assign(first, last - first, result, std::iter_swap<InputIterator, OutputIterator>);
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------
template<class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator brick_copy_if(InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred, /*vector=*/std::false_type) noexcept {
    return std::copy_if(first, last, result, pred);
}

template<class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator brick_copy_if(InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    return unseq_backend::simd_copy_if(first, last-first, result, pred);
#else
    return std::copy_if(first, last, result, pred);
#endif
}

// TODO: Try to use transform_reduce for combining brick_copy_if_phase1 on IsVector.
template<class DifferenceType, class InputIterator, class UnaryPredicate>
std::pair<DifferenceType, DifferenceType> brick_calc_mask_1(
    InputIterator first, InputIterator last, bool* __restrict mask, UnaryPredicate pred, /*vector=*/std::false_type) noexcept {
    auto count_true  = DifferenceType(0);
    auto count_false = DifferenceType(0);
    auto size = std::distance(first, last);

    for (; first != last; ++first, ++mask) {
        *mask = pred(*first);
        if (*mask) {
            ++count_true;
        }
    }
    return std::make_pair(count_true, size - count_true);
}

template<class DifferenceType, class InputIterator, class UnaryPredicate>
std::pair<DifferenceType, DifferenceType> brick_calc_mask_1(
    InputIterator first, InputIterator last, bool* __restrict mask, UnaryPredicate pred, /*vector=*/std::true_type) noexcept {
    auto result = unseq_backend::simd_calc_mask_1(first, last - first, mask, pred);
    return std::make_pair(result, (last - first) - result);
}

template<class InputIterator, class OutputIterator>
void brick_copy_by_mask(InputIterator first, InputIterator last, OutputIterator result, bool* mask, /*vector=*/std::false_type) noexcept {
    for (; first != last; ++first, ++mask) {
        if (*mask) {
            *result = *first;
            ++result;
        }
    }
}

template<class InputIterator, class OutputIterator>
void brick_copy_by_mask(InputIterator first, InputIterator last, OutputIterator result, bool* __restrict mask, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    unseq_backend::simd_copy_by_mask(first, last - first, result, mask);
#else
    brick_copy_by_mask(first, last, result, mask, std::false_type());
#endif

}

template<class InputIterator, class OutputIterator1, class OutputIterator2>
void brick_partition_by_mask(InputIterator first, InputIterator last, OutputIterator1 out_true,
    OutputIterator2 out_false, bool* mask, /*vector=*/std::false_type) noexcept {
    for (; first != last; ++first, ++mask) {
        if (*mask) {
            *out_true = *first;
            ++out_true;
        }
        else {
            *out_false = *first;
            ++out_false;
        }
    }
}

template<class InputIterator, class OutputIterator1, class OutputIterator2>
void brick_partition_by_mask(InputIterator first, InputIterator last, OutputIterator1 out_true,
    OutputIterator2 out_false, bool* mask, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    unseq_backend::simd_partition_by_mask(first, last - first, out_true, out_false, mask);
#else
    brick_partition_by_mask(first, last, out_true, out_false, mask, std::false_type());
#endif

}

template<class InputIterator, class OutputIterator, class UnaryPredicate, class IsVector>
OutputIterator pattern_copy_if(InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred, IsVector is_vector, /*parallel=*/std::false_type) noexcept {
    return brick_copy_if(first, last, result, pred, is_vector);
}

template<class InputIterator, class OutputIterator, class UnaryPredicate, class IsVector>
OutputIterator pattern_copy_if(InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred, IsVector is_vector, /*parallel=*/std::true_type) {
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;
    const difference_type n = last-first;
    if( difference_type(1) < n ) {
        par_backend::buffer<bool> mask_buf(n);
        if( mask_buf ) {
            return except_handler([n, first, last, result, is_vector, pred, &mask_buf]() {
                bool* mask = mask_buf.get();
                difference_type m;
                par_backend::parallel_strict_scan( n, difference_type(0),
                    [=](difference_type i, difference_type len) {                               // Reduce
                        return brick_calc_mask_1<difference_type>(first+i, first+(i+len),
                                                                 mask + i,
                                                                 pred,
                                                                 is_vector).first;
                    },
                    std::plus<difference_type>(),                                               // Combine
                    [=](difference_type i, difference_type len, difference_type initial) {      // Scan
                        brick_copy_by_mask(first+i, first+(i+len),
                                                            result+initial,
                                                            mask + i,
                                                            is_vector);
                    },
                    [&m](difference_type total) {m=total;});
                return result + m;
            });
        }
    }
    // Out of memory or trivial sequence - use serial algorithm
    return brick_copy_if(first, last, result, pred, is_vector);
}

//------------------------------------------------------------------------
// unique
//------------------------------------------------------------------------

template<class ForwardIterator, class BinaryPredicate>
ForwardIterator brick_unique(ForwardIterator first, ForwardIterator last, BinaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::unique(first, last, pred);
}

template<class ForwardIterator, class BinaryPredicate>
ForwardIterator brick_unique(ForwardIterator first, ForwardIterator last, BinaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::unique(first, last, pred);
}

template<class ForwardIterator, class BinaryPredicate, class IsVector>
ForwardIterator pattern_unique(ForwardIterator first, ForwardIterator last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_unique(first, last, pred, is_vector);
}

template<class ForwardIterator, class BinaryPredicate, class IsVector>
ForwardIterator pattern_unique(ForwardIterator first, ForwardIterator last, BinaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_unique(first, last, pred, is_vector);
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template<class InputIterator, class OutputIterator, class BinaryPredicate>
OutputIterator brick_unique_copy(InputIterator first, InputIterator last, OutputIterator result, BinaryPredicate pred, /*vector=*/std::false_type) noexcept {
    return std::unique_copy(first, last, result, pred);
}

template<class InputIterator, class OutputIterator, class BinaryPredicate>
OutputIterator brick_unique_copy(InputIterator first, InputIterator last, OutputIterator result, BinaryPredicate pred, /*vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    return unseq_backend::simd_unique_copy(first, last-first, result, pred);
#else
    return std::unique_copy(first, last, result, pred);
#endif
}

template<class InputIterator, class OutputIterator, class BinaryPredicate, class IsVector>
OutputIterator pattern_unique_copy(InputIterator first, InputIterator last, OutputIterator result, BinaryPredicate pred, IsVector is_vector, /*parallel=*/std::false_type) noexcept {
    return brick_unique_copy(first, last, result, pred, is_vector);
}

template<class DifferenceType, class InputIterator, class BinaryPredicate>
DifferenceType brick_calc_mask_2(InputIterator first, InputIterator last, bool* __restrict mask, BinaryPredicate pred, /*vector=*/std::false_type) noexcept {
    DifferenceType count = 0;
    for (; first != last; ++first, ++mask) {
        *mask = !pred(*first, *(first-1));
        count += *mask;
    }
    return count;
}

template<class DifferenceType, class InputIterator, class BinaryPredicate>
DifferenceType brick_calc_mask_2(InputIterator first, InputIterator last, bool* __restrict mask, BinaryPredicate pred, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_calc_mask_2(first, last-first, mask, pred);
}

template<class InputIterator, class OutputIterator, class BinaryPredicate, class IsVector>
OutputIterator pattern_unique_copy(InputIterator first, InputIterator last, OutputIterator result, BinaryPredicate pred, IsVector is_vector, /*parallel=*/std::true_type) {
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;
    const difference_type n = last-first;
    if( difference_type(2)<n ) {
        par_backend::buffer<bool> mask_buf(n);
        if( difference_type(2)<n && mask_buf ) {
            return except_handler([n, first, result, pred, is_vector, &mask_buf]() {
                bool* mask = mask_buf.get();
                difference_type m;
                par_backend::parallel_strict_scan( n, difference_type(0),
                    [=](difference_type i, difference_type len) -> difference_type {          // Reduce
                        difference_type extra = 0;
                        if( i==0 ) {
                            // Special boundary case
                            mask[i] = true;
                            if( --len==0 ) return 1;
                            ++i;
                            ++extra;
                        }
                        return brick_calc_mask_2<difference_type>(
                            first+i, first+(i+len),
                            mask + i,
                            pred,
                            is_vector) + extra;
                    },
                    std::plus<difference_type>(),                                               // Combine
                    [=](difference_type i, difference_type len, difference_type initial) {      // Scan
                        // Phase 2 is same as for pattern_copy_if
                        brick_copy_by_mask(
                            first+i, first+(i+len),
                            result+initial,
                            mask + i,
                            is_vector);
                    },
                    [&m](difference_type total) {m=total;});
                return result + m;
            });
        }
    }
    // Out of memory or trivial sequence - use serial algorithm
    return brick_unique_copy(first, last, result, pred, is_vector);
}

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------

template<class BidirectionalIterator>
void brick_reverse(BidirectionalIterator first, BidirectionalIterator last, /*is_vector=*/std::false_type) noexcept {
    std::reverse(first, last);
}

template<class BidirectionalIterator>
void brick_reverse(BidirectionalIterator first, BidirectionalIterator last, /*is_vector=*/std::true_type) noexcept {
    const auto n = (last - first) - 1;
    unseq_backend::simd_it_walk_1(first, (last - first) / 2, [n, first](BidirectionalIterator in) {
        using std::swap;
        std::iter_swap(in, first + (n - (in - first)));
    });
}

// this brick is called in parallel version, so we can use iterator arithmetic
template<class BidirectionalIterator>
void brick_reverse(BidirectionalIterator first, BidirectionalIterator last, BidirectionalIterator d_first, /*is_vector=*/std::false_type) noexcept {
    using std::swap;
    for (; first != last; --last, ++d_first) {
        std::iter_swap(last - 1, d_first);
    }
}

// this brick is called in parallel version, so we can use iterator arithmetic
template<class BidirectionalIterator>
void brick_reverse(BidirectionalIterator first, BidirectionalIterator last,
BidirectionalIterator d_first, /*is_vector=*/std::true_type) noexcept {
    const auto n = (last - first) - 1;
    unseq_backend::simd_it_walk_1(first, last - first, [n, first, d_first](BidirectionalIterator in) {
        using std::swap;
        std::iter_swap(in, d_first + (n - (in - first)));
    });
}

template<class BidirectionalIterator, class IsVector>
void pattern_reverse(BidirectionalIterator first, BidirectionalIterator last, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    brick_reverse(first, last, is_vector);
}

template<class BidirectionalIterator, class IsVector>
void pattern_reverse(BidirectionalIterator first, BidirectionalIterator last, IsVector is_vector, /*is_parallel=*/std::true_type) {
    par_backend::parallel_for(first, first + (last - first) / 2, [is_vector, first, last](BidirectionalIterator inner_first, BidirectionalIterator inner_last) {
        brick_reverse(inner_first, inner_last, last - (inner_last - first), is_vector);
    });
}

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------

template<class BidirectionalIterator, class ForwardIterator>
ForwardIterator brick_reverse_copy(BidirectionalIterator first, BidirectionalIterator last, ForwardIterator d_first, /*is_vector=*/std::false_type) noexcept {
    return std::reverse_copy(first, last, d_first);
}

template<class BidirectionalIterator, class ForwardIterator>
ForwardIterator brick_reverse_copy(BidirectionalIterator first, BidirectionalIterator last, ForwardIterator d_first, /*is_vector=*/std::true_type) noexcept {
    unseq_backend::simd_it_walk_2(first, last - first, d_first, [first, last](BidirectionalIterator in, ForwardIterator out) {
        *out = *(last - ((in - first) + 1));
    });
    return d_first + (last - first);
}

template<class BidirectionalIterator, class ForwardIterator, class IsVector>
ForwardIterator pattern_reverse_copy(BidirectionalIterator first, BidirectionalIterator last, ForwardIterator d_first, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
  return brick_reverse_copy(first, last, d_first, is_vector);
}

template<class BidirectionalIterator, class ForwardIterator, class IsVector>
ForwardIterator pattern_reverse_copy(BidirectionalIterator first, BidirectionalIterator last, ForwardIterator d_first, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    auto len = last - first;
    par_backend::parallel_for(first, last, [is_vector, first, len, d_first] (BidirectionalIterator inner_first, BidirectionalIterator inner_last) {
        brick_reverse_copy(inner_first, inner_last, d_first + (len - (inner_last - first)), is_vector);
    });
  return d_first + len;
}

//------------------------------------------------------------------------
// rotate
//------------------------------------------------------------------------
template<class ForwardIterator>
ForwardIterator brick_rotate(ForwardIterator first, ForwardIterator middle, ForwardIterator last, /*is_vector=*/std::false_type) noexcept {
#if __PSTL_CPP11_STD_ROTATE_BROKEN
    std::rotate(first, middle, last);
    return std::next(first, std::distance(middle, last));
#else
    return std::rotate(first, middle, last);
#endif
}

template<class ForwardIterator>
ForwardIterator brick_rotate(ForwardIterator first, ForwardIterator middle, ForwardIterator last, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return brick_rotate(first, middle, last, std::false_type());
}

template<class ForwardIterator, class IsVector>
ForwardIterator pattern_rotate(ForwardIterator first, ForwardIterator middle, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_rotate(first, middle, last, is_vector);
}

template<class ForwardIterator, class IsVector>
ForwardIterator pattern_rotate(ForwardIterator first, ForwardIterator middle, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_rotate(first, middle, last, is_vector);
}

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------

template<class ForwardIterator, class OutputIterator>
OutputIterator brick_rotate_copy(ForwardIterator first, ForwardIterator middle, ForwardIterator last, OutputIterator result, /*is_vector=*/std::false_type) noexcept {
    return std::rotate_copy(first, middle, last, result);
}

template<class ForwardIterator, class OutputIterator>
OutputIterator brick_rotate_copy(ForwardIterator first, ForwardIterator middle, ForwardIterator last, OutputIterator result, /*is_vector=*/std::true_type) noexcept {
    OutputIterator temp = brick_copy(middle, last, result, std::true_type());
    return brick_copy(first, middle, temp, std::true_type());
}

template<class ForwardIterator, class OutputIterator, class IsVector>
OutputIterator pattern_rotate_copy(ForwardIterator first, ForwardIterator middle, ForwardIterator last, OutputIterator result, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_rotate_copy(first, middle, last, result, is_vector);
}

template<class ForwardIterator, class OutputIterator, class IsVector>
OutputIterator pattern_rotate_copy(ForwardIterator first, ForwardIterator middle, ForwardIterator last, OutputIterator result, IsVector is_vector, /*is_parallel=*/std::true_type) {
    par_backend::parallel_for(first, last, [first, last, middle, result, is_vector](ForwardIterator b, ForwardIterator e) {
        if (b > middle) {
            brick_copy(b, e, result + (b - middle), is_vector);
        }
        else {
            OutputIterator new_result = result + ((last - middle) + (b - first));
            if (e < middle) {
                brick_copy(b, e, new_result, is_vector);
            }
            else {
                brick_copy(b, middle, new_result, is_vector);
                brick_copy(middle, e, result, is_vector);
            }
        }
    });
    return result + (last - first);
}

//------------------------------------------------------------------------
// is_partitioned
//------------------------------------------------------------------------

template<class ForwardIterator, class UnaryPredicate>
bool brick_is_partitioned(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::is_partitioned(first, last, pred);
}

template<class ForwardIterator, class UnaryPredicate>
bool brick_is_partitioned(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::difference_type size_type;
    if (first == last) {
        return true;
    }
    else {
        ForwardIterator result = unseq_backend::simd_first(first, size_type(0), last - first,
            [&pred](ForwardIterator it, size_type i) {return !pred(it[i]); });
        if (result == last) {
            return true;
        }
        else {
            ++result;
            return !unseq_backend::simd_or(result, last - result, pred);
        }
    }
}

template<class ForwardIterator, class UnaryPredicate, class IsVector>
bool pattern_is_partitioned(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_is_partitioned(first, last, pred, is_vector);
}

template<class ForwardIterator, class UnaryPredicate, class IsVector>
bool pattern_is_partitioned(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (first == last) {
        return true;
    }
    else {
        return except_handler([=, &pred]() {
            typedef typename std::iterator_traits<ForwardIterator>::difference_type size_type;

            // State of current range:
            // broken     - current range is not partitioned by pred
            // all_true   - all elements in current range satisfy pred
            // all_false  - all elements in current range don't satisfy pred
            // true_false - elements satisfy pred are placed before elements that don't satisfy pred
            enum reduce_type { not_init = -1, broken, all_true, all_false, true_false };
            reduce_type init = not_init;

            // Array with states that we'll have when state from the left branch is merged with state from the right branch.
            // State is calculated by formula: new_state = table[left_state * 4 + right_state]
            reduce_type table[] = {
                broken, broken,   broken,     broken,
                broken, all_true, true_false, true_false,
                broken, broken,   all_false,  broken,
                broken, broken,   true_false, broken };

            init = par_backend::parallel_reduce(first, last, init,
                [first, &pred, &table, is_vector](ForwardIterator i, ForwardIterator j, reduce_type value)->reduce_type {
                    if (value == broken) {
                        return broken;
                    }
                    reduce_type res = not_init;
                    // if first element satisfy pred
                    if (pred(*i)) {
                        // find first element that don't satisfy pred
                        ForwardIterator x = brick_find_if(i + 1, j, not_pred<UnaryPredicate>(pred), is_vector);
                        if (x != j) {
                            // find first element after "x" that satisfy pred
                            ForwardIterator y = brick_find_if(x + 1, j, pred, is_vector);
                            // if it was found then range isn't partitioned by pred
                            if (y != j) {
                                return broken;
                            }
                            else {
                                res = true_false;
                            }
                        }
                        else {
                            res = all_true;
                        }
                    }
                    else { // if first element doesn't satisfy pred
                        // then we should find the first element that satisfy pred.
                        // If we found it then range isn't partitioned by pred
                        if (brick_find_if(i + 1, j, pred, is_vector) != j) {
                            return broken;
                        }
                        else {
                            res = all_false;
                        }
                    }
                    // if we have value from left range then we should calculate the result
                    return (value == -1) ? res : table[value * 4 + res];
                },

                [&table](reduce_type val1, reduce_type val2)->reduce_type {
                    if (val1 == broken || val2 == broken) {
                        return broken;
                    }
                    // calculate the result for new big range
                    return table[val1 * 4 + val2];
            });
            return init != broken;
        });
    }
}

//------------------------------------------------------------------------
// partition
//------------------------------------------------------------------------

template<class ForwardIterator, class UnaryPredicate>
ForwardIterator brick_partition(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::partition(first, last, pred);
}

template<class ForwardIterator, class UnaryPredicate>
ForwardIterator brick_partition(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::partition(first, last, pred);
}

template<class ForwardIterator, class UnaryPredicate, class IsVector>
ForwardIterator pattern_partition(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_partition(first, last, pred, is_vector);
}

template<class ForwardIterator, class UnaryPredicate, class IsVector>
ForwardIterator pattern_partition(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {

    // partitioned range: elements before pivot satisfy pred (true part),
    //                    elements after pivot don't satisfy pred (false part)
    struct PartitionRange {
        ForwardIterator begin;
        ForwardIterator pivot;
        ForwardIterator end;
    };

    return except_handler([=]() {
        PartitionRange init{ last, last, last };

        // lambda for merging two partitioned ranges to one partitioned range
        auto reductor = [first, is_vector](PartitionRange val1, PartitionRange val2)->PartitionRange {
            ForwardIterator new_pivot;
            auto size1 = val1.end - val1.pivot;
            auto size2 = val2.pivot - val2.begin;

            // if all elements in left range satisfy pred then we can move new pivot to pivot of right range
            if (val1.end == val1.pivot) {
                new_pivot = val2.pivot;
            }
            // if true part of right range greater than false part of left range
            // then we should swap the false part of left range and last part of true part of right range
            else if (size2 > size1) {
                par_backend::parallel_for(
                    val1.pivot, val1.pivot + size1,
                    [val1, val2, size1, is_vector](ForwardIterator i, ForwardIterator j) {
                        brick_swap_ranges(i, j, (val2.pivot - size1) + (i - val1.pivot), is_vector);
                    }
                );
                new_pivot = val2.pivot - size1;
            }
            // else we should swap the first part of false part of left range and true part of right range
            else {
                par_backend::parallel_for(
                    val1.pivot, val1.pivot + size2,
                    [val1, val2, is_vector](ForwardIterator i, ForwardIterator j) {
                    brick_swap_ranges(i, j, val2.begin + (i - val1.pivot), is_vector);
                }
                );
                new_pivot = val1.pivot + size2;
            }
            return { val2.begin - (val1.end - val1.begin), new_pivot, val2.end };
        };

        PartitionRange result = par_backend::parallel_reduce(first, last, init,
            [first, &pred, is_vector, reductor](ForwardIterator i, ForwardIterator j, PartitionRange value)->PartitionRange {
                //1. serial partition
                ForwardIterator pivot = brick_partition(i, j, pred, is_vector);

                // 2. merging of two ranges (left and right respectively)
                return reductor(value, { i, pivot, j });
            },
            reductor
        );
        return result.pivot;
    });
}

//------------------------------------------------------------------------
// stable_partition
//------------------------------------------------------------------------

template<class BidirectionalIterator, class UnaryPredicate>
BidirectionalIterator brick_stable_partition(BidirectionalIterator first, BidirectionalIterator last, UnaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::stable_partition(first, last, pred);
}

template<class BidirectionalIterator, class UnaryPredicate>
BidirectionalIterator brick_stable_partition(BidirectionalIterator first, BidirectionalIterator last, UnaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::stable_partition(first, last, pred);
}

template<class BidirectionalIterator, class UnaryPredicate, class IsVector>
BidirectionalIterator pattern_stable_partition(BidirectionalIterator first, BidirectionalIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallelization=*/std::false_type) noexcept {
    return brick_stable_partition(first, last, pred, is_vector);
}

template<class BidirectionalIterator, class UnaryPredicate, class IsVector>
BidirectionalIterator pattern_stable_partition(BidirectionalIterator first, BidirectionalIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallelization=*/std::true_type) noexcept {
    // partitioned range: elements before pivot satisfy pred (true part),
    //                    elements after pivot don't satisfy pred (false part)
    struct PartitionRange {
        BidirectionalIterator begin;
        BidirectionalIterator pivot;
        BidirectionalIterator end;
    };
    typedef typename std::iterator_traits<BidirectionalIterator>::value_type T;

    return except_handler([=]() {
        PartitionRange init{ last, last, last };

        // lambda for merging two partitioned ranges to one partitioned range
        auto reductor = [first, is_vector, pred](PartitionRange val1, PartitionRange val2)->PartitionRange {
            BidirectionalIterator new_pivot;
            auto size1 = val1.end - val1.pivot;
            auto size2 = val2.pivot - val2.begin;

            // if all elements in left range satisfy pred then we can move new pivot to pivot of right range
            if (val1.end == val1.pivot) {
                new_pivot = val2.pivot;
            }
            // if true part of right range greater than false part of left range
            // then we should swap the false part of left range and last part of true part of right range
            else {
                brick_rotate(val1.pivot, val2.begin, val2.pivot, is_vector);
                new_pivot = val2.pivot - size1;
            }
            return { val2.begin - (val1.end - val1.begin), new_pivot, val2.end };
        };

        PartitionRange result = par_backend::parallel_reduce(first, last, init,
            [first, &pred, is_vector, reductor](BidirectionalIterator i, BidirectionalIterator j, PartitionRange value)->PartitionRange {
                //1. serial stable_partition
                BidirectionalIterator pivot = brick_stable_partition(i, j, pred, is_vector);

                // 2. merging of two ranges (left and right respectively)
                return reductor(value, { i, pivot, j });
            },
            reductor);
        return result.pivot;
    });
}

//------------------------------------------------------------------------
// partition_copy
//------------------------------------------------------------------------

template<class InputIterator, class OutputIterator1, class OutputIterator2, class UnaryPredicate>
std::pair<OutputIterator1, OutputIterator2>
brick_partition_copy(InputIterator first, InputIterator last, OutputIterator1 out_true, OutputIterator2 out_false, UnaryPredicate pred, /*is_vector=*/std::false_type) noexcept {
    return std::partition_copy(first, last, out_true, out_false, pred);
}

template<class InputIterator, class OutputIterator1, class OutputIterator2, class UnaryPredicate>
std::pair<OutputIterator1, OutputIterator2>
brick_partition_copy(InputIterator first, InputIterator last, OutputIterator1 out_true, OutputIterator2 out_false, UnaryPredicate pred, /*is_vector=*/std::true_type) noexcept {
#if (__PSTL_MONOTONIC_PRESENT)
    return unseq_backend::simd_partition_copy(first, last - first, out_true, out_false, pred);
#else
    return std::partition_copy(first, last, out_true, out_false, pred);
#endif
}


template<class InputIterator, class OutputIterator1, class OutputIterator2, class UnaryPredicate, class IsVector>
std::pair<OutputIterator1, OutputIterator2>
pattern_partition_copy(InputIterator first, InputIterator last, OutputIterator1 out_true, OutputIterator2 out_false, UnaryPredicate pred, IsVector is_vector,/*is_parallelization=*/std::false_type) noexcept {
    return brick_partition_copy(first, last, out_true, out_false, pred, is_vector);
}

template<class InputIterator, class OutputIterator1, class OutputIterator2, class UnaryPredicate, class IsVector>
std::pair<OutputIterator1, OutputIterator2>
pattern_partition_copy(InputIterator first, InputIterator last, OutputIterator1 out_true, OutputIterator2 out_false, UnaryPredicate pred, IsVector is_vector, /*is_parallelization=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<InputIterator>::difference_type difference_type;
    typedef std::pair<difference_type, difference_type> return_type;
    const difference_type n = last - first;
    if (difference_type(1) < n) {
        par_backend::buffer<bool> mask_buf(n);
        if (mask_buf) {
            return except_handler([n, first, last, out_true, out_false, is_vector, pred, &mask_buf]() {
                bool* mask = mask_buf.get();
                return_type m;
                par_backend::parallel_strict_scan(n, std::make_pair(difference_type(0), difference_type(0)),
                    [=](difference_type i, difference_type len) {                             // Reduce
                    return brick_calc_mask_1<difference_type>(first + i, first + (i + len),
                        mask + i,
                        pred,
                        is_vector);
                    },
                    [](const return_type& x, const return_type& y)-> return_type {
                        return std::make_pair(x.first + y.first, x.second + y.second);
                    },                                                                        // Combine
                    [=](difference_type i, difference_type len, return_type initial) {        // Scan
                        brick_partition_by_mask(first + i, first + (i + len),
                            out_true + initial.first,
                            out_false + initial.second,
                            mask + i,
                            is_vector);
                    },
                    [&m](const return_type& total) {m = total;}
                );
                return std::make_pair(out_true + m.first, out_false + m.second);
            });
        }
    }
    // Out of memory or trivial sequence - use serial algorithm
    return brick_partition_copy(first, last, out_true, out_false, pred, is_vector);
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------

template<class RandomAccessIterator, class Compare, class IsVector, class IsMoveConstructible>
void pattern_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp, IsVector /*is_vector*/, /*is_parallel=*/std::false_type, IsMoveConstructible) noexcept {
    std::sort(first, last, comp);
}


template<class RandomAccessIterator, class Compare, class IsVector>
void pattern_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp, IsVector /*is_vector*/, /*is_parallel=*/std::true_type, /*is_move_constructible=*/std::true_type ) {
    except_handler([=]() {
        par_backend::parallel_stable_sort(first, last, comp,
            [](RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
            std::sort(first, last, comp);
        },
        last - first);
    });
}

//------------------------------------------------------------------------
// stable_sort
//------------------------------------------------------------------------

template<class RandomAccessIterator, class Compare, class IsVector>
void pattern_stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp, IsVector /*is_vector*/, /*is_parallel=*/std::false_type) noexcept {
    std::stable_sort(first, last, comp);
}

template<class RandomAccessIterator, class Compare, class IsVector>
void pattern_stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp, IsVector /*is_vector*/, /*is_parallel=*/std::true_type) {
    except_handler([=]() {
        par_backend::parallel_stable_sort(first, last, comp,
            [](RandomAccessIterator first, RandomAccessIterator last, Compare comp) {
            std::stable_sort(first, last, comp);
        });
    });
}

//------------------------------------------------------------------------
// partial_sort
//------------------------------------------------------------------------

template<class RandomAccessIterator, class Compare, class IsVector>
void pattern_partial_sort(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp, IsVector, /*is_parallel=*/std::false_type) noexcept {
    std::partial_sort(first, middle, last, comp);
}

template <class RandomAccessIterator, class Compare, class IsVector>
void pattern_partial_sort(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp, IsVector, /*is_parallel=*/std::true_type) noexcept {
    const auto n = middle - first;
    except_handler([=]() {
        par_backend::parallel_stable_sort(first, last, comp,
            [n](RandomAccessIterator begin, RandomAccessIterator end, Compare comp) {
            if(n < end - begin)
                std::partial_sort(begin, begin + n, end, comp);
            else
                std::sort(begin, end, comp);
        },
        n);
    });
}

//------------------------------------------------------------------------
// partial_sort_copy
//------------------------------------------------------------------------

template<class ForwardIterator, class RandomAccessIterator, class Compare, class IsVector>
RandomAccessIterator pattern_partial_sort_copy(ForwardIterator first, ForwardIterator last, RandomAccessIterator d_first, RandomAccessIterator d_last, Compare comp, IsVector, /*is_parallel=*/std::false_type) noexcept {
    return std::partial_sort_copy(first, last, d_first, d_last, comp);
}

template<class ForwardIterator, class RandomAccessIterator, class Compare, class IsVector>
RandomAccessIterator pattern_partial_sort_copy(ForwardIterator first, ForwardIterator last, RandomAccessIterator d_first, RandomAccessIterator d_last, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (last == first || d_last == d_first) {
        return d_first;
    }
    auto n1 =   last -   first;
    auto n2 = d_last - d_first;
    return except_handler([=]() {
        if (n2 >= n1) {
            par_backend::parallel_stable_sort(d_first, d_first + n1, comp,
                [n1, first, d_first, is_vector](RandomAccessIterator i, RandomAccessIterator j, Compare comp) {
                ForwardIterator i1 = first + (i - d_first);
                ForwardIterator j1 = first + (j - d_first);

                // 1. Copy elements from input to output
#if !__PSTL_ICC_18_OMP_SIMD_BROKEN
                brick_copy(i1, j1, i, is_vector);
#else
                std::copy(i1, j1, i);
#endif
                // 2. Sort elements in output sequence
                std::sort(i, j, comp);
            },
            n1);
            return d_first + n1;
        }
        else {
            typedef typename std::iterator_traits<ForwardIterator>::value_type T1;
            typedef typename std::iterator_traits<RandomAccessIterator>::value_type T2;
            par_backend::buffer<T1> buf(n1);
            if (buf) {
                T1* r = buf.get();

                par_backend::parallel_stable_sort(r, r+n1, comp,
                    [n2, first, r](T1* i, T1* j, Compare comp) {
                    ForwardIterator it = first + (i - r);

                    // 1. Copy elements from input to raw memory
                    for (T1* k=i; k != j; ++k, ++it) {
                        ::new (k) T2(*it);
                    }

                    // 2. Sort elements in temporary buffer
                    if (n2 < j - i)
                        std::partial_sort(i, i + n2, j, comp);
                    else
                        std::sort(i, j, comp);
                },
                n2);

                // 3. Move elements from temporary buffer to output
                par_backend::parallel_for(r, r + n2,
                    [r, d_first, is_vector](T1* i, T1* j) {
                    brick_move(i, j, d_first + (i - r), is_vector);
                });
                return d_first + n2;
            }
            return std::partial_sort_copy(first, last, d_first, d_last, comp);
        }
    });
}

//------------------------------------------------------------------------
// count
//------------------------------------------------------------------------
template<class ForwardIterator, class Predicate>
typename std::iterator_traits<ForwardIterator>::difference_type
brick_count(ForwardIterator first, ForwardIterator last, Predicate pred, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_count(first, last-first, pred);
}

template<class ForwardIterator, class Predicate>
typename std::iterator_traits<ForwardIterator>::difference_type
brick_count(ForwardIterator first, ForwardIterator last, Predicate pred, /* is_vector = */ std::false_type) noexcept {
    return std::count_if(first, last, pred);
}

template<class ForwardIterator, class Predicate, class IsVector>
typename std::iterator_traits<ForwardIterator>::difference_type
pattern_count(ForwardIterator first, ForwardIterator last, Predicate pred, /* is_parallel */ std::false_type, IsVector vec) noexcept {
    return brick_count(first, last, pred, vec);
}

template<class ForwardIterator, class Predicate, class IsVector>
typename std::iterator_traits<ForwardIterator>::difference_type
pattern_count(ForwardIterator first, ForwardIterator last, Predicate pred, /* is_parallel */ std::true_type, IsVector vec) {
    typedef typename std::iterator_traits<ForwardIterator>::difference_type size_type;
    return except_handler([=]() {
        return par_backend::parallel_reduce(first, last, size_type(0),
            [pred, vec](ForwardIterator begin, ForwardIterator end, size_type value)->size_type {
            return value + brick_count(begin, end, pred, vec);
            },
            std::plus<size_type>()
        );
    });
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------
template<class ForwardIt, class BinaryPredicate>
ForwardIt brick_adjacent_find(ForwardIt first, ForwardIt last, BinaryPredicate pred, /* IsVector = */ std::true_type, bool or_semantic) noexcept {
    return unseq_backend::simd_adjacent_find(first, last, pred, or_semantic);
}

template<class ForwardIt, class BinaryPredicate>
ForwardIt brick_adjacent_find(ForwardIt first, ForwardIt last, BinaryPredicate pred, /* IsVector = */ std::false_type, bool or_semantic) noexcept {
    return std::adjacent_find(first, last, pred);
}

template<class ForwardIt, class BinaryPredicate, class IsVector>
ForwardIt pattern_adjacent_find(ForwardIt first, ForwardIt last, BinaryPredicate pred, /* is_parallel */ std::false_type, IsVector vec, bool or_semantic) noexcept {
    return brick_adjacent_find(first, last, pred, vec, or_semantic);
}

template<class ForwardIt, class BinaryPredicate, class IsVector>
ForwardIt pattern_adjacent_find(ForwardIt first, ForwardIt last, BinaryPredicate pred, /* is_parallel */ std::true_type, IsVector vec, bool or_semantic) {
    if (last - first < 2)
        return last;

    return except_handler([=]() {
        return par_backend::parallel_reduce(first, last, last,
            [last, pred, vec, or_semantic](ForwardIt begin, ForwardIt end, ForwardIt value)->ForwardIt {

            // TODO: investigate performance benefits from the use of shared variable for the result,
            // checking (compare_and_swap idiom) its value at first.
            if (or_semantic && value < last) {//found
                return value;
            }

            if (value > begin) {
                // modify end to check the predicate on the boundary values;
                // TODO: to use a custom range with boundaries overlapping
                // TODO: investigate what if we remove "if" below and run algorithm on range [first, last-1)
                // then check the pair [last-1, last)
                if (end != last)
                    ++end;

                //correct the global result iterator if the "brick" returns a local "last"
                const ForwardIt res = brick_adjacent_find(begin, end, pred, vec, or_semantic);
                if (res < end)
                    value = res;
            }
            return value;
        },
            [](ForwardIt x, ForwardIt y)->ForwardIt { return x < y ? x : y; } //reduce a value
        );
    });
}

//------------------------------------------------------------------------
// nth_element
//------------------------------------------------------------------------

template<class RandomAccessIterator, class Compare>
void brick_nth_element(RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last, Compare comp, /* is_vector = */ std::false_type) noexcept {
    std::nth_element(first, nth, last, comp);
}

template<class RandomAccessIterator, class Compare>
void brick_nth_element(RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last, Compare comp, /* is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    std::nth_element(first, nth, last, comp);
}

template<class RandomAccessIterator, class Compare, class IsVector>
void pattern_nth_element(RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last, Compare comp, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    brick_nth_element(first, nth, last, comp, is_vector);
}

template<class RandomAccessIterator, class Compare, class IsVector>
void pattern_nth_element(RandomAccessIterator first, RandomAccessIterator nth, RandomAccessIterator last, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    brick_nth_element(first, nth, last, comp, is_vector);
}

//------------------------------------------------------------------------
// fill, fill_n
//------------------------------------------------------------------------
template<class ForwardIterator, class T>
void brick_fill(ForwardIterator first, ForwardIterator last, const T& value, /* is_vector = */ std::true_type) noexcept {
    unseq_backend::simd_fill_n(first, last-first, value);
}

template<class ForwardIterator, class T>
void brick_fill(ForwardIterator first, ForwardIterator last, const T& value, /* is_vector = */std::false_type) noexcept {
    std::fill(first, last, value);
}

template<class ForwardIterator, class T, class IsVector>
void pattern_fill(ForwardIterator first, ForwardIterator last, const T& value, /*is_parallel=*/std::false_type, IsVector vec) noexcept {
    brick_fill(first, last, value, vec);
}

template<class ForwardIterator, class T, class IsVector>
ForwardIterator pattern_fill(ForwardIterator first, ForwardIterator last, const T& value, /*is_parallel=*/std::true_type, IsVector vec) {
    return except_handler([first, last, &value, vec]() {
        par_backend::parallel_for(first, last, [&value, vec](ForwardIterator begin, ForwardIterator end) {
            brick_fill(begin, end, value, vec); });
        return last;
    });
}

template<class OutputIterator, class Size, class T>
OutputIterator brick_fill_n(OutputIterator first, Size count, const T& value, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_fill_n(first, count, value);
}

template<class OutputIterator, class Size, class T>
OutputIterator brick_fill_n(OutputIterator first, Size count, const T& value, /* is_vector = */ std::false_type) noexcept {
    return std::fill_n(first, count, value);
}

template<class OutputIterator, class Size, class T, class IsVector>
OutputIterator pattern_fill_n(OutputIterator first, Size count, const T& value, /*is_parallel=*/std::false_type, IsVector vec) noexcept {
    return brick_fill_n(first, count, value, vec);
}

template<class OutputIterator, class Size, class T, class IsVector>
OutputIterator pattern_fill_n(OutputIterator first, Size count, const T& value, /*is_parallel=*/std::true_type, IsVector vec) {
    return pattern_fill(first, first + count, value, std::true_type(), vec);
}

//------------------------------------------------------------------------
// generate, generate_n
//------------------------------------------------------------------------
template<class ForwardIterator, class Generator>
void brick_generate(ForwardIterator first, ForwardIterator last, Generator g, /* is_vector = */ std::true_type) noexcept {
    unseq_backend::simd_generate_n(first, last-first, g);
}

template<class ForwardIterator, class Generator>
void brick_generate(ForwardIterator first, ForwardIterator last, Generator g, /* is_vector = */std::false_type) noexcept {
    std::generate(first, last, g);
}

template<class ForwardIterator, class Generator, class IsVector>
void pattern_generate(ForwardIterator first, ForwardIterator last, Generator g, /*is_parallel=*/std::false_type, IsVector vec) noexcept {
    brick_generate(first, last, g, vec);
}

template<class ForwardIterator, class Generator, class IsVector>
ForwardIterator pattern_generate(ForwardIterator first, ForwardIterator last, Generator g, /*is_parallel=*/std::true_type, IsVector vec) {
    return except_handler([=]() {
        par_backend::parallel_for(first, last, [g, vec](ForwardIterator begin, ForwardIterator end) {
            brick_generate(begin, end, g, vec); });
        return last;
    });
}

template<class OutputIterator, class Size, class Generator>
OutputIterator brick_generate_n(OutputIterator first, Size count, Generator g, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_generate_n(first, count, g);
}

template<class OutputIterator, class Size, class Generator>
OutputIterator brick_generate_n(OutputIterator first, Size count, Generator g, /* is_vector = */ std::false_type) noexcept {
    return std::generate_n(first, count, g);
}

template<class OutputIterator, class Size, class Generator, class IsVector>
OutputIterator pattern_generate_n(OutputIterator first, Size count, Generator g, /*is_parallel=*/std::false_type, IsVector vec) noexcept {
    return brick_generate_n(first, count, g, vec);
}

template<class OutputIterator, class Size, class Generator, class IsVector>
OutputIterator pattern_generate_n(OutputIterator first, Size count, Generator g, /*is_parallel=*/std::true_type, IsVector vec) {
    return pattern_generate(first, first + count, g, std::true_type(), vec);
}

//------------------------------------------------------------------------
// remove
//------------------------------------------------------------------------

template<class ForwardIterator, class UnaryPredicate>
ForwardIterator brick_remove_if(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, /* is_vector = */ std::false_type) noexcept {
    return std::remove_if(first, last, pred);
}

template<class ForwardIterator, class UnaryPredicate>
ForwardIterator brick_remove_if(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_remove_if(first, last - first, pred);
}

template<class ForwardIterator, class UnaryPredicate, class IsVector>
ForwardIterator pattern_remove_if(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallel*/ std::false_type) noexcept {
    return brick_remove_if(first, last, pred, is_vector);
}

template<class ForwardIterator, class UnaryPredicate, class IsVector>
ForwardIterator pattern_remove_if(ForwardIterator first, ForwardIterator last, UnaryPredicate pred, IsVector is_vector, /*is_parallel*/ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_remove_if(first, last, pred, is_vector);
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_merge(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator d_first, Compare comp, /* is_vector = */ std::false_type) noexcept {
    return std::merge(first1, last1, first2, last2, d_first, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_merge(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator d_first, Compare comp, /* is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::merge(first1, last1, first2, last2, d_first, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_merge(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator d_first, Compare comp, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    return brick_merge(first1, last1, first2, last2, d_first, comp, is_vector);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_merge(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator d_first, Compare comp, IsVector is_vector, /* is_parallel = */ std::true_type) noexcept {
    except_handler([=]() {
        par_backend::parallel_merge(first1, last1, first2, last2, d_first, comp,
            [is_vector](InputIterator1 f1, InputIterator1 l1, InputIterator2 f2, InputIterator2 l2, OutputIterator f3, Compare comp) {return brick_merge(f1, l1, f2, l2, f3, comp, is_vector); });
    });
    return d_first + (last1 - first1) + (last2 - first2);
}

//------------------------------------------------------------------------
// inplace_merge
//------------------------------------------------------------------------

template<class BidirectionalIterator, class Compare>
void brick_inplace_merge(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp, /* is_vector = */ std::false_type) noexcept {
    std::inplace_merge(first, middle, last, comp);
}

template<class BidirectionalIterator, class Compare>
void brick_inplace_merge(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp, /* is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial")
    std::inplace_merge(first, middle, last, comp);
}

template<class BidirectionalIterator, class Compare, class IsVector>
void pattern_inplace_merge(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    brick_inplace_merge(first, middle, last, comp, is_vector);
}

template<class BidirectionalIterator, class Compare, class IsVector>
void pattern_inplace_merge(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    if (first == last || first == middle || middle == last) {
        return;
    }
    typedef typename std::iterator_traits<BidirectionalIterator>::value_type T;
    auto n = last - first;
    par_backend::buffer<T> buf(n);
    if (buf) {
        T* r = buf.get();
        except_handler([=]() {
            auto move_values = [](BidirectionalIterator x, T* z) {invoke_if_else(std::is_trivial<T>(),
                [&]() {*z = std::move(*x); },
                [&]() {::new (internal::reduce_to_ptr(z)) T(std::move(*x)); }
            ); };

            auto move_sequences = [](BidirectionalIterator first1, BidirectionalIterator last1, T* first2) {
                return brick_uninitialized_move(first1, last1, first2, IsVector()); };

            par_backend::parallel_merge(first, middle, middle, last, r, comp,
                [n, comp, move_values, move_sequences](BidirectionalIterator f1, BidirectionalIterator l1, BidirectionalIterator f2, BidirectionalIterator l2, T* f3,
                    Compare comp) {
                auto func = par_backend::serial_move_merge<decltype(move_values), decltype(move_sequences)>(n, move_values, move_sequences);
                func(f1, l1, f2, l2, f3, comp);
                return f3 + (l1 - f1) + (l2 - f2);
            });
            par_backend::parallel_for(r, r + n,
                [r, first, is_vector](T* i, T* j) {
                brick_move(i, j, first + (i - r), is_vector);
            });
        });
    }
    else {
        std::inplace_merge(first, middle, last, comp);
    }
}

//------------------------------------------------------------------------
// includes
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class Compare>
bool brick_includes(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Compare comp, /* IsVector = */ std::false_type) noexcept {
    return std::includes(first1, last1, first2, last2, comp);
}

template<class InputIterator1, class InputIterator2, class Compare>
bool brick_includes(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Compare comp, /* IsVector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial")
        return std::includes(first1, last1, first2, last2, comp);
}

template<class InputIterator1, class InputIterator2, class Compare, class IsVector>
bool pattern_includes(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Compare comp, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_includes(first1, last1, first2, last2, comp, is_vector);
}

template<class InputIterator1, class InputIterator2, class Compare, class IsVector>
bool pattern_includes(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_includes(first1, last1, first2, last2, comp, is_vector);
}

//------------------------------------------------------------------------
// set_union
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_union(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::false_type) noexcept {
    return std::set_union(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_union(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
    InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_union(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_union(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_set_union(first1, last1, first2, last2, result, comp, is_vector);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_union(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_set_union(first1, last1, first2, last2, result, comp, is_vector);
}

//------------------------------------------------------------------------
// set_intersection
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_intersection(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::false_type) noexcept {
    return std::set_intersection(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_intersection(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_intersection(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_intersection(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_set_intersection(first1, last1, first2, last2, result, comp, is_vector);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_intersection(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_set_intersection(first1, last1, first2, last2, result, comp, is_vector);
}

//------------------------------------------------------------------------
// set_difference
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::false_type) noexcept {
    return std::set_difference(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_difference(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_set_difference(first1, last1, first2, last2, result, comp, is_vector);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_set_difference(first1, last1, first2, last2, result, comp, is_vector);
}

//------------------------------------------------------------------------
// set_symmetric_difference
//------------------------------------------------------------------------

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_symmetric_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp,/*is_vector=*/std::false_type) noexcept {
    return std::set_symmetric_difference(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare>
OutputIterator brick_set_symmetric_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_symmetric_difference(first1, last1, first2, last2, result, comp);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_symmetric_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_set_symmetric_difference(first1, last1, first2, last2, result, comp, is_vector);
}

template<class InputIterator1, class InputIterator2, class OutputIterator, class Compare, class IsVector>
OutputIterator pattern_set_symmetric_difference(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, OutputIterator result, Compare comp, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_set_symmetric_difference(first1, last1, first2, last2, result, comp, is_vector);
}

//------------------------------------------------------------------------
// is_heap_until
//------------------------------------------------------------------------

template<class RandomAccessIterator, class Compare>
RandomAccessIterator brick_is_heap_until(RandomAccessIterator first, RandomAccessIterator last, Compare comp,
    /* is_vector = */ std::false_type) noexcept {
    return std::is_heap_until(first, last, comp);
}

template<class RandomAccessIterator, class Compare>
RandomAccessIterator brick_is_heap_until(RandomAccessIterator first, RandomAccessIterator last, Compare comp,
    /* is_vector = */ std::true_type) noexcept {
    if (last - first < 2)
        return last;
    typedef typename std::iterator_traits<RandomAccessIterator>::difference_type size_type;
    return unseq_backend::simd_first(first, size_type(0), last - first,
        [&comp](RandomAccessIterator it, size_type i) {return comp(it[(i - 1) / 2], it[i]); });
}

template<class RandomAccessIterator, class Compare, class IsVector>
RandomAccessIterator pattern_is_heap_until(RandomAccessIterator first, RandomAccessIterator last, Compare comp,
    IsVector vec, /* is_parallel = */ std::false_type) noexcept {
    return brick_is_heap_until(first, last, comp, vec);
}

template<class RandomAccessIterator, class DifferenceType, class Compare>
RandomAccessIterator is_heap_until_local(RandomAccessIterator first, DifferenceType begin, DifferenceType end, Compare comp,
    /* is_vector = */ std::false_type) noexcept {
    DifferenceType i = begin;
    for (; i < end; ++i) {
        if (comp(first[(i - 1) / 2], first[i])) {
            break;
        }
    }
    return first + i;
}

template<class RandomAccessIterator, class DifferenceType, class Compare>
RandomAccessIterator is_heap_until_local(RandomAccessIterator first, DifferenceType begin, DifferenceType end, Compare comp,
    /* is_vector = */ std::true_type) noexcept {
    return unseq_backend::simd_first(first, begin, end,
        [&comp](RandomAccessIterator it, DifferenceType i) {return comp(it[(i - 1) / 2], it[i]); });
}

template<class RandomAccessIterator, class Compare, class IsVector>
RandomAccessIterator pattern_is_heap_until(RandomAccessIterator first, RandomAccessIterator last, Compare comp,
    IsVector vec, /* is_parallel = */ std::true_type) noexcept {
    if (last - first < 2)
        return last;

    return except_handler([=]() {
        return internal::parallel_find(first, last, [first, last, comp, vec](RandomAccessIterator i, RandomAccessIterator j) {
            return is_heap_until_local(first, i - first, j - first, comp, vec);
        },
        std::less<typename std::iterator_traits<RandomAccessIterator>::difference_type>(), /*is_first=*/true);

    });

}

//------------------------------------------------------------------------
// min_element
//------------------------------------------------------------------------

template <typename ForwardIterator, typename Compare>
ForwardIterator brick_min_element(ForwardIterator first, ForwardIterator last, Compare comp, /* is_vector = */ std::false_type) noexcept {
    return std::min_element(first, last, comp);
}

template <typename ForwardIterator, typename Compare>
ForwardIterator brick_min_element(ForwardIterator first, ForwardIterator last, Compare comp, /* is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::min_element(first, last, comp);
}

template <typename ForwardIterator, typename Compare, typename IsVector>
ForwardIterator pattern_min_element(ForwardIterator first, ForwardIterator last, Compare comp, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    return brick_min_element(first, last, comp, is_vector);
}

template <typename ForwardIterator, typename Compare, typename IsVector>
ForwardIterator pattern_min_element(ForwardIterator first, ForwardIterator last, Compare comp, IsVector is_vector, /* is_parallel = */ std::true_type) noexcept {
    if(first == last)
        return last;

    return except_handler([=]() {
        return par_backend::parallel_reduce(
            first + 1, last, first,
            [=](ForwardIterator begin, ForwardIterator end, ForwardIterator init) -> ForwardIterator {
                const ForwardIterator subresult = brick_min_element(begin, end, comp, is_vector);
                return cmp_iterators_by_values(init, subresult, comp);
            },
            [=](ForwardIterator it1, ForwardIterator it2) -> ForwardIterator {
                return cmp_iterators_by_values(it1, it2, comp);
            }
        );
    });
}

//------------------------------------------------------------------------
// minmax_element
//------------------------------------------------------------------------

template <typename ForwardIterator, typename Compare>
std::pair<ForwardIterator, ForwardIterator> brick_minmax_element(ForwardIterator first, ForwardIterator last, Compare comp, /* is_vector = */ std::false_type) noexcept {
    return std::minmax_element(first, last, comp);
}

template <typename ForwardIterator, typename Compare>
std::pair<ForwardIterator, ForwardIterator> brick_minmax_element(ForwardIterator first, ForwardIterator last, Compare comp, /* is_vector = */ std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::minmax_element(first, last, comp);
}

template <typename ForwardIterator, typename Compare, typename IsVector>
std::pair<ForwardIterator, ForwardIterator> pattern_minmax_element(ForwardIterator first, ForwardIterator last, Compare comp, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    return brick_minmax_element(first, last, comp, is_vector);
}

template <typename ForwardIterator, typename Compare, typename IsVector>
std::pair<ForwardIterator, ForwardIterator> pattern_minmax_element(ForwardIterator first, ForwardIterator last, Compare comp, IsVector is_vector, /* is_parallel = */ std::true_type) noexcept {
    if(first == last)
        return std::make_pair(first, first);

    return except_handler([=]() {
        typedef std::pair<ForwardIterator, ForwardIterator> result_t;

        return par_backend::parallel_reduce(
            first + 1, last, std::make_pair(first, first),
            [=](ForwardIterator begin, ForwardIterator end, result_t init) -> result_t {
                const result_t subresult = brick_minmax_element(begin, end, comp, is_vector);
                return std::make_pair(cmp_iterators_by_values(subresult.first, init.first, comp), cmp_iterators_by_values(init.second, subresult.second, not_pred<Compare>(comp)));
            },
            [=](result_t p1, result_t p2) -> result_t {
                return std::make_pair(cmp_iterators_by_values(p1.first, p2.first, comp), cmp_iterators_by_values(p2.second, p1.second, not_pred<Compare>(comp)));
            }
        );
    });
}

//------------------------------------------------------------------------
// mismatch
//------------------------------------------------------------------------
template<class InputIterator1, class InputIterator2, class BinaryPredicate>
std::pair<InputIterator1, InputIterator2> mismatch_serial(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, BinaryPredicate pred) {
#if __PSTL_CPP14_2RANGE_MISMATCH_EQUAL_PRESENT
    return std::mismatch(first1, last1, first2, last2, pred);
#else
    for (; first1 != last1 && first2 != last2 && pred(*first1, *first2); ++first1,++first2){ }
    return std::make_pair(first1, first2);
#endif
}

template <class InputIterator1, class InputIterator2, class Predicate>
std::pair<InputIterator1, InputIterator2> brick_mismatch(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Predicate pred, /* is_vector = */ std::false_type) noexcept {
    return mismatch_serial(first1, last1, first2, last2, pred);
}

template <class InputIterator1, class InputIterator2, class Predicate>
std::pair<InputIterator1, InputIterator2> brick_mismatch(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Predicate pred, /* is_vector = */ std::true_type) noexcept {
    auto n = std::min(last1 - first1, last2 - first2);
    return unseq_backend::simd_first(first1, n, first2, not_pred<Predicate>(pred));
}

template <class InputIterator1, class InputIterator2, class Predicate, class IsVector>
std::pair<InputIterator1, InputIterator2> pattern_mismatch(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Predicate pred, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    return brick_mismatch(first1, last1, first2, last2, pred, is_vector);
}

template <class InputIterator1, class InputIterator2, class Predicate, class IsVector>
std::pair<InputIterator1, InputIterator2> pattern_mismatch(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, Predicate pred, IsVector is_vector, /* is_parallel = */ std::true_type) noexcept {
    return except_handler([=]() {
        auto n = std::min(last1 - first1, last2 - first2);
        auto result = internal::parallel_find(first1, first1 + n, [first1, first2, pred, is_vector](InputIterator1 i, InputIterator1 j) {
            return brick_mismatch(i, j, first2 + (i - first1), first2 + (j - first1), pred, is_vector).first;
        },
        std::less<typename std::iterator_traits<InputIterator1>::difference_type>(), /*is_first=*/true);
        return std::make_pair(result, first2 + (result - first1));
    });
}

//------------------------------------------------------------------------
// lexicographical_compare
//------------------------------------------------------------------------

template<class ForwardIterator1, class ForwardIterator2, class Compare>
bool brick_lexicographical_compare(ForwardIterator1 first1, ForwardIterator1 last1,
    ForwardIterator2 first2, ForwardIterator2 last2, Compare comp, /* is_vector = */ std::false_type) noexcept {
    return std::lexicographical_compare(first1, last1, first2, last2, comp);
}

template<class ForwardIterator1, class ForwardIterator2, class Compare>
bool brick_lexicographical_compare(ForwardIterator1 first1, ForwardIterator1 last1,
    ForwardIterator2 first2, ForwardIterator2 last2, Compare comp, /* is_vector = */ std::true_type) noexcept {

    if (first2 == last2) { // if second sequence is empty
        return false;
    }
    else if (first1 == last1) { // if first sequence is empty
        return true;
    }
    else {
        typedef typename std::iterator_traits<ForwardIterator1>::reference ref_type1;
        typedef typename std::iterator_traits<ForwardIterator2>::reference ref_type2;
        --last1;
        --last2;
        auto n = std::min(last1 - first1, last2 - first2);
        std::pair<ForwardIterator1, ForwardIterator2> result = unseq_backend::simd_first(first1, n, first2,
            [comp](const ref_type1 x, const ref_type2 y)mutable {return comp(x, y) || comp(y, x); });

        if (result.first == last1 && result.second != last2) { // if first sequence shorter than second
            return !comp(*result.second, *result.first);;
        }
        else { // if second sequence shorter than first or both have the same number of elements
            return comp(*result.first, *result.second);
        }
    }
}

template<class ForwardIterator1, class ForwardIterator2, class Compare, class IsVector>
bool pattern_lexicographical_compare(ForwardIterator1 first1, ForwardIterator1 last1,
    ForwardIterator2 first2, ForwardIterator2 last2, Compare comp, IsVector is_vector, /* is_parallel = */ std::false_type) noexcept {
    return brick_lexicographical_compare(first1, last1, first2, last2, comp, is_vector);
}

template<class ForwardIterator1, class ForwardIterator2, class Compare, class IsVector>
bool pattern_lexicographical_compare(ForwardIterator1 first1, ForwardIterator1 last1,
    ForwardIterator2 first2, ForwardIterator2 last2, Compare comp, IsVector is_vector, /* is_parallel = */ std::true_type) noexcept {

    if (first2 == last2) { // if second sequence is empty
        return false;
    }
    else if (first1 == last1) { // if first sequence is empty
        return true;
    }
    else {
        typedef typename std::iterator_traits<ForwardIterator1>::reference ref_type1;
        typedef typename std::iterator_traits<ForwardIterator2>::reference ref_type2;
        --last1;
        --last2;
        auto n = std::min(last1 - first1, last2 - first2);
        auto result = internal::parallel_find(first1, first1 + n, [first1, first2, &comp, is_vector](ForwardIterator1 i, ForwardIterator1 j) {
            return brick_mismatch(i, j, first2 + (i - first1), first2 + (j - first1),
                [&comp](const ref_type1 x, const ref_type2 y) {return !comp(x, y) && !comp(y, x); }, is_vector).first;
        },
        std::less<typename std::iterator_traits<ForwardIterator1>::difference_type>(), /*is_first=*/true);

        if (result == last1 && first2 + (result - first1) != last2) { // if first sequence shorter than second
            return !comp(*(first2 + (result - first1)), *result);
        }
        else { // if second sequence shorter than first or both have the same number of elements
            return comp(*result, *(first2 + (result - first1)));
        }
    }
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_algorithm_impl_H */
