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

#ifndef __PSTL_vector_impl_H
#define __PSTL_vector_impl_H

#include <algorithm> //for std::min
#include <type_traits>

#include "pstl_config.h"
#include "utils.h"

// This header defines the minimum set of vector routines required
// to support parallel STL.
namespace __pstl {
namespace unseq_backend {

template<class _Iterator, class _DifferenceType, class _Function>
_Iterator simd_walk_1(_Iterator __first, _DifferenceType __n, _Function __f) noexcept {
__PSTL_PRAGMA_SIMD
    for(_DifferenceType __i = 0; __i < __n; ++__i)
        __f(__first[__i]);

    return __first + __n;
}

template<class _Iterator1, class _DifferenceType, class _Iterator2, class _Function>
_Iterator2 simd_walk_2(_Iterator1 __first1, _DifferenceType __n, _Iterator2 __first2, _Function __f) noexcept {
__PSTL_PRAGMA_SIMD
    for(_DifferenceType __i = 0; __i < __n; ++__i)
        __f(__first1[__i], __first2[__i]);
    return __first2 + __n;
}

template<class _Iterator1, class _DifferenceType, class _Iterator2, class _Iterator3, class _Function>
_Iterator3 simd_walk_3(_Iterator1 __first1, _DifferenceType __n, _Iterator2 __first2, _Iterator3 __first3, _Function __f) noexcept {
__PSTL_PRAGMA_SIMD
    for(_DifferenceType __i = 0; __i < __n; ++__i)
        __f(__first1[__i], __first2[__i], __first3[__i]);
    return __first3 + __n;
}

// TODO: check whether simd_first() can be used here
template<class _Index, class _DifferenceType, class _Pred>
bool simd_or(_Index __first, _DifferenceType __n, _Pred __pred) noexcept {
#if __PSTL_EARLYEXIT_PRESENT
    _DifferenceType __i;
__PSTL_PRAGMA_VECTOR_UNALIGNED
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for(__i = 0; __i < __n; ++__i)
        if(__pred(__first[__i]))
            break;
    return __i < __n;
#else
    _DifferenceType __block_size = std::min<_DifferenceType>(4, __n);
    const _Index __last = __first + __n;
    while ( __last != __first ) {
        int32_t __flag = 1;
__PSTL_PRAGMA_SIMD_REDUCTION(&:__flag)
        for ( _DifferenceType __i = 0; __i < __block_size; ++__i )
            if ( __pred(*(__first + __i)) )
                __flag = 0;
        if ( !__flag )
            return true;

        __first += __block_size;
        if ( __last - __first >= __block_size << 1 ) {
            // Double the block size.  Any unnecessary iterations can be amortized against work done so far.
            __block_size <<= 1;
        }
        else {
            __block_size = __last - __first;
        }
    }
    return false;
#endif
}

template<class _Index, class _DifferenceType, class _Compare>
_Index simd_first(_Index __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp) noexcept {
#if __PSTL_EARLYEXIT_PRESENT
    _DifferenceType i = __begin;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for (; __i < __end; ++__i) {
        if (__comp(__first, __i)) {
            break;
        }
    }
    return __first + __i;
#else
    // Experiments show good block sizes like this
    const _DifferenceType __block_size = 8;
    alignas(64) _DifferenceType __lane[__block_size] = { 0 };
    while (__end - __begin >= __block_size) {
        _DifferenceType __found = 0;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_REDUCTION(| :__found)
        for (_DifferenceType __i = __begin; __i < __begin + __block_size; ++__i) {
            const _DifferenceType __t = __comp(__first, __i);
            __lane[__i - __begin] = __t;
            __found |= __t;
        }
        if (__found) {
            _DifferenceType __i;
            // This will vectorize
            for (__i = 0; __i < __block_size; ++__i) {
                if (__lane[__i]) {
                    break;
                }
            }
            return __first + __begin + __i;
        }
        __begin += __block_size;
    }

    //Keep remainder scalar
    while (__begin != __end) {
        if (__comp(__first, __begin)) {
            return __first + __begin;
        }
        ++__begin;
    }
    return __first + __end;
#endif //__PSTL_EARLYEXIT_PRESENT
}

template<class _Index1, class _DifferenceType, class _Index2, class _Pred>
std::pair<_Index1, _Index2> simd_first(_Index1 __first1, _DifferenceType __n, _Index2 __first2, _Pred __pred) noexcept {
#if __PSTL_EARLYEXIT_PRESENT
    _DifferenceType i = 0;
__PSTL_PRAGMA_VECTOR_UNALIGNED
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for(;__i < __n; ++__i)
        if(__pred(__first1[__i], __first2[__i]))
            break;
    return std::make_pair(__first1 + __i, __first2 + __i);
#else
    const _Index1 __last1 = __first1 + __n;
    const _Index2 __last2 = __first2 + __n;
    // Experiments show good block sizes like this
    const _DifferenceType __block_size = 8;
    alignas(64) _DifferenceType __lane[__block_size] = {0};
    while ( __last1 - __first1 >= __block_size ) {
        _DifferenceType __found = 0;
        _DifferenceType __i;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_REDUCTION(|:__found)
        for ( __i = 0; __i < __block_size; ++__i ) {
            const _DifferenceType __t = __pred(__first1[__i], __first2[__i]);
            __lane[__i] = __t;
            __found |= __t;
        }
        if ( __found ) {
            _DifferenceType i;
            // This will vectorize
            for ( __i = 0; __i < __block_size; ++__i ) {
                if ( __lane[__i] ) break;
            }
            return std::make_pair(__first1 + __i, __first2 + __i);
        }
        __first1 += __block_size;
        __first2 += __block_size;
    }

    //Keep remainder scalar
    for(; __last1 != __first1; ++__first1, ++__first2)
        if ( __pred(*(__first1), *(__first2)) )
            return std::make_pair(__first1, __first2);

    return std::make_pair(__last1, __last2);
#endif //__PSTL_EARLYEXIT_PRESENT
}

template<class _Index, class _DifferenceType, class _Pred>
_DifferenceType simd_count(_Index __index, _DifferenceType __n, _Pred pred) noexcept {
    _DifferenceType __count = 0;
__PSTL_PRAGMA_SIMD_REDUCTION(+:__count)
    for (_DifferenceType __i = 0; __i < __n; ++__i)
        if (pred(*(__index + __i)))
            ++__count;

    return __count;
}

template<class _InputIterator, class _DifferenceType, class _OutputIterator, class __BinaryPredicate>
_OutputIterator simd_unique_copy(_InputIterator __first, _DifferenceType __n, _OutputIterator __result,
                                 __BinaryPredicate __pred) noexcept {
    if (__n == 0)
        return __result;

    _DifferenceType __cnt = 1;
    __result[0] = __first[0];

__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 1; __i < __n; ++__i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(__cnt:1)
        if (!__pred(__first[__i], __first[__i - 1])) {
            __result[__cnt] = __first[__i];
            ++__cnt;
        }
    }
    return __result + __cnt;
}

template<class _InputIterator, class _DifferenceType, class _OutputIterator, class _Assigner>
_OutputIterator simd_copy_move(_InputIterator __first, _DifferenceType __n, _OutputIterator __result, _Assigner __assigner) noexcept {
__PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __n; ++__i)
        __assigner(__first + __i, __result + __i);
    return __result + __n;
}

template<class _InputIterator, class _DifferenceType, class _OutputIterator, class UnaryPredicate>
_OutputIterator simd_copy_if(_InputIterator __first, _DifferenceType __n, _OutputIterator __result, UnaryPredicate __pred) noexcept {
    _DifferenceType __cnt = 0;

__PSTL_PRAGMA_SIMD
    for(_DifferenceType __i = 0; __i < __n; ++__i) {
        __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(__cnt:1)
            if(__pred(__first[__i])) {
                __result[__cnt] = __first[__i];
                ++__cnt;
            }
    }
    return __result + __cnt;
}

template<class _InputIterator, class _DifferenceType, class BinaryPredicate>
_DifferenceType simd_calc_mask_2(_InputIterator __first, _DifferenceType __n, bool* __restrict __mask, BinaryPredicate __pred) noexcept {
    _DifferenceType __count = 0;

__PSTL_PRAGMA_SIMD_REDUCTION(+:__count)
    for (_DifferenceType __i = 0; __i < __n; ++__i) {
        __mask[__i] = !__pred(__first[__i], __first[__i - 1]);
        __count += __mask[__i];
    }
    return __count;
}

template<class _InputIterator, class _DifferenceType, class UnaryPredicate>
_DifferenceType simd_calc_mask_1(_InputIterator __first, _DifferenceType __n, bool* __restrict __mask, UnaryPredicate __pred) noexcept {
    _DifferenceType __count = 0;

__PSTL_PRAGMA_SIMD_REDUCTION(+:__count)
    for (_DifferenceType __i = 0; __i < __n; ++__i) {
        __mask[__i] = __pred(__first[__i]);
        __count += __mask[__i];
    }
    return __count;
}

template<class _InputIterator, class _DifferenceType, class _OutputIterator>
void simd_copy_by_mask(_InputIterator __first, _DifferenceType __n, _OutputIterator __result, bool* __restrict __mask) noexcept {
    _DifferenceType __cnt = 0;
__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __n; ++__i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(__cnt:1)
        if (__mask[__i]) {
            __result[__cnt] = __first[__i];
            ++__cnt;
        }
    }
}

template<class _InputIterator, class _DifferenceType, class _OutputIterator1, class _OutputIterator2>
void simd_partition_by_mask(_InputIterator __first, _DifferenceType __n, _OutputIterator1 __out_true, _OutputIterator2 __out_false,
                            bool* __mask) noexcept {
    _DifferenceType __cnt_true = 0, __cnt_false = 0;
__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __n; ++__i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(__cnt_true:1, __cnt_false:1)
        if (__mask[__i]) {
            __out_true[__cnt_true] = __first[__i];
            ++__cnt_true;
        }
        else {
            __out_false[__cnt_false] = __first[__i];
            ++__cnt_false;
        }
    }
}

template<class _Index, class _DifferenceType, class _Tp>
_Index simd_fill_n(_Index __first, _DifferenceType __n, const _Tp& __value) noexcept {
__PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __n; ++__i)
        __first[__i] = __value;
    return __first + __n;
}

template<class _Index, class _DifferenceType, class _Generator>
_Index simd_generate_n(_Index __first, _DifferenceType __size, _Generator __g) noexcept {
__PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __size; ++__i)
        __first[__i] = __g();
    return __first + __size;
}

template<class _Index, class _BinaryPredicate>
_Index simd_adjacent_find(_Index __first, _Index __last, _BinaryPredicate __pred, bool __or_semantic) noexcept {
    if(__last - __first < 2)
        return __last;

    typedef typename std::iterator_traits<_Index>::difference_type _difference_type;
    _difference_type __i = 0;

#if __PSTL_EARLYEXIT_PRESENT
    //Some compiler versions fail to compile the following loop when iterators are used. Indices are used instead
    const _difference_type __n = __last - __first-1;
__PSTL_PRAGMA_VECTOR_UNALIGNED
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for(; __i < __n; ++__i)
        if(pred(__first[__i], __first[__i + 1]))
            break;

    return __i < __n ? __first + __i : __last;
#else
    // Experiments show good block sizes like this
    //TODO: to consider tuning block_size for various data types
    const _difference_type __block_size = 8;
    alignas(64) _difference_type __lane[__block_size] = {0};
    while ( __last - __first >= __block_size ) {
        _difference_type __found = 0;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_REDUCTION(|:__found)
        for ( __i = 0; __i < __block_size-1; ++__i ) {
            //TODO: to improve SIMD vectorization
            const _difference_type __t = __pred(*(__first + __i), *(__first + __i + 1));
            __lane[__i] = __t;
            __found |= __t;
        }

        //Process a pair of elements on a boundary of a data block
        if(__first + __block_size < __last && __pred(*(__first + __i), *(__first + __i + 1)))
            __lane[__i] = __found = 1;

        if ( __found ) {
            if(__or_semantic)
                return __first;

            // This will vectorize
            for ( __i = 0; __i < __block_size; ++__i )
                if ( __lane[__i] ) break;
            return __first + __i; //As far as found is true a result (lane[i] is true) is guaranteed
        }
        __first += __block_size;
    }
    //Process the rest elements
    for (; __last - __first > 1; ++__first)
        if(__pred(*__first, *(__first+1)))
            return __first;

    return __last;
#endif
}

template<typename _InputIterator1, typename _DifferenceType, typename _InputIterator2, typename _Tp, typename _BinaryOperation>
_Tp simd_transform_reduce(_InputIterator1 __first1, _DifferenceType __n, _InputIterator2 __first2, _Tp __init,
                        _BinaryOperation __binary_op) noexcept {
__PSTL_PRAGMA_SIMD_REDUCTION(+:__init)
    for(_DifferenceType __i = 0; __i < __n; ++__i)
        __init += __binary_op(__first1[__i], __first2[__i]);
    return __init;
};

template<typename _InputIterator, typename _DifferenceType, typename _Tp, typename _UnaryOperation>
_Tp simd_transform_reduce(_InputIterator __first, _DifferenceType __n, _Tp __init, _UnaryOperation __unary_op) noexcept {
__PSTL_PRAGMA_SIMD_REDUCTION(+:__init)
    for(_DifferenceType __i = 0; __i < __n; ++__i)
        __init += __unary_op(__first[__i]);
    return __init;
};

template<class _Iterator, class _DifferenceType, class _Function>
_Iterator simd_it_walk_1(_Iterator __first, _DifferenceType __n, _Function __f) noexcept {
__PSTL_PRAGMA_SIMD
    for(_DifferenceType __i = 0; __i < __n; ++__i)
        __f(__first + __i);

    return __first + __n;
}

template<class _Iterator1, class _DifferenceType, class _Iterator2, class _Function>
_Iterator2 simd_it_walk_2(_Iterator1 __first1, _DifferenceType __n, _Iterator2 __first2, _Function __f) noexcept {
__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __n; ++__i)
        __f(__first1 + __i, __first2 + __i);
    return __first2 + __n;
}

template<class _InputIterator, class _DifferenceType, class _OutputIterator1, class _OutputIterator2, class _UnaryPredicate>
std::pair<_OutputIterator1, _OutputIterator2>
simd_partition_copy(_InputIterator __first, _DifferenceType __n, _OutputIterator1 __out_true, _OutputIterator2 __out_false,
                    _UnaryPredicate __pred) noexcept {
    _DifferenceType __cnt_true = 0, __cnt_false = 0;

__PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 0; __i < __n; ++__i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(__cnt_true:1, __cnt_false : 1)
        if (__pred(__first[__i])) {
            __out_true[__cnt_true] = __first[__i];
            ++__cnt_true;
        }
        else {
            __out_false[__cnt_false] = __first[__i];
            ++__cnt_false;
        }
    }
    return std::make_pair(__out_true + __cnt_true, __out_false + __cnt_false);
}

template<class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1 simd_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
                                     _ForwardIterator2 __s_last, _BinaryPredicate __pred) noexcept {
    typedef typename std::iterator_traits<_ForwardIterator1>::difference_type _difference_type;

    const _difference_type __n1 = __last - __first;
    const _difference_type __n2 = __s_last - __s_first;
    if (__n1 == 0 || __n2 == 0) {
        return __last; // according to the standard
    }

    // Common case
    // If __first sequence larger than second then we'll run simd___first with parameters of __first sequence.
    // Otherwise, vice versa.
    if (__n1 < __n2)
    {
        for (; __first != __last; ++__first) {
            if (simd_or(__s_first, __n2,
                internal::equal_value_by_pred<decltype(*__first), _BinaryPredicate>(*__first, __pred))) {
                return __first;
            }
        }
    }
    else {
        for (; __s_first != __s_last; ++__s_first) {
            const auto __result = unseq_backend::simd_first(__first, _difference_type(0), __n1,
                [__s_first, &__pred](_ForwardIterator1 __it, _difference_type __i) {return __pred(__it[__i], *__s_first); });
            if (__result != __last) {
                return __result;
            }
        }
    }
    return __last;
}

template<class _RandomAccessIterator, class _DifferenceType, class _UnaryPredicate>
_RandomAccessIterator simd_remove_if(_RandomAccessIterator __first, _DifferenceType __n, _UnaryPredicate __pred) noexcept {
    // find __first element we need to remove
    auto __current = unseq_backend::simd_first(__first, _DifferenceType(0), __n,
                                [&__pred](_RandomAccessIterator __it, _DifferenceType __i) {return __pred(__it[__i]); });
    __n -= __current - __first;

    // if we have in sequence only one element that pred(current[1]) != false we can exit the function
    if (__n < 2) {
        return __current;
    }

#if __PSTL_MONOTONIC_PRESENT
    _DifferenceType __cnt = 0;
    __PSTL_PRAGMA_SIMD
    for (_DifferenceType __i = 1; __i < __n; ++__i) {
        __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(__cnt:1)
        if (!__pred(__current[__i])) {
            __current[__cnt] = std::move(__current[__i]);
            ++cnt;
        }
    }
    return __current + __cnt;
#else
    return std::remove_if(__current, __current + __n, __pred);
#endif
}
} // namespace unseq_backend
} // namespace __pstl

#endif /* __PSTL_vector_impl_H */
