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

#ifndef __PSTL_unseq_backend_simd_H
#define __PSTL_unseq_backend_simd_H

#include <algorithm> //for std::min
#include <type_traits>

#include "pstl_config.h"
#include "utils.h"

// This header defines the minimum set of vector routines required
// to support parallel STL.
namespace pstl {
namespace unseq_backend {

template<class Iterator, class DifferenceType, class Function>
Iterator simd_walk_1(Iterator first, DifferenceType n, Function f) noexcept {
__PSTL_PRAGMA_SIMD
    for(DifferenceType i = 0; i < n; ++i)
        f(first[i]);

    return first + n;
}

template<class Iterator1, class DifferenceType, class Iterator2, class Function>
Iterator2 simd_walk_2(Iterator1 first1, DifferenceType n, Iterator2 first2, Function f) noexcept {
__PSTL_PRAGMA_SIMD
    for(DifferenceType i = 0; i < n; ++i)
        f(first1[i], first2[i]);
    return first2 + n;
}

template<class Iterator1, class DifferenceType, class Iterator2, class Iterator3, class Function>
Iterator3 simd_walk_3(Iterator1 first1, DifferenceType n, Iterator2 first2, Iterator3 first3, Function f) noexcept {
__PSTL_PRAGMA_SIMD
    for(DifferenceType i = 0; i < n; ++i)
        f(first1[i], first2[i], first3[i]);
    return first3 + n;
}

// TODO: check whether simd_first() can be used here
template<class Index, class DifferenceType, class Pred>
bool simd_or(Index first, DifferenceType n, Pred pred) noexcept {
#if __PSTL_EARLYEXIT_PRESENT
    DifferenceType i;
__PSTL_PRAGMA_VECTOR_UNALIGNED
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for(i = 0; i < n; ++i)
        if(pred(first[i]))
            break;
    return i < n;
#else
    DifferenceType block_size = std::min<DifferenceType>(4, n);
    const Index last = first + n;
    while ( last != first ) {
        int32_t flag = 1;
__PSTL_PRAGMA_SIMD_REDUCTION(&:flag)
        for ( DifferenceType i = 0; i < block_size; ++i )
            if ( pred(*(first + i)) )
                flag = 0;
        if ( !flag )
            return true;

        first += block_size;
        if ( last - first >= block_size << 1 ) {
            // Double the block size.  Any unnecessary iterations can be amortized against work done so far.
            block_size <<= 1;
        }
        else {
            block_size = last - first;
        }
    }
    return false;
#endif
}

template<class Index, class DifferenceType, class Compare>
Index simd_first(Index first, DifferenceType begin, DifferenceType end, Compare comp) noexcept {
#if __PSTL_EARLYEXIT_PRESENT
    DifferenceType i = begin;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for (; i < end; ++i) {
        if (comp(first, i)) {
            break;
        }
    }
    return first + i;
#else
    // Experiments show good block sizes like this
    const DifferenceType block_size = 8;
    alignas(64) DifferenceType lane[block_size] = { 0 };
    while (end - begin >= block_size) {
        DifferenceType found = 0;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_REDUCTION(| :found)
        for (DifferenceType i = begin; i < begin + block_size; ++i) {
            const DifferenceType t = comp(first, i);
            lane[i - begin] = t;
            found |= t;
        }
        if (found) {
            DifferenceType i;
            // This will vectorize
            for (i = 0; i < block_size; ++i) {
                if (lane[i]) {
                    break;
                }
            }
            return first + begin + i;
        }
        begin += block_size;
    }

    //Keep remainder scalar
    while (begin != end) {
        if (comp(first, begin)) {
            return first + begin;
        }
        ++begin;
    }
    return first + end;
#endif //__PSTL_EARLYEXIT_PRESENT
}

template<class Index1, class DifferenceType, class Index2, class Pred>
std::pair<Index1, Index2> simd_first(Index1 first1, DifferenceType n, Index2 first2, Pred pred) noexcept {
#if __PSTL_EARLYEXIT_PRESENT
    DifferenceType i = 0;
__PSTL_PRAGMA_VECTOR_UNALIGNED
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for(;i < n; ++i)
        if(pred(first1[i], first2[i]))
            break;
    return std::make_pair(first1 + i, first2 + i);
#else
    const Index1 last1 = first1 + n;
    const Index2 last2 = first2 + n;
    // Experiments show good block sizes like this
    const DifferenceType block_size = 8;
    alignas(64) DifferenceType lane[block_size] = {0};
    while ( last1 - first1 >= block_size ) {
        DifferenceType found = 0;
        DifferenceType i;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_REDUCTION(|:found)
        for ( i = 0; i < block_size; ++i ) {
            const DifferenceType t = pred(first1[i], first2[i]);
            lane[i] = t;
            found |= t;
        }
        if ( found ) {
            DifferenceType i;
            // This will vectorize
            for ( i = 0; i < block_size; ++i ) {
                if ( lane[i] ) break;
            }
            return std::make_pair(first1 + i, first2 + i);
        }
        first1 += block_size;
        first2 += block_size;
    }

    //Keep remainder scalar
    for(; last1 != first1; ++first1, ++first2)
        if ( pred(*(first1), *(first2)) )
            return std::make_pair(first1, first2);

    return std::make_pair(last1, last2);
#endif //__PSTL_EARLYEXIT_PRESENT
}

template<class Index, class DifferenceType, class Pred>
DifferenceType simd_count(Index first, DifferenceType n, Pred pred) noexcept {
    DifferenceType count = 0;
__PSTL_PRAGMA_SIMD_REDUCTION(+:count)
    for (DifferenceType i = 0; i < n; ++i)
        if (pred(*(first + i)))
            ++count;

    return count;
}

template<class InputIterator, class DifferenceType, class OutputIterator, class BinaryPredicate>
OutputIterator simd_unique_copy(InputIterator first, DifferenceType n, OutputIterator result, BinaryPredicate pred) noexcept {
    if (n == 0)
        return result;

    DifferenceType cnt = 1;
    result[0] = first[0];

__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 1; i < n; ++i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(cnt:1)
        if (!pred(first[i], first[i - 1])) {
            result[cnt] = first[i];
            ++cnt;
        }
    }
    return result + cnt;
}

template<class InputIterator, class DifferenceType, class OutputIterator, class Assigner>
OutputIterator simd_assign(InputIterator first, DifferenceType n, OutputIterator result, Assigner assigner) noexcept {
__PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < n; ++i)
        assigner(first + i, result + i);
    return result + n;
}

template<class InputIterator, class DifferenceType, class OutputIterator, class UnaryPredicate>
OutputIterator simd_copy_if(InputIterator first, DifferenceType n, OutputIterator result, UnaryPredicate pred) noexcept {
    DifferenceType cnt = 0;

__PSTL_PRAGMA_SIMD
    for(DifferenceType i = 0; i < n; ++i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(cnt:1)
        if(pred(first[i])) {
            result[cnt] = first[i];
            ++cnt;
        }
    }
    return result + cnt;
}

template<class InputIterator, class DifferenceType, class BinaryPredicate>
DifferenceType simd_calc_mask_2(InputIterator first, DifferenceType n, bool* __restrict mask, BinaryPredicate pred) noexcept {
    DifferenceType count = 0;

__PSTL_PRAGMA_SIMD_REDUCTION(+:count)
    for (DifferenceType i = 0; i < n; ++i) {
        mask[i] = !pred(first[i], first[i - 1]);
        count += mask[i];
    }
    return count;
}

template<class InputIterator, class DifferenceType, class UnaryPredicate>
DifferenceType simd_calc_mask_1(InputIterator first, DifferenceType n, bool* __restrict mask, UnaryPredicate pred) noexcept {
    DifferenceType count = 0;

__PSTL_PRAGMA_SIMD_REDUCTION(+:count)
    for (DifferenceType i = 0; i < n; ++i) {
        mask[i] = pred(first[i]);
        count += mask[i];
    }
    return count;
}

template<class InputIterator, class DifferenceType, class OutputIterator>
void simd_copy_by_mask(InputIterator first, DifferenceType n, OutputIterator result, bool* __restrict mask) noexcept {
    DifferenceType cnt = 0;
__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < n; ++i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(cnt:1)
        if (mask[i]) {
            result[cnt] = first[i];
            ++cnt;
        }
    }
}

template<class InputIterator, class DifferenceType, class OutputIterator1, class OutputIterator2>
void simd_partition_by_mask(InputIterator first, DifferenceType n, OutputIterator1 out_true, OutputIterator2 out_false, bool* mask) noexcept {
    DifferenceType cnt_true = 0, cnt_false = 0;
__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < n; ++i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(cnt_true:1, cnt_false:1)
        if (mask[i]) {
            out_true[cnt_true] = first[i];
            ++cnt_true;
        }
        else {
            out_false[cnt_false] = first[i];
            ++cnt_false;
        }
    }
}

template<class Index, class DifferenceType, class T>
Index simd_fill_n(Index first, DifferenceType n, const T& value) noexcept {
__PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < n; ++i)
        first[i] = value;
    return first + n;
}

template<class Index, class DifferenceType, class Generator>
Index simd_generate_n(Index first, DifferenceType size, Generator g) noexcept {
__PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < size; ++i)
        first[i] = g();
    return first + size;
}

template<class Index, class BinaryPredicate>
Index simd_adjacent_find(Index first, Index last, BinaryPredicate pred, bool or_semantic) noexcept {
    if(last - first < 2)
        return last;

    typedef typename std::iterator_traits<Index>::difference_type difference_type;
    difference_type i = 0;

#if __PSTL_EARLYEXIT_PRESENT
    //Some compiler versions fail to compile the following loop when iterators are used. Indices are used instead
    const difference_type n = last-first-1;
__PSTL_PRAGMA_VECTOR_UNALIGNED
__PSTL_PRAGMA_SIMD_EARLYEXIT
    for(; i < n; ++i)
        if(pred(first[i], first[i+1]))
            break;

    return i < n ? first + i : last;
#else
    // Experiments show good block sizes like this
    //TODO: to consider tuning block_size for various data types
    const difference_type block_size = 8;
    alignas(64) difference_type lane[block_size] = {0};
    while ( last - first >= block_size ) {
        difference_type found = 0;
__PSTL_PRAGMA_VECTOR_UNALIGNED // Do not generate peel loop part
__PSTL_PRAGMA_SIMD_REDUCTION(|:found)
        for ( i = 0; i < block_size-1; ++i ) {
            //TODO: to improve SIMD vectorization
            const difference_type t = pred(*(first + i), *(first + i + 1));
            lane[i] = t;
            found |= t;
        }

        //Process a pair of elements on a boundary of a data block
        if(first + block_size < last && pred(*(first + i), *(first + i + 1)))
            lane[i] = found = 1;

        if ( found ) {
            if(or_semantic)
                return first;

            // This will vectorize
            for ( i = 0; i < block_size; ++i )
                if ( lane[i] ) break;
            return first + i; //As far as found is true a result (lane[i] is true) is guaranteed
        }
        first += block_size;
    }
    //Process the rest elements
    for (; last - first > 1; ++first)
        if(pred(*first, *(first+1)))
            return first;

    return last;
#endif
}

template<typename InputIterator1, typename DifferenceType, typename InputIterator2, typename T, typename BinaryOperation>
T simd_transform_reduce(InputIterator1 first1, DifferenceType n, InputIterator2 first2, T init, BinaryOperation binary_op) noexcept {
__PSTL_PRAGMA_SIMD_REDUCTION(+:init)
    for(DifferenceType i = 0; i < n; ++i)
        init += binary_op(first1[i], first2[i]);
    return init;
};

template<typename InputIterator, typename DifferenceType, typename T, typename UnaryOperation>
T simd_transform_reduce(InputIterator first, DifferenceType n, T init, UnaryOperation unary_op) noexcept {
__PSTL_PRAGMA_SIMD_REDUCTION(+:init)
    for(DifferenceType i = 0; i < n; ++i)
        init += unary_op(first[i]);
    return init;
};

template<class Iterator, class DifferenceType, class Function>
Iterator simd_it_walk_1(Iterator first, DifferenceType n, Function f) noexcept {
__PSTL_PRAGMA_SIMD
    for(DifferenceType i = 0; i < n; ++i)
        f(first + i);

    return first + n;
}

template<class Iterator1, class DifferenceType, class Iterator2, class Function>
Iterator2 simd_it_walk_2(Iterator1 first1, DifferenceType n, Iterator2 first2, Function f) noexcept {
__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < n; ++i)
        f(first1 + i, first2 + i);
    return first2 + n;
}

template<class InputIterator, class DifferenceType, class OutputIterator1, class OutputIterator2, class UnaryPredicate>
std::pair<OutputIterator1, OutputIterator2>
simd_partition_copy(InputIterator first, DifferenceType n, OutputIterator1 out_true, OutputIterator2 out_false, UnaryPredicate pred) noexcept {
    DifferenceType cnt_true = 0, cnt_false = 0;

__PSTL_PRAGMA_SIMD
    for (DifferenceType i = 0; i < n; ++i) {
__PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(cnt_true:1, cnt_false : 1)
        if (pred(first[i])) {
            out_true[cnt_true] = first[i];
            ++cnt_true;
        }
        else {
            out_false[cnt_false] = first[i];
            ++cnt_false;
        }
    }
    return std::make_pair(out_true + cnt_true, out_false + cnt_false);
}

template<class ForwardIterator1, class ForwardIterator2, class BinaryPredicate>
ForwardIterator1 simd_find_first_of(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last, BinaryPredicate pred) noexcept {
    typedef typename std::iterator_traits<ForwardIterator1>::difference_type difference_type;

    const difference_type n1 = last - first;
    const difference_type n2 = s_last - s_first;
    if (n1 == 0 || n2 == 0) {
        return last; // according to the standard
    }

    // Common case
    // If first sequence larger than second then we'll run simd_first with parameters of first sequence.
    // Otherwise, vice versa.
    if (n1 < n2)
    {
        for (; first != last; ++first) {
            if (simd_or(s_first, n2,
                internal::equal_value_by_pred<decltype(*first), BinaryPredicate>(*first, pred))) {
                return first;
            }
        }
    }
    else {
        for (; s_first != s_last; ++s_first) {
            const auto result = simd_first(first, difference_type(0), n1,
                [s_first, &pred](ForwardIterator1 it, difference_type i) {return pred(it[i], *s_first); });
            if (result != last) {
                return result;
            }
        }
    }
    return last;
}

template<class ForwardIterator, class DifferenceType, class UnaryPredicate>
ForwardIterator simd_remove_if(ForwardIterator first, DifferenceType n, UnaryPredicate pred) noexcept {
    // find first element we need to remove
    auto current = simd_first(first, DifferenceType(0), n, [&pred](ForwardIterator it, DifferenceType i) {return pred(it[i]); });
    n -= current - first;

    // if we have in sequence only one element that pred(current[1]) != false we can exit the function
    if (n < 2) {
        return current;
    }

#if __PSTL_MONOTONIC_PRESENT
    DifferenceType cnt = 0;
    __PSTL_PRAGMA_SIMD
    for (DifferenceType i = 1; i < n; ++i) {
        __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(cnt:1)
        if (!pred(current[i])) {
            current[cnt] = std::move(current[i]);
            ++cnt;
        }
    }
    return current + cnt;
#else
    return std::remove_if(current, current + n, pred);
#endif
}
} // namespace unseq_backend
} // namespace pstl

#endif /* __PSTL_unseq_backend_simd_H */
