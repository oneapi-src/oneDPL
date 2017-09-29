/*
    Copyright (c) 2017 Intel Corporation

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

#ifndef __PSTL_memory_impl_H
#define __PSTL_memory_impl_H

#include <memory>
#include <exception>
#include "execution_policy_impl.h"

namespace __icp_algorithm {

//------------------------------------------------------------------------
// uninitialized_copy
//------------------------------------------------------------------------

template<class InputIterator, class ForwardIterator>
ForwardIterator brick_uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result, /*is_vector=*/std::false_type) noexcept {
    return std::uninitialized_copy(first, last, result);
}

template<class InputIterator, class ForwardIterator>
ForwardIterator brick_uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return std::uninitialized_copy(first, last, result);
}

template<class InputIterator, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_copy(first, last, result, is_vector);
}

template<class InputIterator, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_copy(first, last, result, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_copy_n
//------------------------------------------------------------------------

template<class InputIterator, class Size, class ForwardIterator>
ForwardIterator brick_uninitialized_copy_n(InputIterator first, Size n, ForwardIterator result, /*is_vector=*/std::false_type) noexcept {
    return std::uninitialized_copy_n(first, n, result);
}

template<class InputIterator, class Size, class ForwardIterator>
ForwardIterator brick_uninitialized_copy_n(InputIterator first, Size n, ForwardIterator result, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return std::uninitialized_copy_n(first, n, result);
}

template<class InputIterator, class Size, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_copy_n(InputIterator first, Size n, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_copy_n(first, n, result, is_vector);
}

template<class InputIterator, class Size, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_copy_n(InputIterator first, Size n, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_copy_n(first, n, result, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_move
//------------------------------------------------------------------------

template<class ForwardIterator>
void destroy_serial(ForwardIterator first, ForwardIterator last) {
    typedef typename std::iterator_traits<ForwardIterator>::value_type T;
    while (first != last) {
        (*first).~T();
        ++first;
    }
}

template<class InputIterator, class ForwardIterator>
ForwardIterator brick_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, /*is_vector=*/std::false_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::value_type Value;
    ForwardIterator current = result;

    try {
        while (first != last)
            new (static_cast<void*>(std::addressof(*(result++)))) Value(std::move(*(first++)));

        return result;
    } catch (...) {
        destroy_serial(current, result);
        std::terminate();
    }

    return result;
}

template<class InputIterator, class ForwardIterator>
ForwardIterator brick_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return brick_uninitialized_move(first, last, result, std::false_type());
}

template<class InputIterator, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_move(first, last, result, is_vector);
}

template<class InputIterator, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_move(first, last, result, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_move_n
//------------------------------------------------------------------------

template<class InputIterator, class Size, class ForwardIterator>
ForwardIterator brick_uninitialized_move_n(InputIterator first, Size n, ForwardIterator result, /*is_vector=*/std::false_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::value_type Value;
    ForwardIterator current = result;

    try {
        while (n-- > 0)
            new (static_cast<void*>(std::addressof(*(result++)))) Value(std::move(*(first++)));

        return result;
    } catch (...) {
        destroy_serial(current, result);
        std::terminate();
    }

    return result;
}

template<class InputIterator, class Size, class ForwardIterator>
ForwardIterator brick_uninitialized_move_n(InputIterator first, Size n, ForwardIterator result, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return brick_uninitialized_move_n(first, n, result, std::false_type());
}

template<class InputIterator, class Size, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_move_n(InputIterator first, Size n, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_move_n(first, n, result, is_vector);
}

template<class InputIterator, class Size, class ForwardIterator, class IsVector>
ForwardIterator pattern_uninitialized_move_n(InputIterator first, Size n, ForwardIterator result, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_move_n(first, n, result, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_fill
//------------------------------------------------------------------------

template<class ForwardIterator, class T>
void brick_uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x, /*is_vector=*/std::false_type) noexcept {
    std::uninitialized_fill(first, last, x);
}

template<class ForwardIterator, class T>
void brick_uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    std::uninitialized_fill(first, last, x);
}

template<class ForwardIterator, class T, class IsVector>
void pattern_uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    brick_uninitialized_fill(first, last, x, is_vector);
}

template<class ForwardIterator, class T, class IsVector>
void pattern_uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    brick_uninitialized_fill(first, last, x, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_fill_n
//------------------------------------------------------------------------
// Some C++11 compilers don't have a version of the algorithm std::uninitialized_fill_n that returns an iterator to the element past the last element filled.
template< class ForwardIterator, class Size, class T >
ForwardIterator uninitialized_fill_n_serial(ForwardIterator first, Size n, const T& x)
{
    typedef typename std::iterator_traits<ForwardIterator>::value_type Value;
    auto cur = first;
    try {
        while (n--) {
            ::new (static_cast<void*>(std::addressof(*cur))) Value(x);
            ++cur;
        }
        return cur;
    }
    catch (...) {
        destroy_serial(first, cur);
        std::terminate();
    }
}

template<class ForwardIterator, class Size, class T>
ForwardIterator brick_uninitialized_fill_n(ForwardIterator first, Size n, const T& x, /*is_vector=*/std::false_type) noexcept {
    return uninitialized_fill_n_serial(first, n, x);
}

template<class ForwardIterator, class Size, class T>
ForwardIterator brick_uninitialized_fill_n(ForwardIterator first, Size n, const T& x, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return brick_uninitialized_fill_n(first, n, x, std::false_type());
}

template<class ForwardIterator, class Size, class T, class IsVector>
ForwardIterator pattern_uninitialized_fill_n(ForwardIterator first, Size n, const T& x, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_fill_n(first, n, x, is_vector);
}

template<class ForwardIterator, class Size, class T, class IsVector>
ForwardIterator pattern_uninitialized_fill_n(ForwardIterator first, Size n, const T& x, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_fill_n(first, n, x, is_vector);
}

//------------------------------------------------------------------------
// destroy
//------------------------------------------------------------------------

template<class ForwardIterator>
void brick_destroy(ForwardIterator first, ForwardIterator last, /*is_vector=*/std::false_type) noexcept {
    destroy_serial(first, last);
}

template<class ForwardIterator>
void brick_destroy(ForwardIterator first, ForwardIterator last, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    brick_destroy(first, last, std::false_type());
}

template<class ForwardIterator, class IsVector>
void pattern_destroy(ForwardIterator first, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    brick_destroy(first, last, is_vector);
}

template<class ForwardIterator, class IsVector>
void pattern_destroy(ForwardIterator first, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    brick_destroy(first, last, is_vector);
}

//------------------------------------------------------------------------
// destroy_n
//------------------------------------------------------------------------

template<class ForwardIterator, class Size>
ForwardIterator destroy_n_serial(ForwardIterator first, Size n) {
    typedef typename std::iterator_traits<ForwardIterator>::value_type T;
    while (n--) {
        (*first).~T();
        ++first;
    }
    return first;
}

template<class ForwardIterator, class Size>
ForwardIterator brick_destroy_n(ForwardIterator first, Size n, /*is_vector=*/std::false_type) noexcept {
    return destroy_n_serial(first, n);
}

template<class ForwardIterator, class Size>
ForwardIterator brick_destroy_n(ForwardIterator first, Size n, /*is_vector=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return brick_destroy_n(first, n, std::false_type());
}

template<class ForwardIterator, class Size, class IsVector>
ForwardIterator pattern_destroy_n(ForwardIterator first, Size n, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_destroy_n(first, n, is_vector);
}

template<class ForwardIterator, class Size, class IsVector>
ForwardIterator pattern_destroy_n(ForwardIterator first, Size n, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
__PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_destroy_n(first, n, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_default_construct
//------------------------------------------------------------------------
template <typename T, typename value_tag>
struct Construct {
    void operator()(void* ptr) {
        ::new (ptr) T;
    }
};

template <typename T>
struct Construct<T, std::true_type> {
    void operator()(void* ptr) {
        ::new (ptr) T();
    }
};

template <class ValueTag, class ForwardIterator>
void brick_uninitialized_construct(ForwardIterator first, ForwardIterator last, /*is_vector=*/std::false_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::value_type value_type;
    auto cur = first; // Save the iterator for catching exceptions
    try {
        for (; cur != last; ++cur)
            Construct<value_type, ValueTag>()(static_cast<void*>(std::addressof(*cur)));
    }
    catch (...) {
        destroy_serial(first, cur);
        std::terminate();
    }
}

template <class ValueTag, class ForwardIterator>
void brick_uninitialized_construct(ForwardIterator first, ForwardIterator last, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    brick_uninitialized_construct<ValueTag, ForwardIterator >(first, last, std::false_type());
}

template <class ForwardIterator, class IsVector>
void pattern_uninitialized_default_construct(ForwardIterator first, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    brick_uninitialized_construct<std::false_type, ForwardIterator>(first, last, is_vector);
}

template <class ForwardIterator, class IsVector>
void pattern_uninitialized_default_construct(ForwardIterator first, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    brick_uninitialized_construct<std::false_type, ForwardIterator>(first, last, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_default_construct_n
//------------------------------------------------------------------------

template <class ValueTag, class ForwardIterator, class Size>
ForwardIterator brick_uninitialized_construct_n(ForwardIterator first, Size n, /*is_vector=*/std::false_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::value_type value_type;
    auto cur = first;
    try {
        for (; n > 0; ++cur, --n)
            Construct<value_type, ValueTag>()(static_cast<void*>(std::addressof(*cur)));
        return cur;
    }
    catch (...) {
        destroy_serial(first, cur);
        std::terminate();
    }
}

template <class ValueTag, class ForwardIterator, class Size>
ForwardIterator brick_uninitialized_construct_n(ForwardIterator first, Size n, /*is_vector=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Vectorial algorithm unimplemented, redirected to serial");
    return brick_uninitialized_construct_n<ValueTag, ForwardIterator, Size>(first, n, std::false_type());
}

template <class ForwardIterator, class Size, class IsVector>
ForwardIterator pattern_uninitialized_default_construct_n(ForwardIterator first, Size n, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_construct_n<std::false_type, ForwardIterator, Size>(first, n, is_vector);
}

template <class ForwardIterator, class Size, class IsVector>
ForwardIterator pattern_uninitialized_default_construct_n(ForwardIterator first, Size n, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_construct_n<std::false_type, ForwardIterator, Size>(first, n, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_value_construct
//------------------------------------------------------------------------

template <class ForwardIterator, class IsVector>
void pattern_uninitialized_value_construct(ForwardIterator first, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    brick_uninitialized_construct<std::true_type, ForwardIterator>(first, last, is_vector);
}

template <class ForwardIterator, class IsVector>
void pattern_uninitialized_value_construct(ForwardIterator first, ForwardIterator last, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    brick_uninitialized_construct<std::true_type, ForwardIterator>(first, last, is_vector);
}

//------------------------------------------------------------------------
// uninitialized_value_construct_n
//------------------------------------------------------------------------

template <class ForwardIterator, class Size, class IsVector>
ForwardIterator pattern_uninitialized_value_construct_n(ForwardIterator first, Size n, IsVector is_vector, /*is_parallel=*/std::false_type) noexcept {
    return brick_uninitialized_construct_n<std::true_type, ForwardIterator, Size>(first, n, is_vector);
}

template <class ForwardIterator, class Size, class IsVector>
ForwardIterator pattern_uninitialized_value_construct_n(ForwardIterator first, Size n, IsVector is_vector, /*is_parallel=*/std::true_type) noexcept {
    __PSTL_PRAGMA_MESSAGE("Parallel algorithm unimplemented, redirected to serial");
    return brick_uninitialized_construct_n<std::true_type, ForwardIterator, Size>(first, n, is_vector);
}
} // namespace __icp_algorithm
#endif //__PSTL_memory_impl_H
