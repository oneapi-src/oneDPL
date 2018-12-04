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

#ifndef __PSTL_memory_impl_H
#define __PSTL_memory_impl_H

#include <iterator>

#include "unseq_backend_simd.h"

namespace pstl {
namespace internal {

//------------------------------------------------------------------------
// uninitialized_move
//------------------------------------------------------------------------

template<class _ForwardIterator, class _OutputIterator>
_OutputIterator brick_uninitialized_move(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, /*vector=*/std::false_type) noexcept {
    typedef typename std::iterator_traits<_OutputIterator>::value_type _ValueType2;
    for (; __first != __last; ++__first, ++__result) {
        ::new (std::addressof(*__result)) _ValueType2(std::move(*__first));
    }
    return __result;
}

template<class _ForwardIterator, class _OutputIterator>
_OutputIterator brick_uninitialized_move(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, /*vector=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<_OutputIterator>::value_type __ValueType2;
    typedef typename std::iterator_traits<_ForwardIterator>::reference _ReferenceType1;
    typedef typename std::iterator_traits<_OutputIterator>::reference _ReferenceType2;

    return unseq_backend::simd_walk_2(__first, __last - __first, __result,
        [](_ReferenceType1 __x, _ReferenceType2 __y) { ::new (std::addressof(__y)) __ValueType2(std::move(__x));
    });
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_memory_impl_H */
