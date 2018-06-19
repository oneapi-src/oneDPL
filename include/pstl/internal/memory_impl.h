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

template<class InputIterator, class ForwardIterator>
ForwardIterator brick_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, /*vector=*/std::false_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::value_type value_type2;
    for (; first != last; ++first, ++result) {
        ::new (reduce_to_ptr(result)) value_type2(std::move(*first));
    }
    return result;
}

template<class InputIterator, class ForwardIterator>
ForwardIterator brick_uninitialized_move(InputIterator first, InputIterator last, ForwardIterator result, /*vector=*/std::true_type) noexcept {
    typedef typename std::iterator_traits<ForwardIterator>::value_type value_type2;
    return unseq_backend::simd_it_walk_2(first, last - first, result,
        [](InputIterator first1, ForwardIterator first2) {::new (reduce_to_ptr(first2)) value_type2(std::move(*first1));
    });
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_memory_impl_H */
