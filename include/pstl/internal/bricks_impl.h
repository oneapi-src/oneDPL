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

#ifndef __PSTL_bricks_impl_H
#define __PSTL_bricks_impl_H

namespace pstl {
namespace internal {

template<class Iterator1, class Iterator2, class Function>
Iterator2 brick_it_walk2(Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, /*vector=*/std::false_type) noexcept {
    for (; first1 != last1; ++first1, ++first2)
        f(first1, first2);
    return first2;
}

template<class Iterator1, class Iterator2, class Function>
Iterator2 brick_it_walk2(Iterator1 first1, Iterator1 last1, Iterator2 first2, Function f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_it_walk_2(first1, last1 - first1, first2, f);
}

template<class Iterator1, class Size, class Iterator2, class Function>
Iterator2 brick_it_walk2_n(Iterator1 first1, Size n, Iterator2 first2, Function f, /*vector=*/std::false_type) noexcept {
    for (; n > 0; --n, ++first1, ++first2)
        f(first1, first2);
    return first2;
}

template<class Iterator1, class Size, class Iterator2, class Function>
Iterator2 brick_it_walk2_n(Iterator1 first1, Size n, Iterator2 first2, Function f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_it_walk_2(first1, n, first2, f);
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_bricks_impl_H */
