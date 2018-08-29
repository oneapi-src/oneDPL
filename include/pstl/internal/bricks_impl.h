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

template<class _Iterator1, class _Iterator2, class _Function>
_Iterator2 brick_it_walk2(_Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2, _Function __f, /*vector=*/std::false_type) noexcept {
    for (; __first1 != __last1; ++__first1, ++__first2)
        __f(__first1, __first2);
    return __first2;
}

template<class _Iterator1, class _Iterator2, class _Function>
_Iterator2 brick_it_walk2(_Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2, _Function __f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_it_walk_2(__first1, __last1 - __first1, __first2, __f);
}

template<class _Iterator1, class _Size, class _Iterator2, class _Function>
_Iterator2 brick_it_walk2_n(_Iterator1 __first1, _Size __n, _Iterator2 __first2, _Function __f, /*vector=*/std::false_type) noexcept {
    for (; __n > 0; --__n, ++__first1, ++__first2)
        __f(__first1, __first2);
    return __first2;
}

template<class _Iterator1, class _Size, class _Iterator2, class _Function>
_Iterator2 brick_it_walk2_n(_Iterator1 __first1, _Size __n, _Iterator2 __first2, _Function __f, /*vector=*/std::true_type) noexcept {
    return unseq_backend::simd_it_walk_2(__first1, __n, __first2, __f);
}

} // namespace internal
} // namespace pstl

#endif /* __PSTL_bricks_impl_H */
