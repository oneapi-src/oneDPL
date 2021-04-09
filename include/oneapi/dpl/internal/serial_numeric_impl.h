/*
 *  Copyright (c) Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_SERIAL_NUMERIC_IMPL_H
#define _ONEDPL_SERIAL_NUMERIC_IMPL_H

#if __cplusplus < 201703L
#    define _ONEDPL_HAS_NUMERIC_SERIAL_IMPL 1
#else
// std::reduce is not available in libstdc++7, libstdc++8
#    if __GLIBC__ && (_GLIBCXX_RELEASE < 9)
#        define _ONEDPL_HAS_NUMERIC_SERIAL_IMPL 1
#    endif
#endif

#ifdef _ONEDPL_HAS_NUMERIC_SERIAL_IMPL
#    include <iterator>
namespace oneapi
{
namespace dpl
{
template <class _InputIterator, class _Tp, class _BinaryOp>
_Tp
reduce(_InputIterator __first, _InputIterator __last, _Tp __init, _BinaryOp __b)
{
    for (; __first != __last; ++__first)
        __init = __b(__init, *__first);
    return __init;
}

template <class _InputIterator, class _Tp>
_Tp
reduce(_InputIterator __first, _InputIterator __last, _Tp __init)
{
    return oneapi::dpl::reduce(__first, __last, __init, [](const _Tp& lhs, const _Tp& rhs) { return lhs + rhs; });
}

template <class _InputIterator>
typename ::std::iterator_traits<_InputIterator>::value_type
reduce(_InputIterator __first, _InputIterator __last)
{
    return oneapi::dpl::reduce(__first, __last, typename ::std::iterator_traits<_InputIterator>::value_type{});
}
} // namespace dpl
} // namespace oneapi
#else
#    include <numeric>
namespace oneapi
{
namespace dpl
{
using ::std::reduce;
} // namespace dpl
} // namespace oneapi
#endif
#endif /* _ONEDPL_SERIAL_NUMERIC_IMPL_H */
