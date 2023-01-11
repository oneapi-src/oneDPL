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

#ifndef _ONEDPL_SERIAL_ALGORITHM_IMPL_H
#define _ONEDPL_SERIAL_ALGORITHM_IMPL_H

#if _ONEDPL___cplusplus >= 201703L
#    include <algorithm>
#    include <iterator>
namespace oneapi
{
namespace dpl
{
#    if _ONEDPL_HAS_NUMERIC_SERIAL_IMPL
template <typename _InputIterator, typename _Size, typename _Function>
_InputIterator
for_each_n(_InputIterator __first, _Size __n, _Function __f)
{
    typename ::std::iterator_traits<_InputIterator>::difference_type __n2 = __n;
    while (__n2-- > 0)
    {
        __f(*__first);
        ++__first;
    }
    return __first;
}
#    else
using ::std::for_each_n;
#    endif // _ONEDPL_HAS_NUMERIC_SERIAL_IMPL
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL___cplusplus >= 201703L
#endif // _ONEDPL_SERIAL_ALGORITHM_IMPL_H
