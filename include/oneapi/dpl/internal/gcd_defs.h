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

#ifndef DPCPP_GCD_DEFS_H_
#define DPCPP_GCD_DEFS_H_

#include <type_traits>

namespace oneapi
{
namespace dpl
{
#if __cplusplus > 201103L
template <typename _Mn, typename _Nn>
constexpr ::std::common_type_t<_Mn, _Nn>
gcd(_Mn __m, _Nn __n);

template <typename _Mn, typename _Nn>
constexpr ::std::common_type_t<_Mn, _Nn>
lcm(_Mn __m, _Nn __n);
#endif
} // end namespace dpl
} // end namespace oneapi

#endif
