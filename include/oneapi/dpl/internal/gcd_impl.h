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

#ifndef _ONEDPL_GCD
#define _ONEDPL_GCD

#include <limits>
#include "gcd_defs.h"

namespace oneapi
{
namespace dpl
{

namespace internal
{
template <typename _Result, typename _Source, bool _IsSigned = ::std::is_signed<_Source>::value>
struct __ct_abs;

template <typename _Result, typename _Source>
struct __ct_abs<_Result, _Source, true>
{
    constexpr _Result
    operator()(_Source __t) const noexcept
    {
        if (__t >= 0)
            return __t;
        if (__t == ::std::numeric_limits<_Source>::min())
            return -static_cast<_Result>(__t);
        return -__t;
    }
};

template <typename _Result, typename _Source>
struct __ct_abs<_Result, _Source, false>
{
    constexpr _Result
    operator()(_Source __t) const noexcept
    {
        return __t;
    }
};

} // namespace internal

#if __cplusplus > 201103L
// gcd
template <typename _Mn, typename _Nn>
constexpr ::std::common_type_t<_Mn, _Nn>
gcd(_Mn __m, _Nn __n)
{
    static_assert((::std::is_integral<_Mn>::value && ::std::is_integral<_Nn>::value),
                  "Arguments to gcd must be integer types");
    static_assert((!::std::is_same<typename ::std::remove_cv<_Mn>::type, bool>::value),
                  "First argument to gcd cannot be bool");
    static_assert((!::std::is_same<typename ::std::remove_cv<_Nn>::type, bool>::value),
                  "Second argument to gcd cannot be bool");
    using _Rp = ::std::common_type_t<_Mn, _Nn>;
    using _Wp = ::std::make_unsigned_t<_Rp>;
    _Wp __m1 = static_cast<_Wp>(oneapi::dpl::internal::__ct_abs<_Rp, _Mn>()(__m));
    _Wp __n1 = static_cast<_Wp>(oneapi::dpl::internal::__ct_abs<_Rp, _Nn>()(__n));

    while (__n1 != 0)
    {
        _Wp __t = __m1 % __n1;
        __m1 = __n1;
        __n1 = __t;
    }
    return static_cast<_Rp>(__m1);
}

// lcm
template <typename _Mn, typename _Nn>
constexpr ::std::common_type_t<_Mn, _Nn>
lcm(_Mn __m, _Nn __n)
{
    static_assert((::std::is_integral<_Mn>::value && ::std::is_integral<_Nn>::value),
                  "Arguments to gcd must be integer types");
    static_assert((!::std::is_same<typename ::std::remove_cv<_Mn>::type, bool>::value),
                  "First argument to gcd cannot be bool");
    static_assert((!::std::is_same<typename ::std::remove_cv<_Nn>::type, bool>::value),
                  "Second argument to gcd cannot be bool");
    if (__m == 0 || __n == 0)
        return 0;
    using _Rp = ::std::common_type_t<_Mn, _Nn>;
    _Rp __val1 = oneapi::dpl::internal::__ct_abs<_Rp, _Mn>()(__m) / gcd(__m, __n);
    _Rp __val2 = oneapi::dpl::internal::__ct_abs<_Rp, _Nn>()(__n);
    return __val1 * __val2;
}
#endif

} // end namespace dpl
} // end namespace oneapi

#endif
