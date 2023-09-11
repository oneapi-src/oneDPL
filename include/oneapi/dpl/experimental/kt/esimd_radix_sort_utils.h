// -*- C++ -*-
//===-- esimd_radix_sort_utils.h -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <ext/intel/esimd.hpp>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace oneapi::dpl::experimental::kt::esimd::__impl
{
constexpr int __data_per_step = 16;

namespace __dpl_esimd_ns = sycl::ext::intel::esimd;
namespace __dpl_esimd_ens = sycl::ext::intel::experimental::esimd;

namespace __utils
{

// converts sizeof(_T) to 32 bits, so that it could be used in operations with 32-bit SIMD without changing the type
template <typename _T>
inline constexpr ::std::uint32_t __size32 = sizeof(_T);

template <typename _T, int _N>
void
__copy_from(const _T* __input, ::std::uint32_t __base_offset, __dpl_esimd_ns::simd<_T, _N>& __values)
{
    __values.copy_from(__input + __base_offset);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P>
void
__copy_from(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __input, ::std::uint32_t __base_offset,
          __dpl_esimd_ns::simd<_T, _N>& __values)
{
    __values.copy_from(__input, __base_offset * __size32<_T>);
}

template <typename _T, int _N>
void
__copy_to(_T* __output, ::std::uint32_t __base_offset, const __dpl_esimd_ns::simd<_T, _N>& __values)
{
    __values.copy_to(__output + __base_offset);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P>
void
__copy_to(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __output, ::std::uint32_t __base_offset,
        const __dpl_esimd_ns::simd<_T, _N>& __values)
{
    __values.copy_to(__output, __base_offset * __size32<_T>);
}

template <typename _T, int _N>
__dpl_esimd_ns::simd<_T, _N>
__gather(const _T* __input, __dpl_esimd_ns::simd<::std::uint32_t, _N> __offsets, ::std::uint32_t __base_offset,
       __dpl_esimd_ns::simd_mask<_N> __mask = 1)
{
    return __dpl_esimd_ns::gather(__input + __base_offset, __offsets * __size32<_T>, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) <= sizeof(::std::uint32_t), int> = 0>
__dpl_esimd_ns::simd<_T, _N>
__gather(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __input, __dpl_esimd_ns::simd<::std::uint32_t, _N> __offsets,
       ::std::uint32_t __base_offset, __dpl_esimd_ns::simd_mask<_N> __mask = 1)
{
    return __dpl_esimd_ns::gather<_T>(__input, __offsets * __size32<_T>, __base_offset * __size32<_T>, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
__dpl_esimd_ns::simd<_T, _N>
__gather(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __input, __dpl_esimd_ns::simd<::std::uint32_t, _N> __offsets,
       ::std::uint32_t __base_offset, __dpl_esimd_ns::simd_mask<_N> __mask = 1)
{
    return __dpl_esimd_ens::lsc_gather<_T>(__input, __offsets * __size32<_T> + __base_offset * __size32<_T>, __mask);
}

template <typename _T, int _N>
void
__scatter(_T* __output, __dpl_esimd_ns::simd<::std::uint32_t, _N> __offsets, __dpl_esimd_ns::simd<_T, _N> __values,
        __dpl_esimd_ns::simd_mask<_N> __mask = 1)
{
    __dpl_esimd_ns::scatter(__output, __offsets * __size32<_T>, __values, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) <= sizeof(::std::uint32_t), int> = 0>
void
__scatter(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __output, __dpl_esimd_ns::simd<::std::uint32_t, _N> __offsets,
        __dpl_esimd_ns::simd<_T, _N> __values, __dpl_esimd_ns::simd_mask<_N> __mask = 1)
{
    __dpl_esimd_ns::scatter(__output, __offsets * __size32<_T>, __values, /*global_offset*/ 0, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
void
__scatter(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __output, __dpl_esimd_ns::simd<::std::uint32_t, _N> __offsets,
        __dpl_esimd_ns::simd<_T, _N> __values, __dpl_esimd_ns::simd_mask<_N> __mask = 1)
{
    __dpl_esimd_ens::lsc_scatter<_T>(__output, __offsets * __size32<_T>, __values, __mask);
}

template <typename RT, typename _T>
inline __dpl_esimd_ns::simd<RT, 32>
__scan(__dpl_esimd_ns::simd<_T, 32> __src, const _T __init = 0)
{
    __dpl_esimd_ns::simd<RT, 32> __res;
    __res.template select<8, 4>(0) = __src.template select<8, 4>(0);
    __res[0] += __init;
    __res.template select<8, 4>(1) = __src.template select<8, 4>(1) + __res.template select<8, 4>(0);
    __res.template select<8, 4>(2) = __src.template select<8, 4>(2) + __res.template select<8, 4>(1);
    __res.template select<8, 4>(3) = __src.template select<8, 4>(3) + __res.template select<8, 4>(2);
    __res.template select<4, 1>(4) = __res.template select<4, 1>(4) + __res[3];
    __res.template select<4, 1>(8) = __res.template select<4, 1>(8) + __res[7];
    __res.template select<4, 1>(12) = __res.template select<4, 1>(12) + __res[11];
    __res.template select<4, 1>(16) = __res.template select<4, 1>(16) + __res[15];
    __res.template select<4, 1>(20) = __res.template select<4, 1>(20) + __res[19];
    __res.template select<4, 1>(24) = __res.template select<4, 1>(24) + __res[23];
    __res.template select<4, 1>(28) = __res.template select<4, 1>(28) + __res[27];
    return __res;
}

template <typename RT, typename _T>
inline __dpl_esimd_ns::simd<RT, 16>
scan(__dpl_esimd_ns::simd<_T, 16> __src, const _T __init = 0)
{
    __dpl_esimd_ns::simd<RT, 16> __res;
    __res.template select<4, 4>(0) = __src.template select<4, 4>(0);
    __res[0] += __init;
    __res.template select<4, 4>(1) = __src.template select<4, 4>(1) + __res.template select<4, 4>(0);
    __res.template select<4, 4>(2) = __src.template select<4, 4>(2) + __res.template select<4, 4>(1);
    __res.template select<4, 4>(3) = __src.template select<4, 4>(3) + __res.template select<4, 4>(2);
    __res.template select<4, 1>(4) = __res.template select<4, 1>(4) + __res[3];
    __res.template select<4, 1>(8) = __res.template select<4, 1>(8) + __res[7];
    __res.template select<4, 1>(12) = __res.template select<4, 1>(12) + __res[11];
    return __res;
}

template <typename RT, typename _T>
inline __dpl_esimd_ns::simd<RT, 64>
scan(__dpl_esimd_ns::simd<_T, 64> __src, const _T __init = 0)
{
    __dpl_esimd_ns::simd<RT, 64> __res;
    __res.template select<32, 1>(0) = scan<RT, _T>(__src.template select<32, 1>(0), __init);
    __res.template select<32, 1>(32) = scan<RT, _T>(__src.template select<32, 1>(32), __res[31]);
    return __res;
}

// get bits value (bucket) in a certain radix position
template <::std::uint16_t __radix_mask, typename _T, int _N, std::enable_if_t<::std::is_unsigned_v<_T>, int> = 0>
__dpl_esimd_ns::simd<::std::uint16_t, _N>
__get_bucket(__dpl_esimd_ns::simd<_T, _N> __value, ::std::uint32_t __radix_offset)
{
    return __dpl_esimd_ns::simd<::std::uint16_t, _N>(__value >> __radix_offset) & __radix_mask;
}

template <typename _T, bool __is_ascending, std::enable_if_t<::std::is_integral_v<_T>, int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return ::std::numeric_limits<_T>::max();
    else
        return ::std::numeric_limits<_T>::lowest();
}

// std::numeric_limits<_T>::max and std::numeric_limits<_T>::lowest cannot be used as an idenentity for
// performing radix sort of floating point numbers.
// They do not set the smallest exponent bit (i.e. the max is 7F7FFFFF for 32bit float),
// thus such an identity is not guaranteed to be put at the end of the sorted sequence after each radix sort stage,
// e.g. 00FF0000 numbers will be pushed out by 7F7FFFFF identities when sorting 16-23 bits.
template <typename _T, bool __is_ascending,
          std::enable_if_t<::std::is_floating_point_v<_T> && sizeof(_T) == sizeof(::std::uint32_t), int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return sycl::bit_cast<_T>(0x7FFF'FFFFu);
    else
        return sycl::bit_cast<_T>(0xFFFF'FFFFu);
}

template <typename _T, bool __is_ascending,
          std::enable_if_t<::std::is_floating_point_v<_T> && sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return sycl::bit_cast<_T>(0x7FFF'FFFF'FFFF'FFFFu);
    else
        return sycl::bit_cast<_T>(0xFFFF'FFFF'FFFF'FFFFu);
}

template <bool __is_ascending, int _N>
__dpl_esimd_ns::simd<bool, _N>
__order_preserving_cast(__dpl_esimd_ns::simd<bool, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return !__src;
}

template <bool __is_ascending, typename _UInt, int _N, std::enable_if_t<::std::is_unsigned_v<_UInt>, int> = 0>
__dpl_esimd_ns::simd<_UInt, _N>
__order_preserving_cast(__dpl_esimd_ns::simd<_UInt, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return ~__src; //bitwise inversion
}

template <bool __is_ascending, typename _Int, int _N,
          std::enable_if_t<::std::is_integral_v<_Int>&& ::std::is_signed_v<_Int>, int> = 0>
__dpl_esimd_ns::simd<::std::make_unsigned_t<_Int>, _N>
__order_preserving_cast(__dpl_esimd_ns::simd<_Int, _N> __src)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // __mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask =
        (__is_ascending) ? _UInt(1) << ::std::numeric_limits<_Int>::digits : ::std::numeric_limits<_UInt>::max() >> 1;
    return __src.template bit_cast_view<_UInt>() ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t), int> = 0>
__dpl_esimd_ns::simd<::std::uint32_t, _N>
__order_preserving_cast(__dpl_esimd_ns::simd<_Float, _N> __src)
{
    __dpl_esimd_ns::simd<::std::uint32_t, _N> __uint32_src = __src.template bit_cast_view<::std::uint32_t>();
    __dpl_esimd_ns::simd<::std::uint32_t, _N> __mask;
    __dpl_esimd_ns::simd_mask<_N> __sign_bit_m = (__uint32_src >> 31 == 0);
    if constexpr (__is_ascending)
    {
        __mask = __dpl_esimd_ns::merge(__dpl_esimd_ns::simd<::std::uint32_t, _N>(0x80000000u),
                                       __dpl_esimd_ns::simd<::std::uint32_t, _N>(0xFFFFFFFFu), __sign_bit_m);
    }
    else
    {
        __mask = __dpl_esimd_ns::merge(__dpl_esimd_ns::simd<::std::uint32_t, _N>(0x7FFFFFFFu),
                                       __dpl_esimd_ns::simd<::std::uint32_t, _N>(::std::uint32_t(0)), __sign_bit_m);
    }
    return __uint32_src ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t), int> = 0>
__dpl_esimd_ns::simd<::std::uint64_t, _N>
__order_preserving_cast(__dpl_esimd_ns::simd<_Float, _N> __src)
{
    __dpl_esimd_ns::simd<::std::uint64_t, _N> __uint64_src = __src.template bit_cast_view<::std::uint64_t>();
    __dpl_esimd_ns::simd<::std::uint64_t, _N> __mask;
    __dpl_esimd_ns::simd_mask<_N> __sign_bit_m = (__uint64_src >> 63 == 0);
    if constexpr (__is_ascending)
    {
        __mask = __dpl_esimd_ns::merge(__dpl_esimd_ns::simd<::std::uint64_t, _N>(0x8000000000000000u),
                                       __dpl_esimd_ns::simd<::std::uint64_t, _N>(0xFFFFFFFFFFFFFFFFu), __sign_bit_m);
    }
    else
    {
        __mask = __dpl_esimd_ns::merge(__dpl_esimd_ns::simd<::std::uint64_t, _N>(0x7FFFFFFFFFFFFFFFu),
                                       __dpl_esimd_ns::simd<::std::uint64_t, _N>(::std::uint64_t(0)), __sign_bit_m);
    }
    return __uint64_src ^ __mask;
}

template <typename _T, int _VSize, int _LANES, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(_VSize <= 4), __dpl_esimd_ns::simd<_T, _VSize * _LANES>>
__vector_load(const _T* __src, const __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    return __dpl_esimd_ens::lsc_gather<_T, _VSize, __dpl_esimd_ens::lsc_data_size::default_size, _H1, _H3, _LANES>(
        __src, __offset, __mask);
}

template <typename _T, int _VSize, int _LANES, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(_VSize > 4), __dpl_esimd_ns::simd<_T, _VSize * _LANES>>
__vector_load(const _T* __src, const __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __dpl_esimd_ns::simd<_T, _VSize * _LANES> __res;
    __res.template select<4 * _LANES, 1>(0) = __vector_load<_T, 4, _LANES, _H1, _H3>(__src, __offset, __mask);
    __res.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES) =
        __vector_load<_T, _VSize - 4, _LANES, _H1, _H3>(__src, __offset + 4 * sizeof(_T), __mask);
    return __res;
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize,
          __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline __dpl_esimd_ns::simd<_T, _VSize * _LANES>
__vector_load(const _T* __src, ::std::uint32_t __offset, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    return __vector_load<_T, _VSize, _LANES, _H1, _H3>(__src, {__offset, LaneStride * sizeof(_T)}, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize <= 4), __dpl_esimd_ns::simd<_T, _VSize * _LANES>>
__vector_load(const __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    return __dpl_esimd_ens::lsc_slm_gather<_T, _VSize, __dpl_esimd_ens::lsc_data_size::default_size, _LANES>(__offset, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize > 4), __dpl_esimd_ns::simd<_T, _VSize * _LANES>>
__vector_load(const __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __dpl_esimd_ns::simd<_T, _VSize * _LANES> __res;
    __res.template select<4 * _LANES, 1>(0) = __vector_load<_T, 4, _LANES>(__offset, __mask);
    __res.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES) =
        __vector_load<_T, _VSize - 4, _LANES>(__offset + 4 * sizeof(_T), __mask);
    return __res;
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize>
inline __dpl_esimd_ns::simd<_T, _VSize * _LANES>
__vector_load(::std::uint32_t __offset, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    return __vector_load<_T, _VSize, _LANES>({__offset, LaneStride * sizeof(_T)}, __mask);
}

template <typename TWhatEver>
struct __is_sycl_accessor : ::std::false_type
{
};

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P>
struct __is_sycl_accessor<sycl::accessor<_T, _N, _Mode, sycl::target::device, _P>> : ::std::true_type
{
};

template <typename... _Tp>
inline constexpr bool is_sycl_accessor_v = __is_sycl_accessor<_Tp...>::value;

template <typename _T, int _VSize, int _LANES, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(_VSize <= 4 && _LANES <= 32), void>
__vector_store(_T* __dest, __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __dpl_esimd_ens::lsc_scatter<_T, _VSize, __dpl_esimd_ens::lsc_data_size::default_size, _H1, _H3, _LANES>(__dest, __offset,
                                                                                                        __data, __mask);
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy,
          __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy> && _VSize <= 4 && _LANES <= 32), void>
__vector_store(_AccessorTy __acc, __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __dpl_esimd_ens::lsc_scatter<_T, _VSize, __dpl_esimd_ens::lsc_data_size::default_size, _H1, _H3, _LANES>(__acc, __offset,
                                                                                                        __data, __mask);
}

template <typename _T, int _VSize, int _LANES, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(_LANES > 32), void>
__vector_store(_T* __dest, __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, _VSize, 32>(__dest, __offset.template select<32, 1>(0), __data.template select<_VSize * 32, 1>(0),
                              __mask.template select<32, 1>(0));
    __vector_store<_T, _VSize, _LANES - 32>(__dest, __offset.template select<_LANES - 32, 1>(32),
                                      __data.template select<_VSize*(_LANES - 32), 1>(32),
                                      __mask.template select<_LANES - 32, 1>(32));
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy,
          __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy> && _LANES > 32), void>
__vector_store(_AccessorTy __acc, __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, _VSize, 32>(__acc, __offset.template select<32, 1>(0), __data.template select<_VSize * 32, 1>(0),
                              __mask.template select<32, 1>(0));
    __vector_store<_T, _VSize, _LANES - 32>(__acc, __offset.template select<_LANES - 32, 1>(32),
                                      __data.template select<_VSize*(_LANES - 32), 1>(32),
                                      __mask.template select<_LANES - 32, 1>(32));
}

template <typename _T, int _VSize, int _LANES, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(_VSize > 4 && _LANES <= 32), void>
__vector_store(_T* __dest, __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, 4, _LANES>(__dest, __offset, __data.template select<4 * _LANES, 1>(0), __mask);
    __vector_store<_T, _VSize - 4, _LANES>(__dest, __offset + 4 * sizeof(_T),
                                     __data.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES), __mask);
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy,
          __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy> && _VSize > 4 && _LANES <= 32), void>
__vector_store(_AccessorTy __acc, __dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, 4, _LANES, _AccessorTy>(__acc, __offset, __data.template select<4 * _LANES, 1>(0), __mask);
    __vector_store<_T, _VSize - 4, _LANES, _AccessorTy>(__acc, __offset + 4 * sizeof(_T),
                                                 __data.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES), __mask);
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize,
          __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline void
__vector_store(_T* __dest, ::std::uint32_t __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    // optimization needed here, hard for compiler to optimize the __offset vector calculation
    __vector_store<_T, _VSize, _LANES, _H1, _H3>(__dest, {__offset, LaneStride * sizeof(_T)}, __data, __mask);
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy, int LaneStride = _VSize,
          __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy>), void>
__vector_store(_AccessorTy __acc, ::std::uint32_t __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    // optimization needed here, hard for compiler to optimize the __offset vector calculation
    __vector_store<_T, _VSize, _LANES, _H1, _H3>(__acc, {__offset, LaneStride * sizeof(_T)}, __data, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize <= 4 && _LANES <= 32), void>
__vector_store(__dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __dpl_esimd_ens::lsc_slm_scatter<_T, _VSize>(__offset, __data, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_LANES > 32), void>
__vector_store(__dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, _VSize, 32>(__offset.template select<32, 1>(0), __data.template select<_VSize * 32, 1>(0),
                              __mask.template select<32, 1>(0));
    __vector_store<_T, _VSize, _LANES - 32>(__offset.template select<_LANES - 32, 1>(32),
                                      __data.template select<_VSize*(_LANES - 32), 1>(32),
                                      __mask.template select<_LANES - 32, 1>(32));
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize > 4 && _LANES <= 32), void>
__vector_store(__dpl_esimd_ns::simd<::std::uint32_t, _LANES> __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data,
            __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, 4, _LANES>(__offset, __data.template select<4 * _LANES, 1>(0), __mask);
    __vector_store<_T, _VSize - 4, _LANES>(__offset + 4 * sizeof(_T), __data.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES),
                                     __mask);
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize>
inline void
__vector_store(::std::uint32_t __offset, __dpl_esimd_ns::simd<_T, _VSize * _LANES> __data, __dpl_esimd_ns::simd_mask<_LANES> __mask = 1)
{
    return __vector_store<_T, _VSize, _LANES>({__offset, LaneStride * sizeof(_T)}, __data, __mask);
}

template <int _NElts>
constexpr int
__lsc_slm_block_size_rounding()
{
    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 3)
        return 2;

    if constexpr (_NElts < 4)
        return 3;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    return 64;
}

template <typename _SrcType, int _N, typename _DestType>
constexpr int
__lsc_op_block_size()
{
    return _N * sizeof(_SrcType) / sizeof(_DestType);
}

template <typename _T>
using _LscOpAlignedT = ::std::conditional_t<sizeof(_T) <= sizeof(::std::uint32_t), ::std::uint32_t, ::std::uint64_t>;

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts == __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
inline __dpl_esimd_ns::simd<_T, _N>
__block_load_slm(::std::uint32_t __slm_offset)
{
    __dpl_esimd_ns::simd<_T, _N> __res;
    __res.template bit_cast_view<_OpAlignedT>() = __dpl_esimd_ens::lsc_slm_block_load<_OpAlignedT, _NElts>(__slm_offset);
    return __res;
}

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts != __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
inline __dpl_esimd_ns::simd<_T, _N>
__block_load_slm(::std::uint32_t __slm_offset)
{
    constexpr int __block_size_rounded = __lsc_slm_block_size_rounding<_NElts>();

    __dpl_esimd_ns::simd<_T, _N> __res;
    constexpr int __block_size = __lsc_op_block_size<_OpAlignedT, __block_size_rounded, _T>();
    __res.template select<__block_size, 1>(0) = __block_load_slm<_T, __block_size>(__slm_offset);
    __res.template select<_N - __block_size, 1>(__block_size) =
        __block_load_slm<_T, _N - __block_size>(__slm_offset + __block_size * sizeof(_T));
    return __res;
}

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts == __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
void
__block_store_slm(::std::uint32_t __slm_offset, __dpl_esimd_ns::simd<_T, _N> __data)
{
    __dpl_esimd_ens::lsc_slm_block_store<_OpAlignedT, _NElts>(__slm_offset, __data.template bit_cast_view<::std::uint32_t>());
}

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts != __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
void
__block_store_slm(::std::uint32_t __slm_offset, __dpl_esimd_ns::simd<_T, _N> __data)
{
    constexpr int __block_size_rounded = __lsc_slm_block_size_rounding<_NElts>();

    constexpr int __block_size = __lsc_op_block_size<_OpAlignedT, __block_size_rounded, _T>();
    __block_store_slm<_T, __block_size>(__slm_offset, __data.template select<__block_size, 1>(0));
    __block_store_slm<_T, _N - __block_size>(__slm_offset + __block_size * sizeof(_T),
                                     __data.template select<_N - __block_size, 1>(__block_size));
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint8_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256, 512.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 12)
        return 8;

    if constexpr (_NElts < 16)
        return 12;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    if constexpr (_NElts < 128)
        return 64;

    if constexpr (_NElts < 256)
        return 128;

    if constexpr (_NElts < 512)
        return 256;

    return 512;
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint16_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 4)
        return 2;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    if constexpr (_NElts < 128)
        return 64;

    if constexpr (_NElts < 256)
        return 128;

    return 256;
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint32_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 3)
        return 2;

    if constexpr (_NElts < 4)
        return 3;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    if constexpr (_NElts < 128)
        return 64;

    return 128;
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 3)
        return 2;

    if constexpr (_NElts < 4)
        return 3;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    return 64;
}

template <typename _T, int _N, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none,
          ::std::enable_if_t<_N == __lsc_block_store_size_rounding<_T, _N>(), int> = 0>
inline void
__block_store(_T* __dst, __dpl_esimd_ns::simd<_T, _N> __data)
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for  8 bit data are          4, 8, 12, 16, 32, 64, 128, 256, 512.
    // Allowed _NElts values for 16 bit data are    2,    4, 8,     16, 32, 64, 128, 256.
    // Allowed _NElts values for 32 bit data are 1, 2, 3, 4, 8,     16, 32, 64, 128.
    // Allowed _NElts values for 64 bit data are 1, 2, 3, 4, 8,     16, 32, 64.
    __dpl_esimd_ens::lsc_block_store<::std::uint32_t, _N, __dpl_esimd_ens::lsc_data_size::default_size, _H1, _H3>(
        __dst, __data.template bit_cast_view<::std::uint32_t>(), 1);
}

template <typename _T, int _N, __dpl_esimd_ens::cache_hint _H1 = __dpl_esimd_ens::cache_hint::none,
          __dpl_esimd_ens::cache_hint _H3 = __dpl_esimd_ens::cache_hint::none,
          ::std::enable_if_t<_N != __lsc_block_store_size_rounding<_T, _N>(), int> = 0>
inline void
__block_store(_T* __dst, __dpl_esimd_ns::simd<_T, _N> __data)
{
    constexpr ::std::uint32_t __block_size = 64 * sizeof(::std::uint32_t) / sizeof(_T);

    constexpr int __block_size_rounded = __lsc_block_store_size_rounding<_T, _N>();
    static_assert(__block_size == __block_size_rounded);

    __block_store<_T, __block_size>(__dst, __data.template select<__block_size, 1>(0));
    __block_store<_T, _N - __block_size>(__dst + __block_size, __data.template select<_N - __block_size, 1>(__block_size));
}

template <typename _T, int _N>
inline std::enable_if_t<(_N > 16) && (_N % 16 == 0), __dpl_esimd_ns::simd<_T, _N>>
__create_simd(_T initial, _T step)
{
    __dpl_esimd_ns::simd<_T, _N> ret;
    ret.template select<16, 1>(0) = __dpl_esimd_ns::simd<_T, 16>(0, 1) * step + initial;
    __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
#pragma unroll
    for (int pos = 16; pos < _N; pos += 16)
    {
        ret.template select<16, 1>(pos) = ret.template select<16, 1>(0) + pos * step;
    }
    return ret;
}

} // namespace __utils
} // namespace oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H
