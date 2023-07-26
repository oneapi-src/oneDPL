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
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <ext/intel/esimd.hpp>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace oneapi::dpl::experimental::esimd::impl::utils
{

// converts sizeof(T) to 32 bits, so that it could be used in operations with 32-bit SIMD without changing the type
template <typename T>
inline constexpr ::std::uint32_t size32 = sizeof(T);

template <typename T, int N>
void
copy_from(const T* input, ::std::uint32_t base_offset, sycl::ext::intel::esimd::simd<T, N>& values)
{
    values.copy_from(input + base_offset);
}

template <typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P>
void
copy_from(sycl::accessor<T, 1, Mode, sycl::target::device, P> input, ::std::uint32_t base_offset,
          sycl::ext::intel::esimd::simd<T, N>& values)
{
    values.copy_from(input, base_offset * size32<T>);
}

template <typename T, int N>
void
copy_to(T* output, ::std::uint32_t base_offset, const sycl::ext::intel::esimd::simd<T, N>& values)
{
    values.copy_to(output + base_offset);
}

template <typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P>
void
copy_to(sycl::accessor<T, 1, Mode, sycl::target::device, P> output, ::std::uint32_t base_offset,
        const sycl::ext::intel::esimd::simd<T, N>& values)
{
    values.copy_to(output, base_offset * size32<T>);
}

template <typename T, int N>
sycl::ext::intel::esimd::simd<T, N>
gather(const T* input, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets, ::std::uint32_t base_offset,
       sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    return sycl::ext::intel::esimd::gather(input + base_offset, offsets * size32<T>, mask);
}

template <typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P, ::std::enable_if_t<sizeof(T) <= sizeof(::std::uint32_t), int> = 0>
sycl::ext::intel::esimd::simd<T, N>
gather(sycl::accessor<T, 1, Mode, sycl::target::device, P> input, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
       ::std::uint32_t base_offset, sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    return sycl::ext::intel::esimd::gather<T>(input, offsets * size32<T>, base_offset * size32<T>, mask);
}

template <typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P, ::std::enable_if_t<sizeof(T) == sizeof(::std::uint64_t), int> = 0>
sycl::ext::intel::esimd::simd<T, N>
gather(sycl::accessor<T, 1, Mode, sycl::target::device, P> input,
       sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
       ::std::uint32_t base_offset,
       sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    return __ESIMD_ENS::lsc_gather<T>(input, offsets * size32<T> + base_offset * size32<T>, mask);
}

template <typename T, int N>
void
scatter(T* output, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
        sycl::ext::intel::esimd::simd<T, N> values, sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    sycl::ext::intel::esimd::scatter(output, offsets * size32<T>, values, mask);
}

template<typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P, ::std::enable_if_t<sizeof(T) <= sizeof(::std::uint32_t), int> = 0>
void
scatter(sycl::accessor<T, 1, Mode, sycl::target::device, P> output, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
        sycl::ext::intel::esimd::simd<T, N> values, sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    sycl::ext::intel::esimd::scatter(output, offsets * size32<T>, values, /*global_offset*/ 0, mask);
}

template<typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P, ::std::enable_if_t<sizeof(T) == sizeof(::std::uint64_t), int> = 0>
void
scatter(sycl::accessor<T, 1, Mode, sycl::target::device, P> output,
        sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
        sycl::ext::intel::esimd::simd<T, N> values,
        sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    __ESIMD_ENS::lsc_scatter<T>(output, offsets * size32<T>, values, mask);
}

template <typename T, uint32_t R, uint32_t C>
class simd2d : public sycl::ext::intel::esimd::simd<T, R*C> {
    public:
        auto row(uint16_t r) {return this->template bit_cast_view<T, R, C>().row(r);}
        template <int SizeY, int StrideY, int SizeX, int StrideX>
        auto select(uint16_t OffsetY = 0, uint16_t OffsetX = 0) {
            return this->template bit_cast_view<T, R, C>().template select<SizeY, StrideY, SizeX, StrideX>(OffsetY, OffsetX);
        }
};

template <typename RT, typename T>
inline __ESIMD_NS::simd<RT, 32> scan(__ESIMD_NS::simd<T, 32> src, const T init=0) {
    __ESIMD_NS::simd<RT, 32> result;
    result.template select<8, 4>(0) = src.template select<8, 4>(0);
    result[0] += init;
    result.template select<8, 4>(1) = src.template select<8, 4>(1) + result.template select<8, 4>(0);
    result.template select<8, 4>(2) = src.template select<8, 4>(2) + result.template select<8, 4>(1);
    result.template select<8, 4>(3) = src.template select<8, 4>(3) + result.template select<8, 4>(2);
    result.template select<4, 1>(4) = result.template select<4, 1>(4) + result[3];
    result.template select<4, 1>(8) = result.template select<4, 1>(8) + result[7];
    result.template select<4, 1>(12) = result.template select<4, 1>(12) + result[11];
    result.template select<4, 1>(16) = result.template select<4, 1>(16) + result[15];
    result.template select<4, 1>(20) = result.template select<4, 1>(20) + result[19];
    result.template select<4, 1>(24) = result.template select<4, 1>(24) + result[23];
    result.template select<4, 1>(28) = result.template select<4, 1>(28) + result[27];
    return result;
}

template <typename RT, typename T>
inline __ESIMD_NS::simd<RT, 16> scan(__ESIMD_NS::simd<T, 16> src, const T init=0) {
    __ESIMD_NS::simd<RT, 16> result;
    result.template select<4, 4>(0) = src.template select<4, 4>(0);
    result[0] += init;
    result.template select<4, 4>(1) = src.template select<4, 4>(1) + result.template select<4, 4>(0);
    result.template select<4, 4>(2) = src.template select<4, 4>(2) + result.template select<4, 4>(1);
    result.template select<4, 4>(3) = src.template select<4, 4>(3) + result.template select<4, 4>(2);
    result.template select<4, 1>(4) = result.template select<4, 1>(4) + result[3];
    result.template select<4, 1>(8) = result.template select<4, 1>(8) + result[7];
    result.template select<4, 1>(12) = result.template select<4, 1>(12) + result[11];
    return result;
}

template <typename RT, typename T>
inline __ESIMD_NS::simd<RT, 64> scan(__ESIMD_NS::simd<T, 64> src, const T init=0) {
    __ESIMD_NS::simd<RT, 64> result;
    result.template select<32, 1>(0) = scan<RT, T>(src.template select<32, 1>(0), init);
    result.template select<32, 1>(32) = scan<RT, T>(src.template select<32, 1>(32), result[31]);
    return result;
}

// get bits value (bucket) in a certain radix position
template <::std::uint16_t __radix_mask, typename _T, int _N, std::enable_if_t<::std::is_unsigned_v<_T>, int> = 0>
sycl::ext::intel::esimd::simd<::std::uint16_t, _N>
__get_bucket(sycl::ext::intel::esimd::simd<_T, _N> __value, ::std::uint32_t __radix_offset)
{
    return sycl::ext::intel::esimd::simd<::std::uint16_t, _N>(__value >> __radix_offset) & __radix_mask;
}

template <typename T, bool __is_ascending, std::enable_if_t<::std::is_integral_v<T>, int> = 0>
constexpr T
__sort_identity()
{
    if constexpr (__is_ascending)
        return ::std::numeric_limits<T>::max();
    else
        return ::std::numeric_limits<T>::lowest();
}

// std::numeric_limits<T>::max and std::numeric_limits<T>::lowest cannot be used as an idenentity for
// performing radix sort of floating point numbers.
// They do not set the smallest exponent bit (i.e. the max is 7F7FFFFF for 32bit float),
// thus such an identity is not guaranteed to be put at the end of the sorted sequence after each radix sort stage,
// e.g. 00FF0000 numbers will be pushed out by 7F7FFFFF identities when sorting 16-23 bits.
template <typename T, bool __is_ascending, std::enable_if_t<::std::is_floating_point_v<T> &&
                                                            sizeof(T) == sizeof(::std::uint32_t), int> = 0>
constexpr T
__sort_identity()
{
    if constexpr (__is_ascending)
        return sycl::bit_cast<T>(0x7FFF'FFFFu);
    else
        return sycl::bit_cast<T>(0xFFFF'FFFFu);
}

template <typename T, bool __is_ascending, std::enable_if_t<::std::is_floating_point_v<T> &&
                                                            sizeof(T) == sizeof(::std::uint64_t), int> = 0>
constexpr T
__sort_identity()
{
    if constexpr (__is_ascending)
        return sycl::bit_cast<T>(0x7FFF'FFFF'FFFF'FFFFu);
    else
        return sycl::bit_cast<T>(0xFFFF'FFFF'FFFF'FFFFu);
}

template <bool __is_ascending, int _N>
sycl::ext::intel::esimd::simd<bool, _N>
__order_preserving_cast(sycl::ext::intel::esimd::simd<bool, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return !__src;
}

template <bool __is_ascending, typename _UInt, int _N, std::enable_if_t<::std::is_unsigned_v<_UInt>, int> = 0>
sycl::ext::intel::esimd::simd<_UInt, _N>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_UInt, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return ~__src; //bitwise inversion
}

template <bool __is_ascending, typename _Int, int _N,
          std::enable_if_t<::std::is_integral_v<_Int> && ::std::is_signed_v<_Int>, int> = 0>
sycl::ext::intel::esimd::simd<::std::make_unsigned_t<_Int>, _N>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_Int, _N> __src)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask = (__is_ascending) ? _UInt(1) << ::std::numeric_limits<_Int>::digits
                                              : ::std::numeric_limits<_UInt>::max() >> 1;
    return __src.template bit_cast_view<_UInt>() ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t), int> = 0>
sycl::ext::intel::esimd::simd<::std::uint32_t, _N>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_Float, _N> __src)
{
    sycl::ext::intel::esimd::simd<::std::uint32_t, _N> __uint32_src = __src.template bit_cast_view<::std::uint32_t>();
    sycl::ext::intel::esimd::simd<::std::uint32_t, _N> __mask;
    sycl::ext::intel::esimd::simd_mask<_N> __sign_bit_m = (__uint32_src >> 31 == 0);
    if constexpr (__is_ascending)
    {
        __mask = sycl::ext::intel::esimd::merge(
            sycl::ext::intel::esimd::simd<::std::uint32_t, _N>(0x80000000u),
            sycl::ext::intel::esimd::simd<::std::uint32_t, _N>(0xFFFFFFFFu), __sign_bit_m);
    }
    else
    {
        __mask = sycl::ext::intel::esimd::merge(
            sycl::ext::intel::esimd::simd<::std::uint32_t, _N>(0x7FFFFFFFu),
            sycl::ext::intel::esimd::simd<::std::uint32_t, _N>(::std::uint32_t(0)), __sign_bit_m);
    }
    return __uint32_src ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t), int> = 0>
sycl::ext::intel::esimd::simd<::std::uint64_t, _N>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_Float, _N> __src)
{
    sycl::ext::intel::esimd::simd<::std::uint64_t, _N> __uint64_src = __src.template bit_cast_view<::std::uint64_t>();
    sycl::ext::intel::esimd::simd<::std::uint64_t, _N> __mask;
    sycl::ext::intel::esimd::simd_mask<_N> __sign_bit_m = (__uint64_src >> 63 == 0);
    if constexpr (__is_ascending)
    {
        __mask = sycl::ext::intel::esimd::merge(
            sycl::ext::intel::esimd::simd<::std::uint64_t, _N>(0x8000000000000000u),
            sycl::ext::intel::esimd::simd<::std::uint64_t, _N>(0xFFFFFFFFFFFFFFFFu), __sign_bit_m);
    }
    else
    {
        __mask = sycl::ext::intel::esimd::merge(
            sycl::ext::intel::esimd::simd<::std::uint64_t, _N>(0x7FFFFFFFFFFFFFFFu),
            sycl::ext::intel::esimd::simd<::std::uint64_t, _N>(::std::uint64_t(0)), __sign_bit_m);
    }
    return __uint64_src ^ __mask;
}

template <typename T, int VSize, int LANES,
    __ESIMD_ENS::cache_hint H1=__ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3=__ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(VSize<=4), __ESIMD_NS::simd<T, VSize*LANES> > VectorLoad(const T *src, const __ESIMD_NS::simd<uint32_t, LANES> offset, __ESIMD_NS::simd_mask<LANES> mask=1) {
    return __ESIMD_ENS::lsc_gather<T, VSize, __ESIMD_ENS::lsc_data_size::default_size, H1, H3, LANES>(src, offset, mask);
}

template <typename T, int VSize, int LANES,
    __ESIMD_ENS::cache_hint H1=__ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3=__ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(VSize>4), __ESIMD_NS::simd<T, VSize*LANES> > VectorLoad(const T *src, const __ESIMD_NS::simd<uint32_t, LANES> offset, __ESIMD_NS::simd_mask<LANES> mask=1) {
    __ESIMD_NS::simd<T, VSize*LANES> result;
    result.template select<4*LANES, 1>(0) = VectorLoad<T, 4, LANES, H1, H3>(src, offset, mask);
    result.template select<(VSize-4)*LANES, 1>(4*LANES) = VectorLoad<T, VSize-4, LANES, H1, H3>(src, offset+4*sizeof(T), mask);
    return result;
}

template <typename T, int VSize, int LANES,
    int LaneStride=VSize,
    __ESIMD_ENS::cache_hint H1=__ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3=__ESIMD_ENS::cache_hint::none>
inline __ESIMD_NS::simd<T, VSize*LANES> VectorLoad(const T *src, uint32_t offset, __ESIMD_NS::simd_mask<LANES> mask=1) {
    return VectorLoad<T, VSize, LANES, H1, H3>(src, {offset, LaneStride*sizeof(T)}, mask);
}

template <typename T, int VSize, int LANES>
inline std::enable_if_t<(VSize<=4), __ESIMD_NS::simd<T, VSize*LANES> > VectorLoad(const __ESIMD_NS::simd<uint32_t, LANES> offset, __ESIMD_NS::simd_mask<LANES> mask=1) {
    return __ESIMD_ENS::lsc_slm_gather<T, VSize, __ESIMD_ENS::lsc_data_size::default_size, LANES>(offset, mask);
}

template <typename T, int VSize, int LANES>
inline std::enable_if_t<(VSize>4), __ESIMD_NS::simd<T, VSize*LANES> > VectorLoad(const __ESIMD_NS::simd<uint32_t, LANES> offset, __ESIMD_NS::simd_mask<LANES> mask=1) {
    __ESIMD_NS::simd<T, VSize*LANES> result;
    result.template select<4*LANES, 1>(0) = VectorLoad<T, 4, LANES>(offset, mask);
    result.template select<(VSize-4)*LANES, 1>(4*LANES) = VectorLoad<T, VSize-4, LANES>(offset+4*sizeof(T), mask);
    return result;
}

template <typename T, int VSize, int LANES, int LaneStride=VSize>
inline __ESIMD_NS::simd<T, VSize*LANES> VectorLoad(uint32_t offset, __ESIMD_NS::simd_mask<LANES> mask=1) {
    return VectorLoad<T, VSize, LANES>({offset, LaneStride*sizeof(T)}, mask);
}

template <typename TWhatEver>
struct is_sycl_accessor : ::std::false_type
{
};

template <typename T, int N, sycl::access_mode Mode, sycl::access::placeholder P>
struct is_sycl_accessor<sycl::accessor<T, N, Mode, sycl::target::device, P>> : ::std::true_type
{
};

template <typename... _Tp>
inline constexpr bool is_sycl_accessor_v = is_sycl_accessor<_Tp...>::value;

template <typename T, int VSize, int LANES,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(VSize <= 4 && LANES <= 32), void>
VectorStore(T* dest,
            __ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    __ESIMD_ENS::lsc_scatter<T, VSize, __ESIMD_ENS::lsc_data_size::default_size, H1, H3, LANES>(dest, offset, data, mask);
}

template <typename T, int VSize, int LANES, typename AccessorTy,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<AccessorTy> && VSize <= 4 && LANES <= 32), void>
VectorStore(AccessorTy acc,
            __ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    __ESIMD_ENS::lsc_scatter<T, VSize, __ESIMD_ENS::lsc_data_size::default_size, H1, H3, LANES>(acc, offset, data, mask);
}


template <typename T, int VSize, int LANES,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(LANES > 32), void>
VectorStore(T* dest,
            __ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    VectorStore<T, VSize, 32>(dest, offset.template select<32, 1>(0), data.template select<VSize*32, 1>(0), mask.template select<32, 1>(0));
    VectorStore<T, VSize, LANES - 32>(dest, offset.template select<LANES - 32, 1>(32),
                                      data.template select<VSize * (LANES - 32), 1>(32),
                                      mask.template select<LANES - 32, 1>(32));
}

template <typename T, int VSize, int LANES, typename AccessorTy,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<AccessorTy> && LANES > 32), void>
VectorStore(AccessorTy acc,
            __ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    VectorStore<T, VSize, 32>(acc, offset.template select<32, 1>(0), data.template select<VSize*32, 1>(0), mask.template select<32, 1>(0));
    VectorStore<T, VSize, LANES - 32>(acc, offset.template select<LANES - 32, 1>(32),
                                      data.template select<VSize * (LANES - 32), 1>(32),
                                      mask.template select<LANES - 32, 1>(32));
}


template <typename T, int VSize, int LANES,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(VSize > 4 && LANES <= 32), void>
VectorStore(T* dest,
            __ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    VectorStore<T, 4, LANES>(dest, offset, data.template select<4*LANES, 1>(0), mask);
    VectorStore<T, VSize-4, LANES>(dest, offset+4*sizeof(T), data.template select<(VSize-4)*LANES, 1>(4*LANES), mask);
}

template <typename T, int VSize, int LANES, typename AccessorTy,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<AccessorTy> && VSize > 4 && LANES <= 32), void>
VectorStore(AccessorTy acc,
            __ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    VectorStore<T, 4, LANES, AccessorTy>(acc, offset, data.template select<4 * LANES, 1>(0), mask);
    VectorStore<T, VSize - 4, LANES, AccessorTy>(acc, offset + 4 * sizeof(T),
                                                 data.template select<(VSize - 4) * LANES, 1>(4 * LANES), mask);
}


template <typename T, int VSize, int LANES,
    int LaneStride = VSize,
    __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline void
VectorStore(T* dest,
            uint32_t offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    // optimization needed here, hard for compiler to optimize the offset vector calculation
    VectorStore<T, VSize, LANES, H1, H3>(dest, {offset, LaneStride*sizeof(T)}, data, mask);
}

template <typename T, int VSize, int LANES, typename AccessorTy,
          int LaneStride = VSize,
          __ESIMD_ENS::cache_hint H1 = __ESIMD_ENS::cache_hint::none,
          __ESIMD_ENS::cache_hint H3 = __ESIMD_ENS::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<AccessorTy>), void>
VectorStore(AccessorTy acc,
            uint32_t offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    // optimization needed here, hard for compiler to optimize the offset vector calculation
    VectorStore<T, VSize, LANES, H1, H3>(acc, {offset, LaneStride*sizeof(T)}, data, mask);
}

template <typename T, int VSize, int LANES>
inline std::enable_if_t<(VSize <= 4 && LANES <= 32), void>
VectorStore(__ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    __ESIMD_ENS::lsc_slm_scatter<T, VSize>(offset, data, mask);
}

template <typename T, int VSize, int LANES>
inline std::enable_if_t<(LANES > 32), void>
VectorStore(__ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    VectorStore<T, VSize, 32>(offset.template select<32, 1>(0), data.template select<VSize*32, 1>(0), mask.template select<32, 1>(0));
    VectorStore<T, VSize, LANES - 32>(offset.template select<LANES - 32, 1>(32), data.template select<VSize * (LANES - 32), 1>(32), mask.template select<LANES-32, 1>(32));
}

template <typename T, int VSize, int LANES>
inline std::enable_if_t<(VSize > 4 && LANES <= 32), void>
VectorStore(__ESIMD_NS::simd<uint32_t, LANES> offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    VectorStore<T, 4, LANES>(offset, data.template select<4*LANES, 1>(0), mask);
    VectorStore<T, VSize-4, LANES>(offset+4*sizeof(T), data.template select<(VSize-4)*LANES, 1>(4*LANES), mask);
}

template <typename T, int VSize, int LANES, int LaneStride=VSize>
inline void
VectorStore(uint32_t offset,
            __ESIMD_NS::simd<T, VSize * LANES> data,
            __ESIMD_NS::simd_mask<LANES> mask = 1)
{
    return VectorStore<T, VSize, LANES>({offset, LaneStride*sizeof(T)}, data, mask);
}

template <typename T, int N>
inline std::enable_if_t< (N*sizeof(T)/sizeof(uint32_t)<=64), __ESIMD_NS::simd<T, N> > BlockLoad(uint32_t slm_offset)
{
    __ESIMD_NS::simd<T, N> result;
    result.template bit_cast_view<uint32_t>() = __ESIMD_ENS::lsc_slm_block_load<uint32_t, N*sizeof(T)/sizeof(uint32_t)>(slm_offset);
    return result;
}

template <typename T, int N>
inline std::enable_if_t< (N*sizeof(T)/sizeof(uint32_t)>64), __ESIMD_NS::simd<T, N> > BlockLoad(uint32_t slm_offset)
{
    __ESIMD_NS::simd<T, N> result;
    constexpr uint32_t BLOCK_SIZE = 64*sizeof(uint32_t)/sizeof(T);
    result.template select<BLOCK_SIZE, 1>(0) = BlockLoad<T, BLOCK_SIZE>(slm_offset);
    result.template select<N-BLOCK_SIZE, 1>(BLOCK_SIZE) = BlockLoad<T, N-BLOCK_SIZE>(slm_offset+BLOCK_SIZE*sizeof(T));
    return result;
}

template <typename T, int N>
inline std::enable_if_t< (N*sizeof(T)/sizeof(uint32_t)<=64), void> BlockStore(uint32_t slm_offset, __ESIMD_NS::simd<T, N> data)
{
    __ESIMD_ENS::lsc_slm_block_store<uint32_t, N*sizeof(T)/sizeof(uint32_t)>(slm_offset, data.template bit_cast_view<uint32_t>());
}

template <typename T, int N>
inline std::enable_if_t< (N*sizeof(T)/sizeof(uint32_t)>64), void> BlockStore(uint32_t slm_offset, __ESIMD_NS::simd<T, N> data)
{
    constexpr uint32_t BLOCK_SIZE = 64*sizeof(uint32_t)/sizeof(T);
    BlockStore<T, BLOCK_SIZE>(slm_offset, data.template select<BLOCK_SIZE, 1>(0));
    BlockStore<T, N-BLOCK_SIZE>(slm_offset+BLOCK_SIZE*sizeof(T), data.template select<N-BLOCK_SIZE, 1>(BLOCK_SIZE));
}

template <typename T, int N,
    __ESIMD_ENS::cache_hint H1=__ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3=__ESIMD_ENS::cache_hint::none>
inline std::enable_if_t< (N*sizeof(T)<=256), void> BlockStore(T* dst, __ESIMD_NS::simd<T, N> data)
{
    __ESIMD_ENS::lsc_block_store<uint32_t, N, __ESIMD_ENS::lsc_data_size::default_size, H1, H3>(dst, data.template bit_cast_view<uint32_t>(), 1);
}

template <typename T, int N,
    __ESIMD_ENS::cache_hint H1=__ESIMD_ENS::cache_hint::none,
    __ESIMD_ENS::cache_hint H3=__ESIMD_ENS::cache_hint::none>
inline std::enable_if_t< (N*sizeof(T)>256), void> BlockStore(T* dst, __ESIMD_NS::simd<T, N> data)
{
    constexpr uint32_t BLOCK_SIZE = 64*sizeof(uint32_t)/sizeof(T);
    BlockStore<T, BLOCK_SIZE>(dst, data.template select<BLOCK_SIZE, 1>(0));
    BlockStore<T, N-BLOCK_SIZE>(dst + BLOCK_SIZE, data.template select<N-BLOCK_SIZE, 1>(BLOCK_SIZE));
}

template <typename T, int N>
inline std::enable_if_t< (N>16) && (N % 16 == 0), __ESIMD_NS::simd<T, N> > create_simd(T initial, T step) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    simd<T, N> ret;
    ret.template select<16, 1>(0) = simd<T, 16>(0, 1) * step + initial;
    fence<fence_mask::sw_barrier>();
    #pragma unroll
    for (int pos = 16; pos<N; pos+=16) {
        ret.template select<16, 1>(pos) = ret.template select<16, 1>(0) + pos * step;
    }
    return ret;
}

}
#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H
