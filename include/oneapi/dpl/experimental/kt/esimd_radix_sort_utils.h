#ifndef _ONEDPL_kt_esimd_radix_sort_utils_H
#define _ONEDPL_kt_esimd_radix_sort_utils_H

#include <ext/intel/esimd.hpp>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace oneapi::dpl::experimental::esimd::impl::utils
{

template <typename SIMD, typename T>
void
load_simd(SIMD& simd, T* input, ::std::uint32_t offset)
{
    simd.copy_from(input + offset);
}

template <typename SIMD, typename... Args>
void
load_simd(SIMD& simd, const sycl::accessor<Args...>& input, ::std::uint32_t offset)
{
    simd.copy_from(input, offset);
}

template <typename SIMD, typename T>
void
store_simd(const SIMD& simd, T* input, ::std::uint32_t offset)
{
    simd.copy_to(output + offset);
}

template <typename SIMD, typename... Args>
void
store_simd(const SIMD& simd, sycl::accessor<Args...>& output, ::std::uint32_t offset)
{
    simd.copy_to(output, offset);
}

template <typename T, int N>
sycl::ext::intel::esimd::simd<T, N>
gather(T* input, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets, ::std::uint32_t base_offset)
{
    return sycl::ext::intel::esimd::gather(input + base_offset, offsets);
}

template <typename T, int N, typename... Args>
sycl::ext::intel::esimd::simd<T, N>
gather(const sycl::accessor<Args...>& input, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
       ::std::uint32_t base_offset)
{
    return sycl::ext::intel::esimd::gather<T>(input, offsets, base_offset);
}

template <typename T, int N>
void
scatter(T* input, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
        sycl::ext::intel::esimd::simd<T, N> vals, sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    return sycl::ext::intel::esimd::scatter(input, offsets, vals, mask);
}

template<typename T, int N, typename... Args>
void
scatter(sycl::accessor<Args...>& input, sycl::ext::intel::esimd::simd<::std::uint32_t, N> offsets,
        sycl::ext::intel::esimd::simd<T, N> vals, sycl::ext::intel::esimd::simd_mask<N> mask = 1)
{
    sycl::ext::intel::esimd::scatter(input, offsets, vals, /*global_offset*/ 0, mask);
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
sycl::ext::intel::esimd::simd<RT, 32>
scan(sycl::ext::intel::esimd::simd<T, 32> src)
{
	sycl::ext::intel::esimd::simd<RT, 32> result;
	result.template select<8, 4>(0) = src.template select<8, 4>(0);
	result.template select<8, 4>(1) = src.template select<8, 4>(1) + src.template select<8, 4>(0);
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
sycl::ext::intel::esimd::simd<RT, 16>
scan(sycl::ext::intel::esimd::simd<T, 16> src)
{
	sycl::ext::intel::esimd::simd<RT, 16> result;
	result.template select<4, 4>(0) = src.template select<4, 4>(0);
	result.template select<4, 4>(1) = src.template select<4, 4>(1) + src.template select<4, 4>(0);
	result.template select<4, 4>(2) = src.template select<4, 4>(2) + result.template select<4, 4>(1);
	result.template select<4, 4>(3) = src.template select<4, 4>(3) + result.template select<4, 4>(2);
	result.template select<4, 1>(4) = result.template select<4, 1>(4) + result[3];
	result.template select<4, 1>(8) = result.template select<4, 1>(8) + result[7];
	result.template select<4, 1>(12) = result.template select<4, 1>(12) + result[11];
	return result;
}

// get bits value (bucket) in a certain radix position
template <::std::uint32_t __radix_mask, typename _T, int _N>
sycl::ext::intel::esimd::simd<_T, _N>
__get_bucket(sycl::ext::intel::esimd::simd<_T, _N> __value, ::std::uint32_t __radix_offset)
{
    return (__value >> __radix_offset) & sycl::ext::intel::esimd::simd<_T, _N>(__radix_mask);
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

template <bool __is_ascending, typename _UInt, int _N>
typename ::std::enable_if_t<::std::is_unsigned_v<_UInt>, sycl::ext::intel::esimd::simd<_UInt, _N>>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_UInt, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return ~__src; //bitwise inversion
}

template <bool __is_ascending, typename _Int, int _N>
typename ::std::enable_if_t<::std::is_integral_v<_Int>&& ::std::is_signed_v<_Int>,
                            sycl::ext::intel::esimd::simd<::std::make_unsigned_t<_Int>, _N>>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_Int, _N> __src)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask = (__is_ascending) ? 1 << ::std::numeric_limits<_Int>::digits : ~_UInt(0) >> 1;
    return __src.template bit_cast_view<_UInt>() ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N>
typename ::std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t),
                            sycl::ext::intel::esimd::simd<::std::uint32_t, _N>>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_Float, _N> __src)
{
    auto __uint32_src = __src.template bit_cast_view<::std::uint32_t>();
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
            sycl::ext::intel::esimd::simd<::std::uint32_t, _N>(static_cast<::std::uint32_t>(0)), __sign_bit_m);
    }
    sycl::ext::intel::esimd::simd<::std::uint32_t, _N> __res = __uint32_src ^ __mask;
    return __res;
}

template <bool __is_ascending, typename _Float, int _N>
typename ::std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t),
                            sycl::ext::intel::esimd::simd<::std::uint64_t, _N>>
__order_preserving_cast(sycl::ext::intel::esimd::simd<_Float, _N> __src)
{
    auto __uint64_src = __src.template bit_cast_view<::std::uint64_t>();
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
            sycl::ext::intel::esimd::simd<::std::uint64_t, _N>(static_cast<::std::uint64_t>(0)), __sign_bit_m);
    }
    sycl::ext::intel::esimd::simd<::std::uint64_t, _N> __res = __uint64_src ^ __mask;
    return __res;
}

}
#endif // _ONEDPL_kt_esimd_radix_sort_utils_H
