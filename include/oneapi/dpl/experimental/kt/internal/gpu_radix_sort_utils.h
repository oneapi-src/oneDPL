#pragma once

#include <CL/sycl.hpp>
#include <cstdint>
#include <random>

namespace oneapi::dpl::experimental::kt::gpu::__impl::__utils{

#define FORCE_INLINE __attribute__((flatten)) __attribute__((always_inline)) inline

template <typename keyT>
FORCE_INLINE uint32_t getDigit(keyT key, uint32_t lane, uint32_t digit_bits) {
    uint32_t mask = (1 << digit_bits) - 1;
    return (key >> (lane * digit_bits)) & mask;
};

//TODO: avoid code duplication: copied from esimd_radix_sort_utils.h
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


template <uint32_t MATCH_BITS>
FORCE_INLINE uint32_t matchAny(sycl::nd_item<1> &id, uint32_t value) {
    auto match = sycl::ext::oneapi::group_ballot(id.get_sub_group(), true);
#pragma unroll
    for (int i = 0; i < MATCH_BITS; i++) {
        bool vote = (value & (1 << i)) != 0;
        sycl::ext::oneapi::sub_group_mask mask =
            sycl::ext::oneapi::group_ballot(id.get_sub_group(), vote);
        if (!vote)
            match = ~mask & match;
        else
            match = mask & match;
    }
    uint32_t result = 0;
    match.extract_bits(result);
    return result;
}

void slmBarrier(sycl::nd_item<1> &id) { id.barrier(sycl::access::fence_space::local_space); }

template <uint32_t N, typename T, uint32_t L = 4>
struct Vector
{
    std::enable_if_t<N%L==0, sycl::vec<T, L> > data[N / L];
    FORCE_INLINE void load(sycl::sub_group &&sb, T *src) {
#pragma unroll
        for (uint32_t i = 0; i < N / L; i++) {
            data[i] = sb.load<L>(sycl::global_ptr<T>(&src[i * sb.get_max_local_range()[0] * L]));
        }
    }
    FORCE_INLINE T &operator[](uint32_t i) { return data[i / L][i % L]; }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl::__utils
