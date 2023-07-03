#include <oneapi/dpl/experimental/kernel_templates>

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace utils_sycl
{
// Bitwise type casting, same as C++20 std::bit_cast
template <typename _Dst, typename _Src>
::std::enable_if_t<
    sizeof(_Dst) == sizeof(_Src) && ::std::is_trivially_copyable_v<_Dst> && ::std::is_trivially_copyable_v<_Src>, _Dst>
__dpl_bit_cast(const _Src& __src) noexcept
{
    return sycl::bit_cast<_Dst>(__src);

}

template <bool __is_ascending>
bool
__order_preserving_cast(bool __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return !__val;
}

template <bool __is_ascending, typename _UInt, std::enable_if_t<::std::is_unsigned_v<_UInt>, int> = 0>
_UInt
__order_preserving_cast(_UInt __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return ~__val; //bitwise inversion
}

template <bool __is_ascending, typename _Int,
          std::enable_if_t<::std::is_integral_v<_Int> && ::std::is_signed_v<_Int>, int> = 0>
::std::make_unsigned_t<_Int>
__order_preserving_cast(_Int __val)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask =
        (__is_ascending) ? _UInt(1) << ::std::numeric_limits<_Int>::digits : ::std::numeric_limits<_UInt>::max() >> 1;
    return __val ^ __mask;
}

template <bool __is_ascending, typename _Float,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t), int> = 0>
::std::uint32_t
__order_preserving_cast(_Float __val)
{
    ::std::uint32_t __uint32_val = __dpl_bit_cast<::std::uint32_t>(__val);
    ::std::uint32_t __mask;
    // __uint32_val >> 31 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint32_val >> 31 == 0) ? 0x80000000u : 0xFFFFFFFFu;
    else
        __mask = (__uint32_val >> 31 == 0) ? 0x7FFFFFFFu : ::std::uint32_t(0);
    return __uint32_val ^ __mask;
}

template <bool __is_ascending, typename _Float,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t), int> = 0>
::std::uint64_t
__order_preserving_cast(_Float __val)
{
    ::std::uint64_t __uint64_val = __dpl_bit_cast<::std::uint64_t>(__val);
    ::std::uint64_t __mask;
    // __uint64_val >> 63 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint64_val >> 63 == 0) ? 0x8000000000000000u : 0xFFFFFFFFFFFFFFFFu;
    else
        __mask = (__uint64_val >> 63 == 0) ? 0x7FFFFFFFFFFFFFFFu : ::std::uint64_t(0);
    return __uint64_val ^ __mask;
}

// get bits value (bucket) in a certain radix position
template <::std::uint32_t __radix_mask, typename _T>
::std::uint32_t
__get_bucket(_T __value, ::std::uint32_t __radix_offset)
{
    return (__value >> __radix_offset) & _T(__radix_mask);
}
} // utils_sycl

template <typename T>
typename ::std::enable_if_t<std::is_arithmetic_v<T>, void>
generate_data(T* input, std::size_t size)
{
    std::default_random_engine gen{42};
    std::size_t unique_threshold = 75 * size / 100;
    if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
        std::generate(input, input + unique_threshold, [&]{ return dist(gen); });
    }
    else
    {
        std::uniform_real_distribution<T> dist_real(std::numeric_limits<T>::min(), log2(1e12));
        std::uniform_int_distribution<int> dist_binary(0, 1);
        auto randomly_signed_real = [&dist_real, &dist_binary, &gen](){
            auto v = exp2(dist_real(gen));
            return dist_binary(gen) == 0 ? v: -v;
        };
        std::generate(input, input + unique_threshold, [&]{ return randomly_signed_real(); });
    }
    for(uint32_t i = 0, j = unique_threshold; j < size; ++i, ++j)
    {
        input[j] = input[i];
    }
}

template<typename T, uint32_t RADIX_BITS, bool isAscending>
void test(uint32_t n)
{
    using global_hist_t = uint32_t;

    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t HW_TG_COUNT = 64;
    constexpr uint32_t THREAD_PER_TG = 64;
    constexpr uint32_t NBITS =  sizeof(T) * 8;
    constexpr uint32_t STAGES = (NBITS + RADIX_BITS - 1) / RADIX_BITS; // ceiling division
    constexpr uint32_t GLOBAL_OFFSET_SIZE = BINCOUNT * STAGES;

    sycl::queue q{};
    auto input = sycl::malloc_shared<T>(n, q);
    auto ref = sycl::malloc_host<T>(n, q);
    auto hist = sycl::malloc_shared<uint32_t>(GLOBAL_OFFSET_SIZE, q);
    auto hist_ref = sycl::malloc_host<uint32_t>(GLOBAL_OFFSET_SIZE, q);

    generate_data(input, n);
    std::copy(input, input + n, ref);

    // get histogram on device
    sycl::event event_chain = q.memset(hist, 0, GLOBAL_OFFSET_SIZE * sizeof(global_hist_t));
    sycl::nd_range<1> range{HW_TG_COUNT * THREAD_PER_TG, THREAD_PER_TG};
    event_chain = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<1> nd_item) [[intel::sycl_explicit_simd]] {
            oneapi::dpl::experimental::esimd::impl::global_histogram<T, decltype(input), RADIX_BITS, STAGES,
                                                                     HW_TG_COUNT, THREAD_PER_TG, isAscending>(
                nd_item, n, input, hist);
        });
    });
    event_chain.wait();

    // get histogram on host
    std::fill(hist_ref, hist_ref + GLOBAL_OFFSET_SIZE, 0);
    for(uint32_t i = 0; i < n; ++i)
    {
        auto ordered_key = utils_sycl::__order_preserving_cast<isAscending>(ref[i]);
        for(uint32_t stage = 0; stage < STAGES; ++stage)
        {
            constexpr uint32_t MASK = BINCOUNT - 1;
            uint32_t bin = utils_sycl::__get_bucket<MASK>(ordered_key, stage * RADIX_BITS);
            ++hist_ref[stage * BINCOUNT + bin];
        }
    }

    // verify
    uint32_t err_count = 0;
    for(uint32_t i = 0; i < GLOBAL_OFFSET_SIZE; ++i)
    {
        // checking the last bin in global histogram is not necessary: it is not used in radix sort
        if(i % BINCOUNT != BINCOUNT - 1)
        {
            if(hist[i] != hist_ref[i])
            {
                ++err_count;
                if(err_count <= 3)
                {
                    std::cout << "\texpected: " << hist_ref[i] << ", got: " << hist[i] << ", at: " << i << std::endl;
                }
            }
        }
    }
    std::cout << "\terror count: " << err_count << std::endl;;

    sycl::free(input, q);
    sycl::free(ref, q);
    sycl::free(hist, q);
    sycl::free(hist_ref, q);
}

int main()
{
    const std::vector<std::size_t> sizes = {
        6, 16, 43, 256, 316, 2048, 5072, 8192, 14001, 1<<14,
        (1<<14)+1, 50000, 67543, 100'000, 1<<17, 179'581, 250'000, 1<<18,
        (1<<18)+1, 500'000, 888'235, 1'000'000, 1<<20, 10'000'000
    };

    for(auto size: sizes)
    {
        // 64-bit, ascending
        std::cout << "test<uint64_t, 8, true>(" << size << ")" << std::endl;
        test<uint64_t, 8, true>(size);
        std::cout << "test<int64_t, 8, true>(" << size << ")" << std::endl;
        test<int64_t, 8, true>(size);
        std::cout << "test<double, 8, true>(" << size << ")" << std::endl;
        test<double, 8, true>(size);

        // 64-bit, descending
        std::cout << "test<uint64_t, 8, false>(" << size << ")" << std::endl;
        test<uint64_t, 8, false>(size);
        std::cout << "test<int64_t, 8, false>(" << size << ")" << std::endl;
        test<int64_t, 8, false>(size);
        std::cout << "test<double, 8, false>(" << size << ")" << std::endl;
        test<double, 8, false>(size);

        // 32-bit ascending
        std::cout << "test<int, 8, true>(" << size << ")" << std::endl;
        test<int, 8, true>(size);
        std::cout << "test<uint32_t, 8, true>(" << size << ")" << std::endl;
        test<uint32_t, 8, true>(size);
        std::cout << "test<float, 8, true>(" << size << ")" << std::endl;
        test<float, 8, true>(size);

        // 32-bit descending
        std::cout << "test<int, 8, false>(" << size << ")" << std::endl;
        test<int, 8, false>(size);
        std::cout << "test<uint32_t, 8, false>(" << size << ")" << std::endl;
        test<uint32_t, 8, false>(size);
        std::cout << "test<float, 8, false>(" << size << ")" << std::endl;
        test<float, 8, false>(size);

        // 16-bit ascending
        std::cout << "test<int16_t, 8, true>(" << size << ")" << std::endl;
        test<int16_t, 8, true>(size);
        std::cout << "test<uint16_t, 8, true>(" << size << ")" << std::endl;
        test<uint16_t, 8, true>(size);

        // 16-bit descending
        std::cout << "test<int16_t, 8, false>(" << size << ")" << std::endl;
        test<int16_t, 8, false>(size);
        std::cout << "test<uint16_t, 8, false>(" << size << ")" << std::endl;
        test<uint16_t, 8, false>(size);

        // 8-bit ascending
        std::cout << "test<char, 8, true>(" << size << ")" << std::endl;
        test<char, 8, true>(size);
        std::cout << "test<int4_t, 8, true>(" << size << ")" << std::endl;
        test<int8_t, 8, true>(size);
        std::cout << "test<uint4_t, 8, true>(" << size << ")" << std::endl;
        test<uint8_t, 8, true>(size);

        // 8-bit descending
        std::cout << "test<char, 8, false>(" << size << ")" << std::endl;
        test<char, 8, false>(size);
        std::cout << "test<int4_t, 8, false>(" << size << ")" << std::endl;
        test<int8_t, 8, false>(size);
        std::cout << "test<uint4_t, 8, false>(" << size << ")" << std::endl;
        test<uint8_t, 8, false>(size);
    }
    std::cout << "done" << std::endl;
}