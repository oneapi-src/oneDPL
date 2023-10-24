#pragma once

#include <CL/sycl.hpp>
#include <cstdint>
#include <random>

#define FORCE_INLINE __attribute__((flatten)) __attribute__((always_inline)) inline

template <int A>
struct Int2Type {
    enum { VALUE = A };
};

FORCE_INLINE void randomFillArray(uint32_t *dst, uint32_t n) {
    for (int i = 0; i < n; i++) dst[i] = rand();
}

template <typename keyT>
FORCE_INLINE uint32_t getDigit(keyT key, uint32_t lane, uint32_t digit_bits) {
    uint32_t mask = (1 << digit_bits) - 1;
    return (key >> (lane * digit_bits)) & mask;
};

template <uint32_t DIGIT_BITS = 8>
void refRadixSortForOneDigit(
    uint32_t *array, uint32_t size, uint32_t *array_out, uint32_t PASS = 0) {
    uint32_t digit_counters[2 << DIGIT_BITS];
    for (int i = 0; i < (2 << DIGIT_BITS); i++) digit_counters[i] = 0;
    for (int i = 0; i < size; i++) {
        uint32_t digit = getDigit(array[i], PASS, DIGIT_BITS);
        digit_counters[digit]++;
    }
    for (int j = 1; j < (2 << DIGIT_BITS); j++) {
        digit_counters[j] += digit_counters[j - 1];
    }
    for (int i = size - 1; i >= 0; i--) {
        uint32_t digit = getDigit(array[i], PASS, DIGIT_BITS);
        array_out[(--digit_counters[digit])] = array[i];
    }
}

FORCE_INLINE void refExclusiveScan(
    uint32_t *histogram, uint32_t *scan, uint32_t num_bins, uint32_t num_lanes) {
    for (uint32_t i = 0; i < num_lanes; i++) {
        scan[i * num_bins] = 0;
        for (uint32_t j = 1; j < num_bins; j++) {
            scan[i * num_bins + j] = scan[i * num_bins + j - 1] + histogram[i * num_bins + j - 1];
        }
    }
}

// Print an array
void printArray(uint32_t array[], int size, int increase = 1) {
    for (int i = 0; i < size; i += increase) {
        printf("%u  ", array[i]);
    }
    printf("\n");
}

template <uint32_t DIGIT_BITS = 8>
void refExclusiveSum(uint32_t *array, uint32_t size, uint32_t *array_out, uint32_t PASS = 0) {
    uint32_t radix_digits = 1 << DIGIT_BITS, digit_counters[radix_digits];
    memset(digit_counters, 0, sizeof(digit_counters));
    for (int i = 0; i < size; i++) {
        uint32_t digit = getDigit(array[i], PASS, DIGIT_BITS);
        digit_counters[digit]++;
    }
    array_out[0] = 0;
    for (uint32_t j = 1; j < radix_digits; j++) {
        array_out[j] = array_out[j - 1] + digit_counters[j - 1];
    }
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

template <uint32_t MATCH_BITS>
inline uint32_t matchAnyASM(uint32_t value, Int2Type<16>) {
    uint32_t result;
#ifdef __SYCL_DEVICE_ONLY__
    asm("mov (M1, 16) %[result](0,0)<1> %%sr0(0,2)<0;1,0>" : [result] "+rw"(result));
#pragma unroll
    for (int i = 0; i < MATCH_BITS; i++) {
        asm("{\n"
            ".decl t1 v_type=G type=ud num_elts=16 align=wordx32\n"
            ".decl t3 v_type=G type=ud num_elts=1 align=word\n"
            ".decl t4 v_type=G type=ud num_elts=16 align=wordx32\n"
            ".decl p1 v_type=P num_elts=16\n"
            "and (M1, 16) t1(0,0)<1> %[value](0,0)<1;1,0> %[bin_digit]\n"
            "cmp.ne (M1, 16) p1 t1(0,0)<1;1,0> 0x0:d\n"
            "mov (M1, 1) t3(0,0)<1> p1\n"
            "(p1) sel (M1, 16) t4(0,0)<1> 0:d -1:d\n"
            "bfn.x28 (M1, 16) %[result](0,0)<1> %[result](0,0)<1;1,0> t3(0,0)<0;1,0> "
            "t4(0,0)<1;1,0>\n"
            "}\n"
            : [result] "+rw"(result)
            : [value] "rw"(value), [bin_digit] "i"(1 << i));
    }
#endif
    return result;
}

template <uint32_t MATCH_BITS>
inline uint32_t matchAnyASM(uint32_t value, Int2Type<32>) {
    uint32_t result;
#ifdef __SYCL_DEVICE_ONLY__
    asm("mov (M1, 32) %[result](0,0)<1> %%sr0(0,2)<0;1,0>" : [result] "+rw"(result));
#pragma unroll
    for (int i = 0; i < MATCH_BITS; i++) {
        asm("{\n"
            ".decl t1 v_type=G type=ud num_elts=32 align=wordx64\n"
            ".decl t3 v_type=G type=ud num_elts=1 align=dword\n"
            ".decl t4 v_type=G type=ud num_elts=32 align=wordx64\n"
            ".decl p1 v_type=P num_elts=32\n"
            "and (M1, 32) t1(0,0)<1> %[value](0,0)<1;1,0> %[bin_digit]\n"
            "cmp.ne (M1, 32) p1 t1(0,0)<1;1,0> 0x0:d\n"
            "mov (M1, 1) t3(0,0)<1> p1\n"
            "(p1) sel (M1, 32) t4(0,0)<1> 0:d -1:d\n"
            "bfn.x28 (M1, 32) %[result](0,0)<1> %[result](0,0)<1;1,0> t3(0,0)<0;1,0> "
            "t4(0,0)<1;1,0>\n"
            "}\n"
            : [result] "+rw"(result)
            : [value] "rw"(value), [bin_digit] "i"(1 << i));
    }
#endif
    return result;
}

void slmBarrier(sycl::nd_item<1> &id) { id.barrier(sycl::access::fence_space::local_space); }

enum LSC_STCC {
    LSC_STCC_DEFAULT,
    LSC_STCC_L1UC_L3UC,  // Override to L1 uncached and L3 uncached
    LSC_STCC_L1UC_L3WB,  // Override to L1 uncached and L3 written back
    LSC_STCC_L1WT_L3UC,  // Override to L1 written through and L3 uncached
    LSC_STCC_L1WT_L3WB,  // Override to L1 written through and L3 written back
    LSC_STCC_L1S_L3UC,   // Override to L1 streaming and L3 uncached
    LSC_STCC_L1S_L3WB,   // Override to L1 streaming and L3 written back
    LSC_STCC_L1WB_L3WB,  // Override to L1 written through and L3 written back
};

SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint(
    void *base, int immElemOff, sycl::uint val, enum LSC_STCC cacheOpt);

template <uint32_t WARP_THREADS>
inline void lscStrideLoad4(uint32_t *keys, uint32_t *src) {
#ifdef __SYCL_DEVICE_ONLY__
    asm("{\n"
        ".decl vec v_type=G type=ud num_elts=64\n"
        "lsc_load.ugm (M1, 1) vec:d32x64t flat[%[ptr]]:a64\n"
        "mov (M1, 16) %[v0](0,0)<1> vec(0,0)<1;1,0>\n"
        "mov (M1, 16) %[v1](0,0)<1> vec(1,0)<1;1,0>\n"
        "mov (M1, 16) %[v2](0,0)<1> vec(2,0)<1;1,0>\n"
        "mov (M1, 16) %[v3](0,0)<1> vec(3,0)<1;1,0>\n"
        "}\n"
        : [v0] "+rw"(keys[0]), [v1] "+rw"(keys[1]), [v2] "+rw"(keys[2]), [v3] "+rw"(keys[3])
        : [ptr] "rw"(&src[0]));
#endif
}

template <uint32_t N, typename T, uint32_t L = 4>
struct Vector
{
    std::enable_if_t<N%L==0, sycl::vec<T, L> > data[N / L];
    FORCE_INLINE void load(sycl::ext::oneapi::sub_group &&sb, T *src) {
#pragma unroll
        for (uint32_t i = 0; i < N / L; i++) {
            data[i] = sb.load<L>(sycl::global_ptr<T>(&src[i * sb.get_max_local_range()[0] * L]));
        }
    }
    FORCE_INLINE T &operator[](uint32_t i) { return data[i / L][i % L]; }
};
