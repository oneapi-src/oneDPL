#ifndef _ONEDPL_esimd_radix_sort_utils_H
#define _ONEDPL_esimd_radix_sort_utils_H

#include <ext/intel/esimd.hpp>
#include "sycl_defs.h"
#include <cstdint>

namespace oneapi::dpl::experimental::esimd::impl
{
template<typename SIMD, typename Input, std::enable_if_t<std::is_pointer_v<Input>, bool> = true>
inline void
copy_from(SIMD& simd, const Input& input, uint32_t offset)
{
    simd.copy_from(input + offset);
}

template<typename SIMD, typename Input, std::enable_if_t<!std::is_pointer_v<Input>, bool> = true>
inline void
copy_from(SIMD& simd, const Input& input, uint32_t offset)
{
    simd.copy_from(input, offset);
}

template<typename SIMD, typename Output, std::enable_if_t<std::is_pointer_v<Output>, bool> = true>
inline void
copy_to(const SIMD& simd, Output& output, uint32_t offset)
{
    simd.copy_to(output + offset);
}

template<typename SIMD, typename Output, std::enable_if_t<!std::is_pointer_v<Output>, bool> = true>
inline void
copy_to(const SIMD& simd, Output& output, uint32_t offset)
{
    simd.copy_to(output, offset);
}

template <typename RT, typename T>
inline sycl::ext::intel::esimd::simd<RT, 32> scan(sycl::ext::intel::esimd::simd<T, 32> src) {
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
inline sycl::ext::intel::esimd::simd<RT, 16> scan(sycl::ext::intel::esimd::simd<T, 16> src) {
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

constexpr auto
div_up(auto a, auto b)
{
    return (a + b - 1) / b;
}

template <typename InT, typename KeyT, uint32_t RADIX_BITS, uint32_t TG_COUNT, uint32_t THREAD_PER_TG>
void
global_histogram(auto idx, size_t n, InT in, uint32_t *p_global_offset, uint32_t *p_sync_buffer) {
    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;

    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using device_addr_t = uint32_t;

    slm_init(16384);
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);
    constexpr uint32_t PROCESS_SIZE = 128;
    constexpr uint32_t addr_step = TG_COUNT * THREAD_PER_TG * PROCESS_SIZE;

    uint32_t local_tid = idx.get_local_linear_id();
    uint32_t tid = idx.get_global_linear_id();

    constexpr uint32_t SYNC_SEGMENT_COUNT = 64;
    constexpr uint32_t SYNC_SEGMENT_SIZE_DW = 128;
    if (tid < SYNC_SEGMENT_COUNT) {
        simd<uint32_t, SYNC_SEGMENT_SIZE_DW> sync_init = 0;
        sync_init.copy_to(p_sync_buffer + SYNC_SEGMENT_SIZE_DW * tid);
    }

    if ((tid - local_tid) * PROCESS_SIZE > n) {
        //no work for this tg;
        return;
    }

    // cooperative fill 0
    {
        constexpr uint32_t BUFFER_SIZE = STAGES * BINCOUNT;
        constexpr uint32_t THREAD_SIZE = BUFFER_SIZE / THREAD_PER_TG;
        slm_block_store<global_hist_t, THREAD_SIZE>(local_tid*THREAD_SIZE*sizeof(global_hist_t), 0);
    }
    barrier();

    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<global_hist_t, BINCOUNT * STAGES> state_hist_grf(0);
    bin_t MASK = BINCOUNT-1;

    device_addr_t read_addr;
    for (read_addr = tid * PROCESS_SIZE; read_addr < n; read_addr += addr_step) {
        if (read_addr+PROCESS_SIZE < n) {
            // keys.copy_from(in+read_addr);
            copy_from(keys, in, read_addr);
        }
        else
        {
            simd<uint32_t, 16> lane_id(0, 1);
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                simd_mask<16> m = (s+lane_id)<(n-read_addr);
                sycl::ext::intel::esimd::simd offset((read_addr + s + lane_id)*sizeof(KeyT));
                simd<KeyT, 16> source = lsc_gather<KeyT, 1,
                        // lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(in+read_addr+s, lane_id*sizeof(KeyT), m);
                        lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(in, offset, m);
                keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(-1), m);
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < STAGES; i++) //4*3 = 12 instr
        {
            bins = (keys >> (i * RADIX_BITS))&MASK;
            #pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s++)
            {
                state_hist_grf[i * BINCOUNT + bins[s]]++;// 256K * 4 * 1.25 = 1310720 instr for grf indirect addr
            }
        }
    }

    //atomic add to the state counter in slm
    #pragma unroll
    for (uint32_t s = 0; s < BINCOUNT * STAGES; s+=16) {
        simd<uint32_t, 16> offset(0, sizeof(global_hist_t));
        lsc_slm_atomic_update<atomic_op::add, global_hist_t, 16>(s*sizeof(global_hist_t)+offset, state_hist_grf.template select<16, 1>(s), 1);
    }

    barrier();

    {
        // bin count 256, 4 stages, 1K uint32_t, by 64 threads, happen to by 16-wide each thread. will not work for other config.
        constexpr uint32_t BUFFER_SIZE = STAGES * BINCOUNT;
        constexpr uint32_t THREAD_SIZE = BUFFER_SIZE / THREAD_PER_TG;
        simd<global_hist_t, THREAD_SIZE> group_hist = slm_block_load<global_hist_t, THREAD_SIZE>(local_tid*THREAD_SIZE*sizeof(global_hist_t));
        simd<uint32_t, THREAD_SIZE> offset(0, 4);
        lsc_atomic_update<atomic_op::add>(p_global_offset + local_tid*THREAD_SIZE, offset, group_hist, simd_mask<THREAD_SIZE>(1));
    }
}

template <typename T, uint32_t R, uint32_t C>
class simd2d:public sycl::ext::intel::esimd::simd<T, R*C> {
    public:
        auto row(uint16_t r) {return this->template bit_cast_view<T, R, C>().row(r);}
        template <int SizeY, int StrideY, int SizeX, int StrideX>
        auto select(uint16_t OffsetY = 0, uint16_t OffsetX = 0) {
            return this->template bit_cast_view<T, R, C>().template select<SizeY, StrideY, SizeX, StrideX>(OffsetY, OffsetX);
        }
};

void inline init_global_sync(uint32_t * psync, uint32_t tg_id) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    simd<uint32_t, 16> lane_id(0, 4);
    simd<uint32_t, 16> old_value = lsc_atomic_update<atomic_op::load, uint32_t, 16>(psync, lane_id, 1);
    if (tg_id == 0) {
        if (!(old_value==1).all()) {
            lsc_atomic_update<atomic_op::store, uint32_t, 16>(psync, lane_id, 1, 1);
        }
    } else {
        uint32_t try_count = 0;
        while (!(old_value==1).all()) {
            old_value = lsc_atomic_update<atomic_op::load, uint32_t, 16>(psync, lane_id, 1);
            if (try_count++ > 10240) break;
        }
    }
}

void inline global_sync(uint32_t *psync, uint32_t sync_id, uint32_t count, uint32_t gid, uint32_t tid) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    //assume initial is 1, do inc, then repeat load until count is met, then the first one atomic reduce by count to reset to 1, do not use store to 1, because second round might started.
    psync += sync_id;
    // uint32_t prev = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(psync, 0, 1)[0];
    uint32_t prev = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
    uint32_t current;
    current = -1;
    uint32_t try_count = 0;
    while (current != count+1) {
        // current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, 0, 1)[0];
        current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
        if (try_count++ > 5120) break;
    }
}

void inline global_wait(uint32_t *psync, uint32_t sync_id, uint32_t count, uint32_t gid, uint32_t tid) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    //assume initial is 1, do inc, then repeat load until count is met, then the first one atomic reduce by count to reset to 1, do not use store to 1, because second round might started.
    psync += sync_id;
    uint32_t current = -1;
    uint32_t try_count = 0;
    while (current != count) {
        // current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, 0, 1)[0];
        current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
        if (try_count++ > 5120) break;
    }
}
}
#endif // _ONEDPL_esimd_radix_sort_utils_H