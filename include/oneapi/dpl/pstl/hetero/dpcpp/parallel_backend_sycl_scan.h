// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_parallel_backend_sycl_scan_H
#define _ONEDPL_parallel_backend_sycl_scan_H

#include <cstdint>
#include <sycl/sycl.hpp>

namespace oneapi::dpl::experimental::kt
{

inline namespace igpu {

constexpr ::std::size_t SUBGROUP_SIZE = 32;

template <typename Type, typename UseAtomic64, template <typename, typename> typename LookbackScanMemory,
          typename TileId>
struct ScanMemoryManager
{
    using _TileIdT = typename TileId::_TileIdT;
    using _LookbackScanMemory = LookbackScanMemory<Type, UseAtomic64>;
    using _FlagT = typename _LookbackScanMemory::_FlagT;

    ScanMemoryManager(sycl::queue q) : q{q} {};

    ::std::uint8_t*
    scan_memory_ptr() noexcept
    {
        return scan_memory_begin;
    };

    _TileIdT*
    tile_id_ptr() noexcept
    {
        return tile_id_begin;
    };

    void
    allocate(::std::size_t num_wgs)
    {
        ::std::size_t scan_memory_size = _LookbackScanMemory::get_memory_size(num_wgs);
        constexpr ::std::size_t padded_tileid_size = TileId::get_padded_memory_size();
        constexpr ::std::size_t tileid_size = TileId::get_memory_size();

        auto mem_size_bytes = scan_memory_size + padded_tileid_size;

        scratch = sycl::malloc_device<::std::uint8_t>(mem_size_bytes, q);

        scan_memory_begin = scratch;

        void* base_tileid_ptr = reinterpret_cast<void*>(scan_memory_begin + scan_memory_size);
        size_t remainder = mem_size_bytes - scan_memory_size;

        tile_id_begin = reinterpret_cast<_TileIdT*>(
            ::std::align(::std::alignment_of_v<_TileIdT>, tileid_size, base_tileid_ptr, remainder));
    }

    sycl::event
    async_free(sycl::event dependency)
    {
        return q.submit(
            [e = dependency, ptr = scratch, q_ = q](sycl::handler& hdl)
            {
                hdl.depends_on(e);
                hdl.host_task([=]() { sycl::free(ptr, q_); });
            });
    }

    void free()
    {
        sycl::free(scratch, q);
    }

  private:
    ::std::uint8_t* scratch = nullptr;
    ::std::uint8_t* scan_memory_begin = nullptr;
    _TileIdT* tile_id_begin = nullptr;

    sycl::queue q;
};

template <typename _T, typename UseAtomic64>
struct LookbackScanMemory;

template <typename _T>
struct LookbackScanMemory<_T, /* UseAtomic64=*/::std::false_type>
{
    using _FlagT = ::std::uint32_t;
    using _AtomicFlagRefT = sycl::atomic_ref<_FlagT, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

    static constexpr _FlagT NOT_READY = 0;
    static constexpr _FlagT PARTIAL_MASK = 1;
    static constexpr _FlagT FULL_MASK = 2;
    static constexpr _FlagT OUT_OF_BOUNDS = 4;

    static constexpr ::std::size_t padding = SUBGROUP_SIZE;

    // LookbackScanMemory: [Partial Value, ..., Full Value, ..., Flag, ...]
    // Each section has num_wgs + padding elements
    LookbackScanMemory(::std::uint8_t* scan_memory_begin, ::std::size_t num_wgs)
        : num_elements(get_num_elements(num_wgs)), tile_values_begin(reinterpret_cast<_T*>(scan_memory_begin)),
          flags_begin(get_flags_begin(scan_memory_begin, num_wgs))
    {
    }

    void
    set_partial(::std::size_t tile_id, _T val)
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        tile_values_begin[tile_id + padding] = val;
        atomic_flag.store(PARTIAL_MASK);
    }

    void
    set_full(::std::size_t tile_id, _T val)
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        tile_values_begin[tile_id + padding + num_elements] = val;
        atomic_flag.store(FULL_MASK);
    }

    _AtomicFlagRefT
    get_flag(::std::size_t tile_id) const
    {
        return _AtomicFlagRefT(*(flags_begin + tile_id + padding));
    }

    _T
    get_value(::std::size_t tile_id, _FlagT flag) const
    {
        // full_value and partial_value are num_elements apart
        return *(tile_values_begin + tile_id + padding + num_elements * is_full(flag));
    }

    static ::std::size_t
    get_tile_values_bytes(::std::size_t num_elements)
    {
        return (2 * num_elements) * sizeof(_T);
    }

    static ::std::size_t
    get_flag_bytes(::std::size_t num_elements)
    {
        return num_elements * sizeof(_FlagT);
    }

    static ::std::size_t
    get_padded_flag_bytes(::std::size_t num_elements)
    {
        // sizeof(_FlagT) extra bytes for possible intenal alignment
        return get_flag_bytes(num_elements) + sizeof(_FlagT);
    }

    static _FlagT*
    get_flags_begin(::std::uint8_t* scan_memory_begin, ::std::size_t num_wgs)
    {
        // Aligned flags
        ::std::size_t num_elements = get_num_elements(num_wgs);
        ::std::size_t tile_values_bytes = get_tile_values_bytes(num_elements);
        void* base_flags = reinterpret_cast<void*>(scan_memory_begin + tile_values_bytes);
        auto remainder = get_padded_flag_bytes(num_elements); // scan_memory_bytes - tile_values_bytes
        return reinterpret_cast<_FlagT*>(
            ::std::align(::std::alignment_of_v<_FlagT>, get_flag_bytes(num_elements), base_flags, remainder));
    }

    static ::std::size_t
    get_memory_size(::std::size_t num_wgs)
    {
        ::std::size_t num_elements = get_num_elements(num_wgs);
        // sizeof(_T) extra bytes are not needed because LookbackScanMemory is going at the beginning of the scratch
        ::std::size_t tile_values_bytes = get_tile_values_bytes(num_elements);
        // Padding to provide room for aligment
        ::std::size_t flag_bytes = get_padded_flag_bytes(num_elements);

        return tile_values_bytes + flag_bytes;
    }

    static ::std::size_t
    get_num_elements(::std::size_t num_wgs)
    {
        return padding + num_wgs;
    }

    static bool
    is_ready(_FlagT flag)
    {
        return flag != NOT_READY;
    }

    static bool
    is_full(_FlagT flag)
    {
        return flag == FULL_MASK;
    }

    static bool
    is_out_of_bounds(_FlagT flag)
    {
        return flag == OUT_OF_BOUNDS;
    }

  private:
    ::std::size_t num_elements;
    _FlagT* flags_begin;
    _T* tile_values_begin;
};

template <typename _T>
struct LookbackScanMemory<_T, /* UseAtomic64=*/::std::true_type>
{
    using _FlagT = ::std::uint64_t;
    using _AtomicFlagRefT = sycl::atomic_ref<_FlagT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

    // Each flag is divided in 2 32bit values
    // 32..63 status bits
    // 00..31 value bits
    // Example: status = full scanned value, int value = 15:
    // 1000 0000 0000 0000 0000 0000 0000 0000 | 0000 0000 0000 0000 0000 0000 0000 1111

    // Status values:
    // 00xxxx - not computed
    // 01xxxx - partial
    // 10xxxx - full
    // 110000 - out of bounds

    static constexpr _FlagT NOT_READY = 0;
    static constexpr _FlagT PARTIAL_MASK = 1l << (sizeof(_FlagT) * 8 - 2);
    static constexpr _FlagT FULL_MASK = 1l << (sizeof(_FlagT) * 8 - 1);
    static constexpr _FlagT OUT_OF_BOUNDS = PARTIAL_MASK | FULL_MASK;

    static constexpr _FlagT VALUE_MASK = (1l << sizeof(::std::uint32_t) * 8) - 1; // 32 bit mask to store value

    static constexpr ::std::size_t padding = SUBGROUP_SIZE;

    LookbackScanMemory(::std::uint8_t* scan_memory_begin, ::std::size_t num_wgs)
        : num_elements(get_num_elements(num_wgs)), flags_begin(get_flags_begin(scan_memory_begin, num_wgs))
    {
    }

    void
    set_partial(::std::size_t tile_id, _T val)
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        atomic_flag.store(PARTIAL_MASK | static_cast<::std::uint32_t>(val));
    }

    void
    set_full(::std::size_t tile_id, _T val)
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        atomic_flag.store(FULL_MASK | static_cast<::std::uint32_t>(val));
    }

    _AtomicFlagRefT
    get_flag(::std::size_t tile_id) const
    {
        return _AtomicFlagRefT(*(flags_begin + tile_id + padding));
    }

    _T
    get_value(::std::size_t, _FlagT flag) const
    {
        return static_cast<_T>(flag & VALUE_MASK);
    }

    static _FlagT*
    get_flags_begin(::std::uint8_t* scan_memory_begin, ::std::size_t)
    {
        return reinterpret_cast<_FlagT*>(scan_memory_begin);
    }

    static ::std::size_t
    get_memory_size(::std::size_t num_wgs)
    {
        ::std::size_t num_elements = get_num_elements(num_wgs);
        return num_elements * sizeof(_FlagT);
    }

    static ::std::size_t
    get_num_elements(::std::size_t num_wgs)
    {
        return padding + num_wgs;
    }

    static bool
    is_ready(_FlagT flag)
    {
        // flag & OUT_OF_BOUNDS != NOT_READY means it has either partial or full value, or is out of bounds
        return (flag & OUT_OF_BOUNDS) != NOT_READY;
    }

    static bool
    is_full(_FlagT flag)
    {
        return (flag & OUT_OF_BOUNDS) == FULL_MASK;
    }

    static bool
    is_out_of_bounds(_FlagT flag)
    {
        return (flag & OUT_OF_BOUNDS) == OUT_OF_BOUNDS;
    }

  private:
    ::std::size_t num_elements;
    _FlagT* flags_begin;
};

struct TileId
{
    using _TileIdT = ::std::uint32_t;
    using _AtomicTileRefT = sycl::atomic_ref<_TileIdT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

    TileId(_TileIdT* tileid_memory) : tile_counter(*(tileid_memory)) {}

    constexpr static ::std::size_t
    get_padded_memory_size()
    {
        // extra sizeof(_TileIdT) for possible aligment issues
        return sizeof(_TileIdT) + sizeof(_TileIdT);
    }

    constexpr static ::std::size_t
    get_memory_size()
    {
        // extra sizeof(_TileIdT) for possible aligment issues
        return sizeof(_TileIdT);
    }

    _TileIdT
    fetch_inc()
    {
        return tile_counter.fetch_add(1);
    }

    _AtomicTileRefT tile_counter;
};

struct cooperative_lookback
{

    template <typename _T, typename _Subgroup, typename BinOp,
              template <typename, typename> typename LookbackScanMemory, typename UseAtomic64>
    _T
    operator()(std::uint32_t tile_id, const _Subgroup& subgroup, BinOp bin_op,
               LookbackScanMemory<_T, UseAtomic64> memory)
    {
        using _LookbackScanMemory = LookbackScanMemory<_T, UseAtomic64>;
        using FlagT = typename _LookbackScanMemory::_FlagT;

        _T sum = 0;
        constexpr int offset = -1;
        int local_id = subgroup.get_local_id();

        for (int tile = static_cast<int>(tile_id) + offset; tile >= 0; tile -= SUBGROUP_SIZE)
        {
            auto atomic_flag = memory.get_flag(tile - local_id); //
            FlagT flag;
            do
            {
                flag = atomic_flag.load();
            } while (!sycl::all_of_group(subgroup, _LookbackScanMemory::is_ready(flag) ||
                                                       (tile - local_id < 0))); // Loop till all ready

            bool is_full = _LookbackScanMemory::is_full(flag);
            auto is_full_ballot = sycl::ext::oneapi::group_ballot(subgroup, is_full);
            auto lowest_item_with_full = is_full_ballot.find_low();

            // TODO: Use identity_fn for out of bounds values
            _T contribution = local_id <= lowest_item_with_full && (tile - local_id >= 0)
                                  ? memory.get_value(tile - local_id, flag)
                                  : _T{0};

            // Sum all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            sum = bin_op(sum, contribution);
            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (is_full_ballot.any())
                break;

        }
        sum = sycl::reduce_over_group(subgroup, sum, bin_op);

        return sum;
    }
};

template <typename _KernelParam, typename _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp>
void
single_pass_scan_impl_single_wg(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;

    static_assert(std::is_same_v<_Inclusive, ::std::true_type>, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;
    // Avoid non_uniform n by padding up to a multiple of wgsize
    constexpr ::std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    constexpr ::std::size_t num_workitems = wgsize;

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto tile_vals = sycl::local_accessor<_Type, 1>(sycl::range<1>{elems_in_tile}, hdl);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for(
            sycl::nd_range<1>(num_workitems, wgsize), [=
        ](const sycl::nd_item<1>& item) [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
                auto group = item.get_group();
                ::std::uint32_t local_id = item.get_local_id(0);
                constexpr ::std::uint32_t stride = wgsize;
                auto subgroup = item.get_sub_group();

                constexpr std::uint32_t tile_id = 0;
                constexpr std::uint32_t wg_begin = 0;
                constexpr std::uint32_t wg_end = elems_in_tile;

                std::uint32_t wg_local_memory_size = elems_in_tile;

                auto out_begin = __out_rng.begin();
                _Type carry = 0;

                // Global load into local
                if (wg_end > n)
                    wg_local_memory_size = n;

                //TODO: assumes default ctor produces identity w.r.t. __binary_op
                // _Type my_reducer{};
                if (wg_end <= n)
                {
#pragma unroll
                    for (std::uint32_t step = 0; step < elems_per_workitem; ++step)
                    {
                        ::std::uint32_t i = stride * step;
                        _Type in_val = __in_rng[i + local_id];
                        // my_reducer = __binary_op(my_reducer, in_val);
                        _Type out = sycl::inclusive_scan_over_group(group, in_val, __binary_op, carry);
                        out_begin[i + local_id] = out;
                        carry = group_broadcast(group, out, stride - 1);
                    }
                }
                else
                {
#pragma unroll
                    for (std::uint32_t step = 0; step < elems_per_workitem; ++step)
                    {
                        ::std::uint32_t i = stride * step;
                        _Type in_val;

                        if (i + local_id < n)
                        {
                            in_val = __in_rng[i + local_id];
                            // my_reducer = __binary_op(my_reducer, in_val);
                        }
                        _Type out = sycl::inclusive_scan_over_group(group, in_val, __binary_op, carry);
                        if (i + local_id < n)
                        {
                            out_begin[i + local_id] = out;
                        }
                        carry = group_broadcast(group, out, stride - 1);
                    }
                }
            });
    });

    event.wait();
}

template <typename _KernelParam, typename _Inclusive, typename _UseAtomic64, typename _UseDynamicTileID,
          typename _InRange, typename _OutRange, typename _BinaryOp>
void
single_pass_scan_impl(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;
    using _TileIdT = TileId::_TileIdT;
    using _LookbackScanMemory = LookbackScanMemory<_Type, _UseAtomic64>;
    using _FlagT = typename _LookbackScanMemory::_FlagT;

    static_assert(std::is_same_v<_Inclusive, ::std::true_type>, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;
    // Avoid non_uniform n by padding up to a multiple of wgsize
    constexpr ::std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_workitems = num_wgs * wgsize;

    ScanMemoryManager<_Type, _UseAtomic64, LookbackScanMemory, TileId> scratch(__queue);
    scratch.allocate(num_wgs);

    // Memory Structure:
    // [Lookback Scan Memory, Tile Id Counter]
    auto scan_memory_begin = scratch.scan_memory_ptr();
    auto status_flags_begin = _LookbackScanMemory::get_flags_begin(scan_memory_begin, num_wgs);
    auto tile_id_begin = scratch.tile_id_ptr();

    ::std::size_t num_elements = _LookbackScanMemory::get_num_elements(num_wgs);
    // fill_num_wgs num_elements + 1 to also initialize tile_id_counter
    ::std::size_t fill_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(num_elements + 1, wgsize);

    auto fill_event = __queue.memset(status_flags_begin, 0, num_elements * sizeof(_FlagT) + 1 * sizeof(_TileIdT));

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, hdl);
        auto tile_vals = sycl::local_accessor<_Type, 1>(sycl::range<1>{elems_in_tile}, hdl);
        hdl.depends_on(fill_event);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for(
            sycl::nd_range<1>(num_workitems, wgsize), [=
        ](const sycl::nd_item<1>& item) [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
                auto group = item.get_group();
                ::std::uint32_t local_id = item.get_local_id(0);
                constexpr ::std::uint32_t stride = wgsize;
                auto subgroup = item.get_sub_group();

                std::uint32_t tile_id;
                if constexpr (std::is_same_v<_UseDynamicTileID, ::std::true_type>)
                {
                    // Obtain unique ID for this work-group that will be used in decoupled lookback
                    TileId dynamic_tile_id(tile_id_begin);
                    if (group.leader())
                    {
                        tile_id_lacc[0] = dynamic_tile_id.fetch_inc();
                    }
                    sycl::group_barrier(group);
                    tile_id = tile_id_lacc[0];
                }
                else
                {
                    tile_id = group.get_group_linear_id();
                }

                // Global load into local
                auto wg_current_offset = (tile_id * elems_in_tile);
                auto wg_next_offset = ((tile_id + 1) * elems_in_tile);
                auto wg_local_memory_size = elems_in_tile;

                if (wg_next_offset > n)
                    wg_local_memory_size = n - wg_current_offset;
                //TODO: assumes default ctor produces identity w.r.t. __binary_op
                _Type my_reducer{};
                if (wg_next_offset <= n)
                {
#pragma unroll
                    for (std::uint32_t i = 0; i < elems_per_workitem; ++i)
                    {
                        _Type in_val = __in_rng[wg_current_offset + local_id + stride * i];
                        my_reducer = __binary_op(my_reducer, in_val);
                        tile_vals[local_id + stride * i] = in_val;
                    }
                }
                else
                {
#pragma unroll
                    for (std::uint32_t i = 0; i < elems_per_workitem; ++i)
                    {
                        if (wg_current_offset + local_id + stride * i < n)
                        {
                            _Type in_val = __in_rng[wg_current_offset + local_id + stride * i];
                            my_reducer = __binary_op(my_reducer, in_val);
                            tile_vals[local_id + stride * i] = in_val;
                        }
                    }
                }

                auto local_sum = sycl::reduce_over_group(group, my_reducer, __binary_op);

                auto in_begin = tile_vals.template get_multi_ptr<sycl::access::decorated::no>().get();
                auto out_begin = __out_rng.begin() + wg_current_offset;

                _Type prev_sum = 0;

                // The first sub-group will query the previous tiles to find a prefix
                if (subgroup.get_group_id() == 0)
                {
                    _LookbackScanMemory scan_mem(scan_memory_begin, num_wgs);

                    if (group.leader())
                        scan_mem.set_partial(tile_id, local_sum);

                    // Find lowest work-item that has a full result (if any) and sum up subsequent partial results to obtain this tile's exclusive sum
                    prev_sum = cooperative_lookback()(tile_id, subgroup, __binary_op, scan_mem);

                    if (group.leader())
                        scan_mem.set_full(tile_id, prev_sum + local_sum);
                }

                _Type carry = sycl::group_broadcast(group, prev_sum, 0);
// TODO: Find a fix for _ONEDPL_PRAGMA_UNROLL
#pragma unroll
                for (::std::uint32_t step = 0; step < elems_per_workitem; ++step)
                {
                    ::std::uint32_t i = stride * step;
                    _Type x;
                    if (i + local_id < wg_local_memory_size)
                    {
                        x = in_begin[i + local_id];
                    }
                    _Type out = sycl::inclusive_scan_over_group(group, x, __binary_op, carry);
                    if (i + local_id < wg_local_memory_size)
                    {
                        out_begin[i + local_id] = out;
                    }
                    carry = group_broadcast(group, out, stride - 1);
                }
            });
    });

    scratch.async_free(event);

    event.wait();
}

// The generic structure for configuring a kernel
template <std::uint16_t ElemsPerWorkItem, std::uint16_t WorkGroupSize, typename KernelName>
struct kernel_param
{
    static constexpr std::uint16_t elems_per_workitem = ElemsPerWorkItem;
    static constexpr std::uint16_t workgroup_size = WorkGroupSize;
    using kernel_name = KernelName;
};

template <typename _KernelParam, typename _Inclusive, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin,
                           _BinaryOp __binary_op)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    // Avoid aspect query overhead for sizeof(Types) > 32 bits
    if constexpr (sizeof(typename std::iterator_traits<_InIterator>::value_type) <= sizeof(std::uint32_t))
    {
        if (__queue.get_device().has(sycl::aspect::atomic64))
        {
            single_pass_scan_impl<_KernelParam, _Inclusive, /* UseAtomic64 */ std::true_type,
                                  /* UseDynamicTileID */ std::false_type>(__queue, __buf1.all_view(), __buf2.all_view(),
                                                                          __binary_op);
        }
        else
        {
            single_pass_scan_impl<_KernelParam, _Inclusive, /* UseAtomic64 */ std::false_type,
                                  /* UseDynamicTileID */ std::false_type>(__queue, __buf1.all_view(), __buf2.all_view(),
                                                                          __binary_op);
        }
    }
    else
    {
        single_pass_scan_impl<_KernelParam, _Inclusive, /* UseAtomic64 */ std::false_type,
                              /* UseDynamicTileID */ std::false_type>(__queue, __buf1.all_view(), __buf2.all_view(),
                                                                      __binary_op);
    }
}

template <typename _KernelParam, typename _Inclusive, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_single_wg_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end,
                                     _OutIterator __out_begin, _BinaryOp __binary_op)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    // Avoid aspect query overhead for sizeof(Types) > 32 bits
    single_pass_scan_impl_single_wg<_KernelParam, /* Inclusive */ std::true_type>(__queue, __buf1.all_view(),
                                                                                  __buf2.all_view(), __binary_op);
}

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin,
                           _BinaryOp __binary_op)
{
    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;
    // Avoid non_uniform n by padding up to a multiple of wgsize
    constexpr ::std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    auto __n = __in_end - __in_begin;

    if (__n <= elems_in_tile)
    {
        single_pass_single_wg_inclusive_scan<_KernelParam, /* Inclusive */ std::true_type>(
            __queue, __in_begin, __in_end, __out_begin, __binary_op);
    }
    else
    {
        single_pass_inclusive_scan<_KernelParam, /* Inclusive */ std::true_type>(__queue, __in_begin, __in_end,
                                                                                 __out_begin, __binary_op);
    }
}

template <typename _KernelParam, typename _InRange, typename _OutRange, typename _NumSelectedRange, typename _UnaryPredicate>
void
single_pass_copy_if_impl_single_wg(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _NumSelectedRange __num_rng, _UnaryPredicate pred)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;
    using _SizeT = uint64_t;
    using _TileIdT = TileId::_TileIdT;

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    constexpr std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_workitems = num_wgs * wgsize;
    assert(num_wgs == 1);

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto wg_copy_if_values = sycl::local_accessor<_Type, 1>(sycl::range<1>{elems_in_tile}, hdl);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng, __num_rng);
        hdl.parallel_for(sycl::nd_range<1>(num_workitems, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            auto group = item.get_group();
            auto wg_local_id = item.get_local_id(0);
            constexpr ::std::uint32_t stride = wgsize;                 
                                                            
            // Global load into local
            _SizeT wg_count = 0;

            // Phase 1: Create wg_count and construct in-order wg_copy_if_values
            if (elems_in_tile <= n) {
#pragma unroll
              for (size_t i = 0; i < elems_in_tile; i += wgsize) {
                _Type val = __in_rng[i + wg_local_id];

                _SizeT satisfies_pred = pred(val);
                _SizeT count = sycl::exclusive_scan_over_group(group, satisfies_pred, wg_count, sycl::plus<_SizeT>());

                if (satisfies_pred)
                  wg_copy_if_values[count] = val;

                wg_count = sycl::group_broadcast(group, count + satisfies_pred, wgsize - 1);
              }
            } else {
              // Edge of input, have to handle memory bounds
              // Might have unneccessary group_barrier calls
#pragma unroll
              for (size_t i = 0; i < elems_in_tile; i += wgsize) {
                _SizeT satisfies_pred = 0;
                _Type val = *std::launder(reinterpret_cast<_Type*>(alloca(sizeof(_Type))));
                if (i + wg_local_id < n) {
                  val = __in_rng[i + wg_local_id];

                  satisfies_pred = pred(val);
                }
                _SizeT count = sycl::exclusive_scan_over_group(group, satisfies_pred, wg_count, sycl::plus<_SizeT>());

                if (satisfies_pred)
                  wg_copy_if_values[count] = val;

                wg_count = sycl::group_broadcast(group, count + satisfies_pred, wgsize - 1);
              }
            }

            // Phase 3: copy values to global memory
            for (int i = wg_local_id; i < wg_count; i += wgsize) {
                __out_rng[i] = wg_copy_if_values[i];
            }
            if (group.leader())
                __num_rng[0] = wg_count;
        });
    });

    event.wait();
}

template <typename _KernelParam, typename _UseAtomic64, typename _UseDynamicTileID, typename _InRange, typename _OutRange, typename _NumSelectedRange, typename _UnaryPredicate>
void
single_pass_copy_if_impl(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _NumSelectedRange __num_rng, _UnaryPredicate pred)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;
    using _SizeT = uint64_t;
    using _TileIdT = TileId::_TileIdT;
    using _LookbackScanMemory = LookbackScanMemory<_SizeT, _UseAtomic64>;
    using _FlagT = typename _LookbackScanMemory::_FlagT;

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    constexpr std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_workitems = num_wgs * wgsize;

    ScanMemoryManager<_SizeT, _UseAtomic64, LookbackScanMemory, TileId> scratch(__queue);
    scratch.allocate(num_wgs);

    // Memory Structure:
    // [Lookback Scan Memory, Tile Id Counter]
    auto scan_memory_begin = scratch.scan_memory_ptr();
    auto status_flags_begin = _LookbackScanMemory::get_flags_begin(scan_memory_begin, num_wgs);
    auto tile_id_begin = scratch.tile_id_ptr();

    ::std::size_t num_elements = _LookbackScanMemory::get_num_elements(num_wgs);
    // fill_num_wgs num_elements + 1 to also initialize tile_id_counter
    ::std::size_t fill_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(num_elements + 1, wgsize);

    auto fill_event = __queue.memset(status_flags_begin, 0, num_elements * sizeof(_FlagT) + 1 * sizeof(_TileIdT));

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto wg_copy_if_values = sycl::local_accessor<_Type, 1>(sycl::range<1>{elems_in_tile}, hdl);

        auto tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, hdl);
        hdl.depends_on(fill_event);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng, __num_rng);
        hdl.parallel_for(sycl::nd_range<1>(num_workitems, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            auto group = item.get_group();
            auto wg_local_id = item.get_local_id(0);
            auto sg = item.get_sub_group();
            constexpr ::std::uint32_t stride = wgsize;                 
                                                            
            // Init tile_id                                 
            std::uint32_t tile_id;                          
            if constexpr (std::is_same_v<_UseDynamicTileID, ::std::true_type>)
            {
                // Obtain unique ID for this work-group that will be used in decoupled lookback
                TileId dynamic_tile_id(tile_id_begin);
                if (group.leader())
                {
                    tile_id_lacc[0] = dynamic_tile_id.fetch_inc();
                }
                sycl::group_barrier(group);
                tile_id = tile_id_lacc[0];
            }
            else
            {
                tile_id = group.get_group_linear_id();
            }

            _SizeT wg_count = 0;

            // Phase 1: Create wg_count and construct in-order wg_copy_if_values
            if ((tile_id + 1) * elems_in_tile <= n) {
#pragma unroll
              for (size_t i = 0; i < elems_in_tile; i += wgsize) {
                _Type val = __in_rng[i + wg_local_id + elems_in_tile * tile_id];

                _SizeT satisfies_pred = pred(val);
                _SizeT count = sycl::exclusive_scan_over_group(group, satisfies_pred, wg_count, sycl::plus<_SizeT>());

                if (satisfies_pred)
                  wg_copy_if_values[count] = val;

                wg_count = sycl::group_broadcast(group, count + satisfies_pred, wgsize - 1);
              }
            } else {
              // Edge of input, have to handle memory bounds
              // Might have unneccessary group_barrier calls
#pragma unroll
              for (size_t i = 0; i < elems_in_tile; i += wgsize) {
                _SizeT satisfies_pred = 0;
                _Type val = *std::launder(reinterpret_cast<_Type*>(alloca(sizeof(_Type))));
                if (i + wg_local_id + elems_in_tile * tile_id < n) {
                  val = __in_rng[i + wg_local_id + elems_in_tile * tile_id];

                  satisfies_pred = pred(val);
                }
                _SizeT count = sycl::exclusive_scan_over_group(group, satisfies_pred, wg_count, sycl::plus<_SizeT>());

                if (satisfies_pred)
                  wg_copy_if_values[count] = val;

                wg_count = sycl::group_broadcast(group, count + satisfies_pred, wgsize - 1);
              }
            }

            // Phase 2: Global scan across wg_count
            _SizeT prev_sum = 0;

            // The first sub-group will query the previous tiles to find a prefix
            if (sg.get_group_id() == 0)
            {
                _LookbackScanMemory scan_mem(scan_memory_begin, num_wgs);

                if (group.leader())
                    scan_mem.set_partial(tile_id, wg_count);

                // Find lowest work-item that has a full result (if any) and sum up subsequent partial results to obtain this tile's exclusive sum
                prev_sum = cooperative_lookback()(tile_id, sg, sycl::plus<_SizeT>(), scan_mem);

                if (group.leader())
                    scan_mem.set_full(tile_id, prev_sum + wg_count);
            }

            _SizeT start_idx = sycl::group_broadcast(group, prev_sum, 0);
 
            // Phase 3: copy values to global memory
            for (int i = wg_local_id; i < wg_count; i += wgsize) {
                __out_rng[start_idx + i] = wg_copy_if_values[i];
            }
            if (tile_id == (num_wgs - 1) && group.leader())
                __num_rng[0] = start_idx + wg_count;
        });
    });

    event.wait();
    scratch.free();
}

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _NumSelectedRange, typename _UnaryPredicate>
void
single_pass_single_wg_copy_if(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin, _NumSelectedRange __num_begin, _UnaryPredicate pred)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    auto __keep_num =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _NumSelectedRange>();
    auto __buf_num = __keep2(__num_begin, __num_begin + 1);

    single_pass_copy_if_impl_single_wg<_KernelParam>(__queue, __buf1.all_view(), __buf2.all_view(), __buf_num.all_view(), pred);
}

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _NumSelectedRange, typename _UnaryPredicate>
void
single_pass_copy_if(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin, _NumSelectedRange __num_begin, _UnaryPredicate pred)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    auto __keep_num =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _NumSelectedRange>();
    auto __buf_num = __keep2(__num_begin, __num_begin + 1);

    single_pass_copy_if_impl<_KernelParam, /* UseAtomic64 */ std::true_type, /* UseDynamicTileID */ std::true_type>(__queue, __buf1.all_view(), __buf2.all_view(), __buf_num.all_view(), pred);
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
