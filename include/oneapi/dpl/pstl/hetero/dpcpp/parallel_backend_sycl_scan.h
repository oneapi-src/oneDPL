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

namespace oneapi::dpl::experimental::kt
{

inline namespace igpu {

static constexpr int SUBGROUP_SIZE = 32;

template<typename _T>
struct __scan_status_flag
{
    // 00xxxx - not computed
    // 01xxxx - partial
    // 10xxxx - full
    // 110000 - out of bounds

    static constexpr bool is_larger_than_32_bits = sizeof(_T)*8 > 32;
    using _StorageType = ::std::conditional_t<is_larger_than_32_bits, ::std::uint64_t, ::std::uint32_t>;
    using _AtomicRefT = sycl::atomic_ref<_StorageType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;

    static constexpr ::std::size_t flag_length = sizeof(_StorageType);

    static constexpr _StorageType partial_mask = 1ul << (flag_length*8 - 2);
    static constexpr _StorageType full_mask = 1ul << (flag_length*8 - 1);
    static constexpr _StorageType value_mask = ~(partial_mask | full_mask);
    static constexpr _StorageType oob_value = partial_mask | full_mask;

    static constexpr int padding = SUBGROUP_SIZE;

    __scan_status_flag(_StorageType* flags_begin, const std::uint32_t tile_id)
      : atomic_flag(*(flags_begin + tile_id + padding))
    {

    }

    void set_partial(const _T val)
    {
        atomic_flag.store(val | partial_mask);
    }

    void set_full(const _T val)
    {
        atomic_flag.store(val | full_mask);
    }

    template<typename _Subgroup, typename BinOp>
    _T cooperative_lookback(std::uint32_t tile_id, const _Subgroup& subgroup, BinOp bin_op, _StorageType* flags_begin)
    {
        _T sum{0};
        int local_id = subgroup.get_local_id();

        for (int tile = static_cast<int>(tile_id) - 1; tile >= 0; tile -= SUBGROUP_SIZE)
        {
            _AtomicRefT tile_atomic(*(flags_begin + tile + padding - local_id));
            _StorageType tile_val = 0;
            do {
                tile_val = tile_atomic.load();

            } while (!sycl::all_of_group(subgroup, tile_val != 0));

            bool is_full = (tile_val & full_mask) && ((tile_val & partial_mask) == 0);
            auto is_full_ballot = sycl::ext::oneapi::group_ballot(subgroup, is_full);
            ::std::uint32_t is_full_ballot_bits{};
            is_full_ballot.extract_bits(is_full_ballot_bits);

            auto lowest_item_with_full = sycl::ctz(is_full_ballot_bits);
            _T contribution = local_id <= lowest_item_with_full ? tile_val & value_mask : _T{0};

            // Sum all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            sum += sycl::reduce_over_group(subgroup, contribution, bin_op);

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (is_full_ballot_bits)
                break;
        }
        return sum;
    }

    _AtomicRefT atomic_flag;
};

template <typename _KernelParam, bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp>
void
single_pass_scan_impl(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;
    using _FlagType = __scan_status_flag<_Type>;
    using _FlagStorageType = __scan_status_flag<_Type>::_StorageType;

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");
    if (sizeof(_Type) > 32 && !__queue.get_device().has(sycl::aspect::atomic64))
    {
        std::cerr << "TODO" << std::endl;
        exit(1);
    }

    const ::std::size_t n = __in_rng.size();
    auto __max_cu = __queue.get_device().template get_info<sycl::info::device::max_compute_units>();
    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::data_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_items = num_wgs * wgsize;


    constexpr int status_flag_padding = 32;
    std::uint32_t status_flags_size = num_wgs+1+status_flag_padding;


    _FlagStorageType* status_flags = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);

    auto fill_event = __queue.submit([&](sycl::handler& hdl) {

        hdl.parallel_for<class scan_kt_init>(sycl::range<1>{status_flags_size}, [=](const sycl::item<1>& item)  {
                int id = item.get_linear_id();
                status_flags[id] = id < status_flag_padding ? _FlagType::oob_value : 0;
        });
    });


#define SCAN_KT_DEBUG 1
#if SCAN_KT_DEBUG
    printf("== LAUNCHING KERNEL - n=%lu - items=%lu - wgs=%lu - wgsize=%lu - elems_per_iter=%lu - max_cu=%u\n", n, num_items, num_wgs, wgsize, elems_per_workitem, __max_cu);
    printf("atomic64 %d\n", __queue.get_device().has(sycl::aspect::atomic64));
    std::vector<_FlagStorageType> debug11v(status_flags_size);
    __queue.memcpy(debug11v.data(), status_flags, status_flags_size * sizeof(_FlagStorageType));

    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "flag_before " << i << " " << debug11v[i] << std::endl;

    _FlagStorageType* debug1 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    _FlagStorageType* debug2 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    _FlagStorageType* debug3 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    _FlagStorageType* debug4 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    _FlagStorageType* debug5 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    _FlagStorageType* debug6 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    _FlagStorageType* debug7 = sycl::malloc_device<_FlagStorageType>(status_flags_size, __queue);
    printf("out_begin %p\n", (void*)__out_rng.begin());
    printf("status_flags %p\n", (void*)status_flags);
#endif

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, hdl);
        hdl.depends_on(fill_event);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for<class scan_kt_main>(sycl::nd_range<1>(num_items, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            auto group = item.get_group();
            auto subgroup = item.get_sub_group();


            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (group.leader())
            {
                sycl::atomic_ref<_FlagStorageType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> idx_atomic(status_flags[status_flags_size-1]);
                tile_id_lacc[0] = idx_atomic.fetch_add(1);
            }
            sycl::group_barrier(group);
            std::uint32_t tile_id = tile_id_lacc[0];
#if SCAN_KT_DEBUG
            debug5[group.get_group_linear_id()] = tile_id;
#endif

            auto current_offset = (tile_id*elems_in_tile);
            auto next_offset = ((tile_id+1)*elems_in_tile);
            if (next_offset > n)
                next_offset = n;
            auto in_begin = __in_rng.begin() + current_offset;
            auto in_end = __in_rng.begin() + next_offset;
            auto out_begin = __out_rng.begin() + current_offset;


#if SCAN_KT_DEBUG
            debug3[tile_id] = current_offset;
            debug4[tile_id] = next_offset;
#endif

            if (current_offset >= n)
                return;

            _Type local_sum = sycl::joint_reduce(group, in_begin, in_end, __binary_op);
#if SCAN_KT_DEBUG
            debug1[tile_id] = local_sum;
#endif

            _Type prev_sum = 0;

            // The first sub-group will query the previous tiles to find a prefix
            if (subgroup.get_group_id() == 0)
            {
                _FlagType flag(status_flags, tile_id);

                if (group.leader())
                    flag.set_partial(local_sum);

                prev_sum = flag.cooperative_lookback(tile_id, subgroup, __binary_op, status_flags);
#if SCAN_KT_DEBUG
                debug2[tile_id] = prev_sum;
#endif

                if (group.leader())
                    flag.set_full(prev_sum + local_sum);
            }


            //sycl::group_barrier(group);
            prev_sum = sycl::group_broadcast(group, prev_sum, 0);
            //sycl::group_barrier(group);
            sycl::joint_inclusive_scan(group, in_begin, in_end, out_begin, __binary_op, prev_sum);
#if SCAN_KT_DEBUG
            sycl::group_barrier(group);
            if (group.leader())
            {
                debug6[tile_id] = *out_begin;
                debug7[tile_id] = *(out_begin+next_offset-1);
            }
#endif
        });
    });


#if SCAN_KT_DEBUG
    event.wait();
    std::vector<_FlagStorageType> debug1v(status_flags_size);
    std::vector<_FlagStorageType> debug2v(status_flags_size);
    std::vector<_FlagStorageType> debug3v(status_flags_size);
    std::vector<_FlagStorageType> debug4v(status_flags_size);
    std::vector<_FlagStorageType> debug5v(status_flags_size);
    std::vector<_FlagStorageType> debug6v(status_flags_size);
    std::vector<_FlagStorageType> debug7v(status_flags_size);
    std::vector<_FlagStorageType> debug_status_v(status_flags_size);
    __queue.memcpy(debug1v.data(), debug1, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug2v.data(), debug2, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug3v.data(), debug3, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug4v.data(), debug4, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug5v.data(), debug5, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug6v.data(), debug6, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug7v.data(), debug7, status_flags_size * sizeof(_FlagStorageType));
    __queue.memcpy(debug_status_v.data(), status_flags, status_flags_size * sizeof(_FlagStorageType));

    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "tile " << i << " " << debug5v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "local_sum " << i << " " << debug1v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
    {
        auto val = (debug_status_v[i] & _FlagType::value_mask);
        int a = val / elems_in_tile;
        int b = val % elems_in_tile;
        std::cout << "flags " << i << " " << std::bitset<sizeof(_FlagStorageType)*8>(debug_status_v[i]) << " (" << val<< " = " << a << "/" << elems_in_tile << "+" << b <<")" << std::endl;
    }
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "lookback " << i << " " << debug2v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "offset " << i << " " << debug3v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "end " << i << " " << debug4v[i] << std::endl;

    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "out_first " << i << " " << debug6v[i] << std::endl;
    for (int i = 0; i < status_flags_size-1; ++i)
        std::cout << "out_last " << i << " " << debug7v[i] << std::endl;
#endif

    auto free_event = __queue.submit(
        [=](sycl::handler& hdl)
        {
            hdl.depends_on(event);
            hdl.host_task([=](){ sycl::free(status_flags, __queue); });
        });

    free_event.wait();

}

// The generic structure for configuring a kernel
template <std::uint16_t DataPerWorkItem, std::uint16_t WorkGroupSize, typename KernelName>
struct kernel_param
{
    static constexpr std::uint16_t data_per_workitem = DataPerWorkItem;
    static constexpr std::uint16_t workgroup_size = WorkGroupSize;
    using kernel_name = KernelName;
};

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin, _BinaryOp __binary_op)
{
    auto __n = __in_end - __in_begin;

    // TODO: check semantics
    if (__n == 0)
        return;

#if 0
    using _Type = std::remove_pointer_t<_InIterator>;
    std::vector<_Type> in_debug(__n);
    __queue.memcpy(in_debug.data(), __in_begin, __n * sizeof(_Type));

   for (int i = 0; i < __n; ++i)
        std::cout << "input_before " << i << " " << in_debug[i] << std::endl;
#endif
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    single_pass_scan_impl<_KernelParam, true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op);

    auto __out_rng = __buf2.all_view();



#if 0
#if 0
    //using _Type = std::remove_pointer_t<_InIterator>;
    std::vector<_Type> in_debug2(__n);
    __queue.memcpy(in_debug2.data(), __in_begin, __n * sizeof(_Type));

    for (int i = 0; i < __n; ++i)
        std::cout << "input_after " << i << " " << in_debug2[i] << std::endl;
#endif
#endif
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
