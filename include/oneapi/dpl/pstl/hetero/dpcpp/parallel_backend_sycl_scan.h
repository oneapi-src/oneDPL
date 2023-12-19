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

    static constexpr bool __is_larger_than_32_bits = sizeof(_T)*8 > 32;
    using _StorageType = ::std::conditional_t<__is_larger_than_32_bits, ::std::uint64_t, ::std::uint32_t>;
    using _AtomicRefT = sycl::atomic_ref<_StorageType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;

    static constexpr ::std::size_t __flag_length = sizeof(_StorageType);

    static constexpr _StorageType __partial_mask = 1ul << (__flag_length*8 - 2);
    static constexpr _StorageType __full_mask = 1ul << (__flag_length*8 - 1);
    static constexpr _StorageType __value_mask = ~(__partial_mask | __full_mask);
    static constexpr _StorageType __oob_value = __partial_mask | __full_mask;

    static constexpr int __padding = SUBGROUP_SIZE;

    __scan_status_flag(_StorageType* __flags_begin, const std::uint32_t __tile_id)
      : __atomic_flag(*(__flags_begin + __tile_id + __padding))
    {

    }

    void set_partial(const _T __val)
    {
        __atomic_flag.store(__val | __partial_mask);
    }

    void set_full(const _T __val)
    {
        __atomic_flag.store(__val | __full_mask);
    }

    template<typename _Subgroup, typename _BinOp>
    _T cooperative_lookback(::std::uint32_t __tile_id, const _Subgroup& __subgroup, _BinOp __bin_op, _StorageType* __flags_begin)
    {
        _T __sum{0};
        auto __local_id = __subgroup.get_local_id();

        for (int __tile = static_cast<int>(__tile_id) - 1; __tile >= 0; __tile -= SUBGROUP_SIZE)
        {
            _AtomicRefT __tile_atomic(*(__flags_begin + __tile + __padding - __local_id));
            _StorageType __tile_val = 0;
            do {
                __tile_val = __tile_atomic.load();
            } while (!sycl::all_of_group(__subgroup, __tile_val != 0));

            bool __is_full = (__tile_val & __full_mask) && ((__tile_val & __partial_mask) == 0);
            auto __is_full_ballot = sycl::ext::oneapi::group_ballot(__subgroup, __is_full);
            ::std::uint32_t __is_full_ballot_bits{};
            __is_full_ballot.extract_bits(__is_full_ballot_bits);

            auto __lowest_item_with_full = sycl::ctz(__is_full_ballot_bits);
            _T __contribution = __local_id <= __lowest_item_with_full ? __tile_val & __value_mask : _T{0};

            // Sum all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            __sum += sycl::reduce_over_group(__subgroup, __contribution, __bin_op);

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (__is_full_ballot_bits)
                break;
        }
        return __sum;
    }

    _AtomicRefT __atomic_flag;
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

    const ::std::size_t __n = __in_rng.size();

    // We need to process the input array by 2^30 chunks for 32-bit ints
    constexpr ::std::size_t __chunk_size = 1ul << (sizeof(_Type) * 8 - 2);
    const ::std::size_t __num_chunks = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __chunk_size);

    auto __max_cu = __queue.get_device().template get_info<sycl::info::device::max_compute_units>();
    constexpr ::std::size_t __wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t __elems_per_workitem = _KernelParam::data_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    std::uint32_t __elems_in_tile = __wgsize * __elems_per_workitem;
    ::std::size_t __num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __elems_in_tile);



    constexpr int __status_flag_padding = SUBGROUP_SIZE;
    std::uint32_t __status_flags_size = __num_wgs+1+__status_flag_padding;


    _FlagStorageType* __status_flags = sycl::malloc_device<_FlagStorageType>(__status_flags_size, __queue);

    auto __fill_event = __queue.submit([&](sycl::handler& __hdl) {
        __hdl.parallel_for<class scan_kt_init>(sycl::range<1>{__status_flags_size}, [=](const sycl::item<1>& __item)  {
            auto __id = __item.get_linear_id();
            __status_flags[__id] = __id < __status_flag_padding ? _FlagType::__oob_value : 0;
        });
    });


#define SCAN_KT_DEBUG 0
#if SCAN_KT_DEBUG
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

    sycl::event __prev_event = __fill_event;
    for (int __chunk = 0; __chunk < __num_chunks; ++__chunk)
    {
        ::std::size_t __current_chunk_size = __chunk == __num_chunks - 1 ? __n % __chunk_size : __chunk_size;
        ::std::size_t __current_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(__current_chunk_size, __elems_in_tile);
        ::std::size_t __current_num_items = __current_num_wgs * __wgsize;

#if SCAN_KT_DEBUG
    printf("== LAUNCHING KERNEL - n=%lu - chunk=%d - items=%lu - wgs=%lu - wgsize=%lu - elems_per_iter=%lu - max_cu=%u\n", n, chunk, current_num_items, num_wgs, wgsize, elems_per_workitem, __max_cu);
#endif

        auto __event = __queue.submit([&](sycl::handler& __hdl) {
            auto __tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, __hdl);
            __hdl.depends_on(__prev_event);

            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
            __hdl.parallel_for<class scan_kt_main>(sycl::nd_range<1>(__current_num_items, __wgsize), [=](const sycl::nd_item<1>& __item)  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
                auto __group = __item.get_group();
                auto __subgroup = __item.get_sub_group();


                // Obtain unique ID for this work-group that will be used in decoupled lookback
                if (__group.leader())
                {
                    sycl::atomic_ref<_FlagStorageType, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> __idx_atomic(__status_flags[__status_flags_size-1]);
                    __tile_id_lacc[0] = __idx_atomic.fetch_add(1);
                }
                sycl::group_barrier(__group);
                ::std::uint32_t __tile_id = __tile_id_lacc[0];
#if SCAN_KT_DEBUG
                debug5[group.get_group_linear_id()] = tile_id;
#endif

                // TODO: only need the cast if size is greater than 2>30, maybe specialize?
                ::std::size_t __current_offset = static_cast<::std::size_t>(__tile_id)*__elems_in_tile;
                ::std::size_t __next_offset = ((static_cast<::std::size_t>(__tile_id)+1)*__elems_in_tile);
                if (__next_offset > __n)
                    __next_offset = __n;
                auto __in_begin = __in_rng.begin() + __current_offset;
                auto __in_end = __in_rng.begin() + __next_offset;
                auto __out_begin = __out_rng.begin() + __current_offset;


#if SCAN_KT_DEBUG
                debug3[tile_id] = current_offset;
                debug4[tile_id] = next_offset;
#endif

                if (__current_offset >= __n)
                    return;

                _Type __local_sum = sycl::joint_reduce(__group, __in_begin, __in_end, __binary_op);
#if SCAN_KT_DEBUG
                debug1[tile_id] = local_sum;
#endif

                _Type __prev_sum = 0;

                // The first sub-group will query the previous tiles to find a prefix
                if (__subgroup.get_group_id() == 0)
                {
                    _FlagType __flag(__status_flags, __tile_id);

                    if (__group.leader())
                        __flag.set_partial(__local_sum);

                    __prev_sum = __flag.cooperative_lookback(__tile_id, __subgroup, __binary_op, __status_flags);
#if SCAN_KT_DEBUG
                    debug2[tile_id] = prev_sum;
#endif

                    if (__group.leader())
                        __flag.set_full(__prev_sum + __local_sum);
                }


                __prev_sum = sycl::group_broadcast(__group, __prev_sum, 0);

                sycl::joint_inclusive_scan(__group, __in_begin, __in_end, __out_begin, __binary_op, __prev_sum);
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

        __prev_event = __event;

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
    }

    auto __free_event = __queue.submit(
        [=](sycl::handler& __hdl)
        {
            __hdl.depends_on(__prev_event);
            __hdl.host_task([=](){ sycl::free(__status_flags, __queue); });
        });

    __free_event.wait();

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

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    single_pass_scan_impl<_KernelParam, true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op);

    auto __out_rng = __buf2.all_view();
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
