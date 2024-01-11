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

inline namespace igpu
{

namespace __impl
{

template <typename... _Name>
class __lookback_init_kernel;

template <typename... _Name>
class __lookback_kernel;


static constexpr int SUBGROUP_SIZE = 32;

template <typename _T>
struct __scan_status_flag
{
    // 00xxxx - not computed
    // 01xxxx - partial
    // 10xxxx - full
    // 110000 - out of bounds

    static constexpr bool __is_larger_than_32_bits = sizeof(_T) * 8 > 32;
    using _StorageType = ::std::conditional_t<__is_larger_than_32_bits, ::std::uint64_t, ::std::uint32_t>;
    using _AtomicRefT = sycl::atomic_ref<_StorageType, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>;

    static constexpr ::std::size_t __flag_length = sizeof(_StorageType);

    static constexpr _StorageType __partial_mask = 1ul << (__flag_length * 8 - 2);
    static constexpr _StorageType __full_mask = 1ul << (__flag_length * 8 - 1);
    static constexpr _StorageType __value_mask = ~(__partial_mask | __full_mask);
    static constexpr _StorageType __oob_value = __partial_mask | __full_mask;

    static constexpr int __padding = SUBGROUP_SIZE;

    __scan_status_flag(_StorageType* __flags_begin, const std::uint32_t __tile_id)
        : __atomic_flag(*(__flags_begin + __tile_id + __padding))
    {
    }

    void
    set_partial(const _T __val)
    {
        __atomic_flag.store(__val | __partial_mask);
    }

    void
    set_full(const _T __val)
    {
        __atomic_flag.store(__val | __full_mask);
    }

    template <typename _Subgroup, typename _BinOp>
    _T
    cooperative_lookback(::std::uint32_t __tile_id, const _Subgroup& __subgroup, _BinOp __bin_op,
                         _StorageType* __flags_begin)
    {
        _T __sum{0};
        auto __local_id = __subgroup.get_local_id();

        for (int __tile = static_cast<int>(__tile_id) - 1; __tile >= 0; __tile -= SUBGROUP_SIZE)
        {
            _AtomicRefT __tile_atomic(*(__flags_begin + __tile + __padding - __local_id));
            _StorageType __tile_val = 0;
            do
            {
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

template <
           typename _KernelName>
struct __lookback_init_submitter;

template <
          typename... _Name>
struct __lookback_init_submitter<
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _StatusFlags, typename _Flag>
    sycl::event
    operator()(sycl::queue __q, _StatusFlags&& __status_flags, ::std::size_t __status_flags_size, ::std::uint16_t __status_flag_padding, _Flag __oob_value) const
    {
        return __q.submit([&](sycl::handler& __hdl) {
        __hdl.parallel_for<_Name...>(sycl::range<1>{__status_flags_size}, [=](const sycl::item<1>& __item) {
            auto __id = __item.get_linear_id();
            __status_flags[__id] = __id < __status_flag_padding ? __oob_value : 0;
        });
    });
    }
};


template <bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp, typename _KernelParam>
sycl::event
__single_pass_scan(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op,
                      _KernelParam)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;
    using _FlagType = __scan_status_flag<_Type>;
    using _FlagStorageType = __scan_status_flag<_Type>::_StorageType;

    using _KernelName = typename _KernelParam::kernel_name;
    using _LookbackInitKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_init_kernel<_KernelName>>;
    using _LookbackKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __lookback_kernel<_KernelName>>;

    const ::std::size_t __n = __in_rng.size();

    if (__n == 0)
      return sycl::event{};

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");

    assert("This device does not support 64-bit atomics" &&
           (sizeof(_Type) < 64 || __queue.get_device().has(sycl::aspect::atomic64)));


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
    std::uint32_t __status_flags_size = __num_wgs + 1 + __status_flag_padding;

    _FlagStorageType* __status_flags = sycl::malloc_device<_FlagStorageType>(__status_flags_size, __queue);

    auto __fill_event = __lookback_init_submitter<_LookbackInitKernel>{}(__queue, __status_flags, __status_flags_size, __status_flag_padding, _FlagType::__oob_value);

    sycl::event __prev_event = __fill_event;
    for (int __chunk = 0; __chunk < __num_chunks; ++__chunk)
    {
        ::std::size_t __current_chunk_size = __chunk == __num_chunks - 1 ? __n % __chunk_size : __chunk_size;
        ::std::size_t __current_num_wgs =
            oneapi::dpl::__internal::__dpl_ceiling_div(__current_chunk_size, __elems_in_tile);
        ::std::size_t __current_num_items = __current_num_wgs * __wgsize;

        auto __event = __queue.submit([&](sycl::handler& __hdl) {
            auto __tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, __hdl);
            __hdl.depends_on(__prev_event);

            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
            __hdl.parallel_for<class scan_kt_main>(
                sycl::nd_range<1>(__current_num_items, __wgsize),
                [=](const sycl::nd_item<1>& __item) [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
                    auto __group = __item.get_group();
                    auto __subgroup = __item.get_sub_group();

                    // Obtain unique ID for this work-group that will be used in decoupled lookback
                    if (__group.leader())
                    {
                        sycl::atomic_ref<_FlagStorageType, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            __idx_atomic(__status_flags[__status_flags_size - 1]);
                        __tile_id_lacc[0] = __idx_atomic.fetch_add(1);
                    }
                    sycl::group_barrier(__group);
                    ::std::uint32_t __tile_id = __tile_id_lacc[0];

                    // TODO: only need the cast if size is greater than 2>30, maybe specialize?
                    ::std::size_t __current_offset = static_cast<::std::size_t>(__tile_id) * __elems_in_tile;
                    ::std::size_t __next_offset = ((static_cast<::std::size_t>(__tile_id) + 1) * __elems_in_tile);
                    if (__next_offset > __n)
                        __next_offset = __n;
                    auto __in_begin = __in_rng.begin() + __current_offset;
                    auto __in_end = __in_rng.begin() + __next_offset;
                    auto __out_begin = __out_rng.begin() + __current_offset;

                    if (__current_offset >= __n)
                        return;

                    _Type __local_sum = sycl::joint_reduce(__group, __in_begin, __in_end, __binary_op);
                    _Type __prev_sum = 0;

                    // The first sub-group will query the previous tiles to find a prefix
                    if (__subgroup.get_group_id() == 0)
                    {
                        _FlagType __flag(__status_flags, __tile_id);

                        if (__group.leader())
                            __flag.set_partial(__local_sum);

                        __prev_sum = __flag.cooperative_lookback(__tile_id, __subgroup, __binary_op, __status_flags);

                        if (__group.leader())
                            __flag.set_full(__prev_sum + __local_sum);
                    }

                    __prev_sum = sycl::group_broadcast(__group, __prev_sum, 0);

                    sycl::joint_inclusive_scan(__group, __in_begin, __in_end, __out_begin, __binary_op, __prev_sum);
                });
        });

        __prev_event = __event;
    }

    auto __free_event = __queue.submit([=](sycl::handler& __hdl) {
        __hdl.depends_on(__prev_event);
        __hdl.host_task([=]() { sycl::free(__status_flags, __queue); });
    });

    return __free_event;
}

} // namespace __impl

template <typename _InRng, typename _OutRng, typename _BinaryOp, typename _KernelParam>
sycl::event
inclusive_scan(sycl::queue __queue, _InRng __in_rng, _OutRng __out_rng,
                           _BinaryOp __binary_op, _KernelParam __param)
{

    return __impl::__single_pass_scan<true>(__queue, __in_rng, __out_rng, __binary_op, __param);
}

template <typename _InIterator, typename _OutIterator, typename _BinaryOp, typename _KernelParam>
sycl::event
inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin,
                           _BinaryOp __binary_op, _KernelParam __param)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    return __impl::__single_pass_scan<true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op, __param);
}

} // namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
