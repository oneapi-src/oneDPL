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

template<typename _T>
struct __scan_status_flag
{
    using _AtomicRefT = sycl::atomic_ref<::std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
    static constexpr std::uint32_t partial_mask = 1 << (sizeof(std::uint32_t)*8 - 2);
    static constexpr std::uint32_t full_mask = 1 << (sizeof(std::uint32_t)*8 - 1);
    static constexpr std::uint32_t value_mask = ~(partial_mask | full_mask);

    __scan_status_flag(std::uint32_t* flags_begin, const std::uint32_t tile_id)
      : atomic_flag(*(flags_begin + tile_id))
    {

    }

    void set_partial(std::uint32_t val)
    {
        atomic_flag.store(val | partial_mask);
    }

    void set_full(std::uint32_t val)
    {
        atomic_flag.store(val | full_mask);
    }

    _T lookback(const std::uint32_t tile_id, std::uint32_t* flags_begin)
    {
        _T sum = 0;
        int i = 0;
        for (std::int32_t tile = static_cast<std::int32_t>(tile_id) - 1; tile >= 0; --tile)
        {
            _AtomicRefT tile_atomic(*(flags_begin + tile));
            std::uint32_t tile_val = 0;
            do {
                tile_val = tile_atomic.load();
            } while (tile_val == 0);

            sum += tile_val & value_mask;

            // If this was a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (tile_val & full_mask)
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

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();
    auto __max_cu = __queue.get_device().template get_info<sycl::info::device::max_compute_units>();
    //std::size_t num_wgs = __max_cu;
    std::size_t num_wgs = 64;

    // TODO: use wgsize and iters per item from _KernelParam
    std::size_t wgsize = n/num_wgs;

    std::uint32_t status_flags_buf_size = num_wgs+1;
    sycl::buffer<uint32_t, 1> status_flags_buf(status_flags_buf_size);

    // TODO: this probably isn't the best way to do this
    {
    sycl::host_accessor<std::uint32_t, 1> status_flags(status_flags_buf);
    for (std::size_t i = 0; i < status_flags_buf_size; ++i)
        status_flags[i] = 0;
    }

//    printf("launching kernel items=%lu wgs=%lu wgsize=%lu max_cu=%lu\n", n, num_wgs, wgsize, __max_cu);


    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto status_flags = sycl::accessor<std::uint32_t, 1, sycl::access_mode::read_write>(status_flags_buf, hdl);
        auto tile_id_lacc = sycl::accessor<std::uint32_t, 1, sycl::access_mode::read_write, sycl::target::local>(sycl::range<1>{1}, hdl);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for(sycl::nd_range<1>(n, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(32)]] {
            auto item_id = item.get_local_linear_id();
            auto group = item.get_group();

            //std::uint32_t elems_in_tile = elems_per_item*wgsize;
            std::uint32_t elems_in_tile = wgsize;

            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (group.leader())
            {
                sycl::atomic_ref<::std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> idx_atomic(status_flags[status_flags_buf_size-1]);
                tile_id_lacc[0] = idx_atomic.fetch_add(1);
            }
            sycl::group_barrier(group);
            std::uint32_t tile_id = tile_id_lacc[0];

            auto in_begin = __in_rng.begin() + (tile_id*elems_in_tile);
            auto in_end = __in_rng.begin() + ((tile_id+1)*elems_in_tile);
            auto out_begin = __out_rng.begin() + (tile_id*elems_in_tile);

            auto local_sum = sycl::joint_reduce(group, in_begin, in_end, __binary_op);

			__scan_status_flag<_Type> flag(status_flags.get_pointer(), tile_id);
			flag.set_partial(local_sum);

            auto prev_sum = flag.lookback(tile_id, status_flags.get_pointer());
            //auto prev_sum = 0;
            flag.set_full(prev_sum + local_sum);

            sycl::joint_inclusive_scan(group, in_begin, in_end, out_begin, __binary_op, prev_sum);
        });
    });

    event.wait();
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
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    single_pass_scan_impl<_KernelParam, true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op);
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
