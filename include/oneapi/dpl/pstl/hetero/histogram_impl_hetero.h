// -*- C++ -*-
//===-- histogram_impl_hetero.h ---------------------------------------------===//
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

#ifndef _ONEDPL_HISTOGRAM_IMPL_HETERO_H
#define _ONEDPL_HISTOGRAM_IMPL_HETERO_H

#include <cstdint>
#include <iterator>
#include "../../internal/histogram_binhash_utils.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/parallel_backend_sycl_histogram.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _BinHash>
struct __binhash_manager_base
{
    //will always be empty, but just to have some type
    using _extra_memory_type = typename ::std::uint8_t;
    _BinHash __bin_hash;
    __binhash_manager_base(_BinHash __bin_hash_) : __bin_hash(__bin_hash_) {}

    ::std::size_t
    get_required_SLM_elements() const
    {
        return 0;
    }

    void
    require_access(sycl::handler& __cgh) const
    {
    }

    auto
    get_device_copyable_binhash() const
    {
        return __bin_hash;
    }
};

// Augmentation for binhash function which stores a range of dynamic memory to determine bin mapping
template <typename _BinHash, typename _BufferType>
struct __binhash_manager_with_buffer : __binhash_manager_base<_BinHash>
{
    using _base_type = __binhash_manager_base<_BinHash>;
    using _extra_memory_type = typename _BinHash::_range_value_type;
    //While this stays "unused" in this struct, __buffer is required to keep the sycl buffer alive until the kernel has
    // been completed (waited on)
    _BufferType __buffer;

    __binhash_manager_with_buffer(_BinHash __bin_hash_, _BufferType __buffer_)
        : _base_type(__bin_hash_), __buffer(__buffer_)
    {
    }

    ::std::size_t
    get_required_SLM_elements() const
    {
        return this->__bin_hash.get_range().size();
    }

    void
    require_access(sycl::handler& __cgh) const
    {
        oneapi::dpl::__ranges::__require_access(__cgh, this->__bin_hash.get_range());
    }

    auto
    get_device_copyable_binhash() const
    {
        return _base_type::get_device_copyable_binhash();
    }
};

template <typename _BinHash>
auto
__make_binhash_manager(_BinHash __bin_hash)
{
    return __binhash_manager_base(__bin_hash);
}

template <typename _Range>
auto
__make_binhash_manager(oneapi::dpl::__internal::__custom_range_binhash<_Range> __bin_hash)
{
    auto __range_to_upg = __bin_hash.get_range();
    auto __keep_boundaries =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read,
                                                decltype(__range_to_upg.begin())>();
    auto __buffer = __keep_boundaries(__range_to_upg.begin(), __range_to_upg.end());
    return __binhash_manager_with_buffer(oneapi::dpl::__internal::__custom_range_binhash(__buffer.all_view()),
                                         __buffer);
}

template <typename _Name>
struct __hist_fill_zeros_wrapper
{
};

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _IdxHashFunc,
          typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy>
__pattern_histogram(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                    _Size __num_bins, _IdxHashFunc __func, _RandomAccessIterator2 __histogram_first)
{
    //If there are no histogram bins there is nothing to do
    if (__num_bins > 0)
    {
        using _global_histogram_type = typename ::std::iterator_traits<_RandomAccessIterator2>::value_type;
        const auto __n = __last - __first;

        // The access mode we we want here is "read_write" + no_init property to cover the reads required by the main
        //  kernel, but also to avoid copying the data in unnecessarily.  In practice, this "write" access mode should
        //  accomplish this as write implies read, and we avoid a copy-in from the host for "write" access mode.
        // TODO: Add no_init property to get_sycl_range to allow expressivity we need here.
        auto __keep_bins =
            oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::write,
                                                    _RandomAccessIterator2>();
        auto __bins_buf = __keep_bins(__histogram_first, __histogram_first + __num_bins);
        auto __bins = __bins_buf.all_view();

        auto __fill_func = oneapi::dpl::__internal::fill_functor<_global_histogram_type>{_global_histogram_type{0}};
        //fill histogram bins with zeros

        auto __init_event = oneapi::dpl::__par_backend_hetero::__parallel_for(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__hist_fill_zeros_wrapper>(__exec),
            unseq_backend::walk_n<_ExecutionPolicy, decltype(__fill_func)>{__fill_func}, __num_bins, __bins);

        if (__n > 0)
        {
            //need __binhash_manager to stay in scope until the kernel completes to keep the buffer alive
            auto __binhash_manager = __make_binhash_manager(__func);
            auto __keep_input =
                oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read,
                                                        _RandomAccessIterator1>();
            auto __input_buf = __keep_input(__first, __last);

            __parallel_histogram(::std::forward<_ExecutionPolicy>(__exec), __init_event, __input_buf.all_view(),
                                 ::std::move(__bins), ::std::move(__binhash_manager))
                .wait();
        }
        else
        {
            __init_event.wait();
        }
    }
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_NUMERIC_IMPL_HETERO_H
