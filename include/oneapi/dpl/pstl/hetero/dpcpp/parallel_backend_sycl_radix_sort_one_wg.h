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

#ifndef _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H
#define _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H

//namespace oneapi
//{
//namespace dpl
//{
//namespace __par_backend_hetero
//{

template<typename _KeyT, typename>
class _TempBuf;

template<typename _KeyT>
class _TempBuf<_KeyT, std::true_type /*shared local memory buffer*/>
{
    uint16_t __buf_size;
public:
    _TempBuf(uint16_t __n): __buf_size(__n) {}
    auto get_acc(sycl::handler& __cgh)
    {
        return sycl::local_accessor<_KeyT, 1>(__buf_size, __cgh);
    }
};

template<typename _KeyT>
class _TempBuf<_KeyT, std::false_type /*global memory buffer*/>
{
    sycl::buffer<_KeyT> __buf;

public:
    _TempBuf(uint16_t __n): __buf(__n) {}
    auto get_acc(sycl::handler& __cgh)
    {
        return sycl::accessor(__buf, __cgh, sycl::read_write, sycl::no_init);
    }
};

template <uint16_t __block_size, typename _KeyT, typename _Wi, typename _Src, typename _Keys>
void
__block_load(const _Wi __wi, const _Src& __src, _Keys& __keys, const uint32_t __n)
{
    constexpr _KeyT __default_key = _KeyT{};

    #pragma unroll
    for (uint16_t __i = 0; __i < __block_size; ++__i)
    {
        const uint16_t __offset = __wi*__block_size + __i;
        if (__offset < __n)
            __keys[__i] = __src[__offset];
        else
            __keys[__i] = __default_key;
    }
}

template <uint16_t __block_size, typename _Item, typename _Wi, typename _Lacc, typename _Keys, typename _Indices>
void
__to_blocked(_Item __it, const _Wi __wi, _Lacc& __exchange_lacc, _Keys& __keys, const _Indices& __indices)
{
    #pragma unroll
    for (uint16_t __i = 0; __i < __block_size; ++__i)
        __exchange_lacc[__indices[__i]] = __keys[__i];

    __dpl_sycl::__group_barrier(__it);

    #pragma unroll
    for (uint16_t __i = 0; __i < __block_size; ++__i)
        __keys[__i] = __exchange_lacc[__wi*__block_size + __i];
}

template<typename _KernelName, uint16_t __wg_size = 256/*work group size*/, uint16_t __block_size = 16,
         ::std::uint32_t __radix = 4, bool __is_asc = true, typename _SLM_tag = std::true_type,
         typename _RangeIn, uint16_t __req_sub_group_size = (__block_size < 4 ? 32 : 16)>
auto __subgroup_radix_sort(sycl::queue __q, _RangeIn&& __src)
{
    constexpr uint16_t __bin_count = 1 << __radix;

    uint16_t __n = __src.size();
    assert(__n <= __block_size*__wg_size);
  
# if _ONEDPL_KERNEL_BUNDLE_PRESENT
    auto __kernel_id = sycl::get_kernel_id<_KernelName>();
    auto __bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(__q.get_context(), {__kernel_id});
# endif
  
    using _KeyT = oneapi::dpl::__internal::__value_t<_RangeIn>;

    _TempBuf<_KeyT, _SLM_tag> __buf(__block_size*__wg_size);

    sycl::nd_range __range {sycl::range{__wg_size}, sycl::range{__wg_size}};
    auto __event = __q.submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __src);

        auto __exchange_lacc = __buf.get_acc(__cgh); //exchange key, size is __block_size*__wg_size
        auto __counter_lacc = sycl::local_accessor<uint32_t, 1>(__wg_size * __bin_count + 1, __cgh);//counter, could be private but use slm here
  
# if _ONEDPL_KERNEL_BUNDLE_PRESENT
        __cgh.use_kernel_bundle(__bundle);
# endif
        __cgh.parallel_for<_KernelName>(
            __range, ([=](sycl::nd_item<1> __it) [[sycl::reqd_sub_group_size(__req_sub_group_size)]]
        {
  
            _KeyT __keys[__block_size];
            uint16_t __wi = __it.get_local_linear_id();
            uint16_t __begin_bit = 0;
            constexpr uint16_t __end_bit = sizeof(_KeyT) * 8; 
  
            __block_load<__block_size, _KeyT>(__wi, __src, __keys, __n);
  
            __dpl_sycl::__group_barrier(__it);
            while (true)
            {
                uint16_t __indices[__block_size]; //indices for inderect access in the "re-order" phase
                {
                    uint32_t* __counters[__block_size]; //pointers(by perfomance reasons) to bucket's counters

                    //1. "counting" phase
                    //counter initialization
                    auto __pcounter = __counter_lacc.get_pointer()+__wi;
                    #pragma unroll
                    for (uint16_t __i = 0; __i < __bin_count; ++__i)
                        __pcounter[__i * __wg_size] = 0;

                    #pragma unroll
                    for (uint16_t __i = 0; __i < __block_size; ++__i)
                    {
                        const int __bin =
                            __get_bucket</*mask*/__bin_count - 1>(__order_preserving_cast<__is_asc>(__keys[__i]), __begin_bit);

                        //"counting" and local offset calculation
                        __counters[__i] = &__pcounter[__bin*__wg_size];
                        __indices[__i] = *__counters[__i];
                        *__counters[__i] = __indices[__i] + 1;
                    }
                    __dpl_sycl::__group_barrier(__it);
  
                    //2. scan phase
                    {
                        //TODO: probably can be futher optimized
  
                        //scan contiguous numbers
                        uint16_t __bin_sum[__bin_count];
                        __bin_sum[0] = __counter_lacc[__wi * __bin_count];
                        for (uint16_t __i = 1; __i < __bin_count; ++__i)
                            __bin_sum[__i] = __bin_sum[__i-1] + __counter_lacc[__wi * __bin_count + __i];
  
                        __dpl_sycl::__group_barrier(__it);
                        //exclusive scan local sum
                        uint16_t __sum_scan = __dpl_sycl::__exclusive_scan_over_group(__it.get_group(), __bin_sum[__bin_count-1], sycl::plus<uint16_t>());
                        //add to local sum, generate exclusive scan result
                        for (uint16_t __i = 0; __i < __bin_count; ++__i)
                            __counter_lacc[__wi * __bin_count + __i + 1] = __sum_scan + __bin_sum[__i];
  
                        if (__wi == 0)
                            __counter_lacc[0] = 0;
                        __dpl_sycl::__group_barrier(__it);
                    }
  
                    #pragma unroll
                    for (uint16_t __i = 0; __i < __block_size; ++__i)
                    {
                        // a global index is a local offset plus a global base index
                        __indices[__i] += *__counters[__i];
                    }
                }
  
                __begin_bit += __radix;

                //3. "re-order" phase
                __dpl_sycl::__group_barrier(__it);
                if (__begin_bit >= __end_bit)
                {
                    // the last iteration - writing out the result
                    for (uint16_t __i = 0; __i < __block_size; ++__i)
                    {
                        const uint16_t __r = __indices[__i];
                        if (__r < __n)
                            __src[__r] = __keys[__i];
                    }
                    return;
                }
                __to_blocked<__block_size>(__it, __wi, __exchange_lacc, __keys, __indices);
                __dpl_sycl::__group_barrier(__it);
            }
        }));
     });
   return __event;
}


//} // namespace __par_backend_hetero
//} // namespace dpl
//} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_radix_sort_one_wg_H */
