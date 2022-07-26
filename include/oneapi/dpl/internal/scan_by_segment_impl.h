/*
 *  Copyright (c) Intel Corporation
 *
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_SCAN_BY_SEGMENT_IMPL_H
#define _ONEDPL_SCAN_BY_SEGMENT_IMPL_H

#if _ONEDPL_BACKEND_SYCL

#include <oneapi/dpl/pstl/algorithm_fwd.h>
#include <oneapi/dpl/pstl/parallel_backend.h>
#include <oneapi/dpl/pstl/hetero/utils_hetero.h>

#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>

#include <array>

namespace oneapi
{
namespace dpl 
{
namespace internal
{

template<typename _NdItem, typename _LocalAcc, typename _IdxType, typename _ValueType, typename _BinaryOp> 
inline _ValueType wg_segmented_scan(_NdItem __item, _LocalAcc __local_acc, _IdxType __local_id, 
    _IdxType __delta_local_id, _ValueType __accumulator, _BinaryOp __binary_op, ::std::size_t __wgroup_size)
{
    _IdxType __first = 0;
    __local_acc[__local_id] = __accumulator;

    sycl::group_barrier(__item.get_group());

    for (int32_t __i = 1; __i < __wgroup_size; __i += __i)
    {
        if (__delta_local_id >= __i)
            __accumulator = __binary_op(__local_acc[__first + __local_id - __i], __accumulator);

        __first = __wgroup_size - __first;
        __local_acc[__first + __local_id] = __accumulator;

        sycl::group_barrier(__item.get_group());
    }

    return (__local_id ? __local_acc[__first + __local_id - 1] : 0);
}


enum class scan_type { inclusive, exclusive };

template <scan_type _scan_type>
struct sycl_scan_by_segment_impl
{   
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _BinaryPredicate, typename _BinaryOperator, typename _T>
    void sycl_inclusive_scan_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values,
        _Range3&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op, _T __init = _T{})
    {
        using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
        using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
        using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using __flag_type = bool;
        
        const int __n = __keys.size();
        __val_type __identity = __val_type{};
        
        if constexpr (__dpl_sycl::__has_known_identity<_BinaryOperator, __val_type>::value)
            __identity = __dpl_sycl::__known_identity<_BinaryOperator, __val_type>::value;
        else
            __identity = __init;

        // TODO: Investigate how to make this a compile-time evaluated tuning parameter based on the data type or other information. 
        constexpr int __vals_per_item = 2; // Assigning 2 elements per work item resulted in best performance on gpu.

        // use workgroup size as the maximum segment size.
        ::std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
        // change __wgroup_size according to local memory limit
        __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size(
            ::std::forward<_ExecutionPolicy>(__exec), sizeof(__key_type) + sizeof(__val_type), __wgroup_size);

        int __n_groups = 1 + std::ceil(__n / (__wgroup_size * __vals_per_item));

        auto __partials = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups)
            .get_buffer();
        
        auto __end_idx = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, 1)
            .get_buffer();

        // the number of segment ends found in each work group 
        auto __seg_ends = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, bool>(__exec, __n_groups)
            .get_buffer();
        
        // 1. Work group reduction
        auto __wg_excl_scan =
        __exec.queue().submit([&](sycl::handler& __cgh) {
            auto __partials_acc = __partials.template get_access<sycl::access_mode::write>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::write>(__cgh);

            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __values, __out_values);

            auto __loc_acc = sycl::accessor<__val_type, 1, 
                                   sycl::access::mode::read_write, 
                                   sycl::access::target::local>{2 * __wgroup_size, __cgh}; 

            __cgh.parallel_for(sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size},
            [=](sycl::nd_item<1> __item) {
                __val_type __accumulator = __identity;

                auto __group = __item.get_group();
                ::std::size_t __global_id = __item.get_global_id();
                ::std::size_t __local_id = __group.get_local_id();
                
                // 1a. Perform a serial scan within the work item over assigned elements. Store partial 
                // reductions in work item memory, and write the accumulated value and number of counted
                // segments into work group memory.
                int32_t __start = __global_id * __vals_per_item;
                int32_t __end = sycl::minimum{}(__start + __vals_per_item, __n);

                if (__global_id == 0 && __local_id == 0)
                    __accumulator = __init;

                // Flag indicating if a new segment exists in the work item's search space 
                int __max_end = 0;
                
                bool __first = true;
                if (__start < __n && __local_id != 0 && __keys[__start] != __keys[__start - 1])
                   __max_end = __local_id;

                for (int32_t __i = __start; __i < __end; ++__i) 
                {
                    if (__first)
                    {
                        __accumulator = __values[__i];
                        __first = false;
                    }
                    else
                        __accumulator = __binary_op(__accumulator, __values[__i]);

                    __out_values[__i] = __accumulator;
                    // clear the accumulator if we reach end of segment 
                    if (__n - 1 == __i || __keys[__i] != __keys[__i+1])
                    {
                        __accumulator = __init;
                        __max_end = __local_id;
                        __first = true;
                    }
                }
                
                __loc_acc[__local_id] = __accumulator;

                // 1b. Perform a work group scan to find the carry in value to apply to each item. 
                auto __delta_local_id = sycl::inclusive_scan_over_group(__group, __max_end, sycl::maximum<>());

                __val_type __carry_in = wg_segmented_scan(__item, __loc_acc, __local_id, 
                    __local_id - __delta_local_id, __accumulator, __binary_op, __wgroup_size); // need to use exclusive scan delta

                // 1c. Update local partial reductions and write to global memory.
                if (__local_id != 0 && __start < __n && __keys[__start] == __keys[__start - 1])
                {
                    for (int32_t __i = __start; __i < __end; ++__i)
                    {
                        __out_values[__i] = __binary_op(__carry_in, __out_values[__i]);

                        if (__i == __n-1 || __keys[__i] != __keys[__i+1]) 
                            break;
                    }
                }
                
                if (__local_id == __wgroup_size - 1) // last work item writes the group's carry out  
                {
                    auto __group_id = __group.get_group_id(0);

                    if (__max_end == 0)
                        __partials_acc[__group_id] = __binary_op(__carry_in, __accumulator); // inclusive with last element
                    
                    else 
                        __partials_acc[__group_id] = __accumulator;
                    
                    bool __seg_present = static_cast<bool>(__delta_local_id);
                    __seg_ends_acc[__group_id] = __seg_present;
                }
            });
        });

        // 2. Apply work group carry outs, calculate output indices, and load results into correct indices. 
        __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_values);
            
            auto __partials_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);

            __cgh.depends_on(__wg_excl_scan);

            __cgh.parallel_for(sycl::nd_range<1>{__n_groups *__wgroup_size, __wgroup_size},
            [=](sycl::nd_item<1> __item) {
                auto __group = __item.get_group();
                auto __group_id = __group.get_group_id(0);
                int __global_id = __item.get_global_id();
                auto __local_id = __item.get_local_id();

                int __wg_agg_idx = __group_id - 1;
                __val_type __agg_collector{};
                bool __first = true;
                
                int __start = __global_id * __vals_per_item;
                int __end = sycl::minimum{}(__start + __vals_per_item, __n);

                // 2a. Calculate the work group's carry-in value.  
                // TODO: currently done serially but expected to be fast assuming n >> max_segment_size.
                // performance expected to degrade if very few segments.
                bool __ag_exists = false;
                if (__local_id == 0 && __wg_agg_idx >= 0)
                {
                    if (__start < __n && __keys[__start] == __keys[__start - 1])
                    {
                        __ag_exists = true;
                        for (int32_t __i = __wg_agg_idx; __i >= 0; --__i) 
                        {
                            const auto& __wg_aggregate = __partials_acc[__i];
                            const auto& __b_seg_end = __seg_ends_acc[__i];

                            if (__first)
                            {
                                __agg_collector = __wg_aggregate;
                                __first = false;
                            }
                            else
                                __agg_collector = __binary_op(__wg_aggregate, __agg_collector);

                            // current aggregate is the last aggregate 
                            if (__b_seg_end)
                                break;
                        }
                    }
                }

                __ag_exists = sycl::group_broadcast(__group, __ag_exists);
                if (!__ag_exists)
                    return;
                
                __agg_collector = sycl::group_broadcast(__group, __agg_collector);

                // 2c. Second pass over the keys, reidentifying end segments and applying work group
                // aggregates if appropriate. Both the key and reduction value are written to the final output  
                ::std::size_t __local_min_key_idx = __n - 1;
                
                // find the smallest index 
                for (int32_t __i = __start; __i < __end; ++__i) 
                {
                    if (__i == __n - 1 || __keys[__i] != __keys[__i+1])
                    {
                        if (__i < __local_min_key_idx)
                            __local_min_key_idx = __i;
                    }
                }

                ::std::size_t __wg_min_seg_end = sycl::reduce_over_group(__group, __local_min_key_idx, sycl::minimum<>());
                
                // the first group will never have an aggregate to apply
                if (__group_id == 0)
                    return;

                for (int32_t __i = __start; __i < sycl::minimum{}(__wg_min_seg_end + 1, __end); ++__i)
                    __out_values[__i] = __binary_op(__agg_collector, __out_values[__i]);
            });
        }).wait();
    }

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _BinaryPredicate, typename _BinaryOperator, typename T>
    void sycl_exclusive_scan_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values,
        _Range3&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op, T __init = T{})
    {
        using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
        using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
        using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using __flag_type = bool;
        
        const int __n = __keys.size();
        __val_type __identity = __val_type{};
    
        if constexpr (__dpl_sycl::__has_known_identity<_BinaryOperator, __val_type>::value)
            __identity = __dpl_sycl::__known_identity<_BinaryOperator, __val_type>::value;
        else
            __identity = {}; // how to handle?

        // TODO: Investigate how to make this a compile-time evaluated tuning parameter based on the data type or other information. 
        constexpr int __vals_per_item = 2; // Assigning 2 elements per work item resulted in best performance on gpu.

        // use workgroup size as the maximum segment size.
        ::std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
        // change __wgroup_size according to local memory limit
        __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size(
            ::std::forward<_ExecutionPolicy>(__exec), sizeof(__key_type) + sizeof(__val_type), __wgroup_size);

        int __n_groups = 1 + std::ceil(__n / (__wgroup_size * __vals_per_item));

        auto __partials = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups)
            .get_buffer();
        
        auto __end_idx = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, 1)
            .get_buffer();

        // the number of segment ends found in each work group 
        auto __seg_ends = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, bool>(__exec, __n_groups)
            .get_buffer();
        
        // 1. Work group reduction
        auto __wg_excl_scan =
        __exec.queue().submit([&](sycl::handler& __cgh) {
            auto __partials_acc = __partials.template get_access<sycl::access_mode::write>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::write>(__cgh);

            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __values, __out_values);

            auto __loc_acc = sycl::accessor<__val_type, 1, 
                                   sycl::access::mode::read_write, 
                                   sycl::access::target::local>{2 * __wgroup_size, __cgh}; 

            __cgh.parallel_for(sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size},
            [=](sycl::nd_item<1> __item) {
                __val_type __accumulator = __identity;

                auto __group = __item.get_group();
                int __global_id = __item.get_global_id();
                int __local_id = __group.get_local_id();
                
                // 1a. Perform a serial scan within the work item over assigned elements. Store partial 
                // reductions in work item memory, and write the accumulated value and number of counted
                // segments into work group memory.
                int __start = __global_id * __vals_per_item;
                int __end = sycl::minimum{}(__start + __vals_per_item, __n);

                if (__global_id == 0 && __local_id == 0)
                    __accumulator = __init;

                // Flag indicating if a new segment exists in the work item's search space 
                int __max_end = 0;
                for (int32_t __i = __start; __i < __end; ++__i) 
                {
                    __out_values[__i] = __accumulator;
                    __accumulator = __binary_op(__accumulator, __values[__i]);

                    // clear the accumulator if we reach end of segment 
                    if (__n - 1 == __i || __keys[__i] != __keys[__i+1])
                    {
                        __accumulator = __init;
                        __max_end = __local_id;   
                    }
                }
                
                __loc_acc[__local_id] = __accumulator;

                // 1b. Perform a work group scan to find the carry in value to apply to each item. 
                auto __delta_local_id = sycl::inclusive_scan_over_group(__group, __max_end, sycl::maximum<>());

                __val_type __carry_in = wg_segmented_scan(__item, __loc_acc, __local_id, 
                    __local_id - __delta_local_id, __accumulator, __binary_op, __wgroup_size); // need to use exclusive scan delta

                // 1c. Update local partial reductions and write to global memory.
                if (__local_id != 0) 
                {
                    for (int32_t __i = __start; __i < __end; ++__i)
                    {
                        __out_values[__i] = __binary_op(__carry_in, __out_values[__i]);

                        if (__i == __n-1 || __keys[__i] != __keys[__i+1]) 
                            break;
                    }
                }
                
                if (__local_id == __wgroup_size - 1) // last work item writes the group's carry out  
                {
                    auto __group_id = __group.get_group_id(0);

                    if (__max_end == 0)
                        __partials_acc[__group_id] = __binary_op(__carry_in, __accumulator); // inclusive with last element
                    
                    else 
                        __partials_acc[__group_id] = __accumulator;
                    
                    __flag_type __seg_present = static_cast<__flag_type>(__delta_local_id);
                    __seg_ends_acc[__group_id] = __seg_present;
                }
            });
        });

        // 2. Apply work group carry outs, calculate output indices, and load results into correct indices. 
        __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_values);
            
            auto __partials_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);

            __cgh.depends_on(__wg_excl_scan);

            __cgh.parallel_for(sycl::nd_range<1>{__n_groups *__wgroup_size, __wgroup_size},
            [=](sycl::nd_item<1> __item) {
                auto __group = __item.get_group();
                auto __group_id = __group.get_group_id(0);
                int __global_id = __item.get_global_id();
                auto __local_id = __item.get_local_id();

                ::std::size_t __wg_agg_idx = __group_id - 1;
                __val_type __agg_collector{};

                // 2a. Calculate the work group's carry-in value.  
                // TODO: currently done serially but expected to be fast assuming n >> max_segment_size.
                // performance expected to degrade if very few segments.
                if (__local_id == 0 && __wg_agg_idx >= 0)
                {
                    for (int32_t __i = __wg_agg_idx; __i >= 0; --__i) 
                    {
                        const auto& __wg_aggregate = __partials_acc[__i];
                        const auto& __b_seg_end = __seg_ends_acc[__i];

                        __agg_collector = __binary_op(__wg_aggregate, __agg_collector);

                        // current aggregate is the last aggregate 
                        if (__b_seg_end)
                            break;
                    }
                }
                
                __agg_collector = sycl::group_broadcast(__group, __agg_collector);

                // 2c. Second pass over the keys, reidentifying end segments and applying work group
                // aggregates if appropriate. Both the key and reduction value are written to the final output
                int __start = __global_id * __vals_per_item;
                int __end = sycl::minimum{}(__start + __vals_per_item, __n);

                int __local_min_key_idx = __n - 1;
                
                // find the smallest index 
                for (int32_t __i = __start; __i < __end; ++__i) 
                {
                    if (__i == __n - 1 || __keys[__i] != __keys[__i+1])
                    {
                        if (__i < __local_min_key_idx)
                            __local_min_key_idx = __i;
                    }
                }

                int __wg_min_seg_end = sycl::reduce_over_group(__group, __local_min_key_idx, sycl::minimum<>());
                
                if (__group_id == 0)
                    return;

                for (int32_t __i = __start; __i < sycl::minimum{}(__wg_min_seg_end + 1, __end); ++__i)
                    __out_values[__i] = __binary_op(__agg_collector, __out_values[__i]);
            });
        }).wait();
    }

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _BinaryPredicate, typename _BinaryOperator, typename T>
    void operator()(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values,
        _Range3&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op, T init = T{})
    {
        if constexpr (_scan_type == scan_type::exclusive)
        {
            sycl_exclusive_scan_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys), 
                ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_values), __binary_pred,
                __binary_op, init);
        }
        else
        {
            sycl_inclusive_scan_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys), 
                ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_values), __binary_pred,
                __binary_op, init);
        }
    }
};

} // end namespace internal
} // end namespace dpl
} // end namespace oneapi
#endif
#endif
