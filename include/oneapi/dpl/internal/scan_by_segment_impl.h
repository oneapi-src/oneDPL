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
template<typename NdItem, typename LocalAcc, typename IdxType, typename ValueType, typename BinaryOp> 
inline ValueType wg_segmented_scan(NdItem item, LocalAcc local_acc, IdxType local_id, 
    IdxType delta_local_id, ValueType accumulator, BinaryOp binary_op, int __wgroup_size)
{
    IdxType first = 0;
    local_acc[first + local_id] = accumulator;

    sycl::group_barrier(item.get_group());

    for (int i = 1; i < __wgroup_size; i += i)
    {
        if (delta_local_id >= i)
            accumulator = binary_op(local_acc[first + local_id - i], accumulator);
        first = __wgroup_size - first;
        local_acc[first + local_id] = accumulator;

        sycl::group_barrier(item.get_group());
    }

    return (local_id ? local_acc[first + local_id - 1] : 0);
}


enum class scan_type { inclusive, exclusive };

template <scan_type _scan_type>
struct scan_by_segment_impl
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _BinaryPredicate, typename _BinaryOperator>
    void operator()(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values,
        _Range3&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
    {
        using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
        using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
        using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using __flag_type = bool;
        
        const int __n = __keys.size();
        __val_type identity = __val_type{};
        
        // TODO: Investigate how to make this a compile-time evaluated tuning parameter based on the data type or other information. 
        constexpr int __vals_per_item = 2; // Assigning 2 elements per work item resulted in best performance on gpu.

        // use workgroup size as the maximum segment size.
        ::std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);
        // change __wgroup_size according to local memory limit
        __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size(
            ::std::forward<_ExecutionPolicy>(__exec), sizeof(__key_type) + sizeof(__val_type), __wgroup_size);

        int __n_groups = 1 + std::ceil(__n / (__wgroup_size * __vals_per_item));

        auto partials = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups)
            .get_buffer();
        
        auto end_idx = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, 1)
            .get_buffer();

        // the number of segment ends found in each work group 
        auto seg_ends = oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, bool>(__exec, __n_groups)
            .get_buffer();
        
        // 1. Work group reduction
        auto wg_excl_scan =
        __exec.queue().submit([&](sycl::handler& cgh) {
            auto partials_acc = partials.template get_access<sycl::access_mode::write>(cgh);
            auto seg_ends_acc = seg_ends.template get_access<sycl::access_mode::write>(cgh);

            oneapi::dpl::__ranges::__require_access(cgh, __keys, __values, __out_values);

            auto loc_acc = sycl::accessor<__val_type, 1, 
                                   sycl::access::mode::read_write, 
                                   sycl::access::target::local>{2 * __wgroup_size, cgh}; 

            cgh.parallel_for(sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size},
            [=](sycl::nd_item<1> item) {
                __val_type accumulator = identity;

                auto group = item.get_group();
                int global_id = item.get_global_id();
                int local_id = group.get_local_id();
                
                // 1a. Perform a serial scan within the work item over assigned elements. Store partial 
                // reductions in work item memory, and write the accumulated value and number of counted
                // segments into work group memory.
                int start = global_id * __vals_per_item;
                int end = sycl::minimum{}(start + __vals_per_item, __n);

                // Flag indicating if a new segment exists in the work item's search space 
                int max_end = 0;
                
                for (int i = start; i < end; ++i) 
                {
                    if constexpr (_scan_type == scan_type::exclusive) 
                    {
                        __out_values[i] = accumulator;
                        accumulator = __binary_op(accumulator, __values[i]);
                    }
                    else // inclusive scan 
                    {
                        accumulator = __binary_op(accumulator, __values[i]);
                        __out_values[i] = accumulator;
                    }

                    // clear the accumulator if we reach end of segment 
                    if (__n - 1 == i || __keys[i] != __keys[i+1])
                    {
                        accumulator = identity;
                        max_end = local_id; 
                    }
                }
                
                loc_acc[local_id] = accumulator;

                // 1b. Perform a work group scan to find the carry in value to apply to each item. 
                auto delta_local_id = sycl::inclusive_scan_over_group(group, max_end, sycl::maximum<>());

                __val_type carry_in = wg_segmented_scan(item, loc_acc, local_id, 
                    local_id - delta_local_id, accumulator, __binary_op, __wgroup_size); // need to use exclusive scan delta

                // 1c. Update local partial reductions and write to global memory.
                if (group.get_group_id() != 0 || local_id != 0)
                {
                    for (int i = start; i < end; ++i)
                    {
                        __out_values[i] = __binary_op(carry_in, __out_values[i]);

                        if (i == __n-1 || __keys[i] != __keys[i+1]) 
                            break;
                    }
                }
                
                if (local_id == __wgroup_size - 1) // last work item writes the group's carry out  
                {
                    auto group_id = group.get_group_id(0);

                    if (max_end == 0)
                        partials_acc[group_id] = __binary_op(carry_in, accumulator); // inclusive with last element
                    
                    else 
                        partials_acc[group_id] = accumulator;

                    seg_ends_acc[group_id] = static_cast<bool>(delta_local_id);
                }
            });
        });

        // 2. Apply work group carry outs, calculate output indices, and load results into correct indices. 
        __exec.queue().submit([&](sycl::handler& cgh) {
            oneapi::dpl::__ranges::__require_access(cgh, __keys, __out_values);
            
            auto partials_acc = partials.template get_access<sycl::access_mode::read>(cgh);
            auto seg_ends_acc = seg_ends.template get_access<sycl::access_mode::read>(cgh);

            cgh.depends_on(wg_excl_scan);

            cgh.parallel_for(sycl::nd_range<1>{__n_groups *__wgroup_size, __wgroup_size},
            [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                auto group_id = group.get_group_id(0);
                int global_id = item.get_global_id();
                auto local_id = item.get_local_id();

                int wg_agg_idx = group_id - 1;
                __val_type agg_collector{};

                // 2a. Calculate the work group's carry-in value.  
                // TODO: currently done serially but expected to be fast assuming n >> max_segment_size.
                // performance expected to degrade if very few segments.
                if (local_id == 0 && wg_agg_idx >= 0)
                {
                    for (int i = wg_agg_idx; i >= 0; --i) 
                    {
                        const auto& wg_aggregate = partials_acc[i];
                        const auto& b_seg_end = seg_ends_acc[i];
                        agg_collector = __binary_op(wg_aggregate, agg_collector);

                        // current aggregate is the last aggregate 
                        if (b_seg_end)
                            break;
                    }
                }
                
                agg_collector = sycl::group_broadcast(group, agg_collector);

                // 2c. Second pass over the keys, reidentifying end segments and applying work group
                // aggregates if appropriate. Both the key and reduction value are written to the final output
                int start = global_id * __vals_per_item;
                int end = sycl::minimum{}(start + __vals_per_item, __n);

                int local_min_key_idx = __n - 1;
                
                // find the smallest index 
                for (int i = start; i < end; ++i) {
                    if (i == __n - 1 || __keys[i] != __keys[i+1])
                    {
                        if (i < local_min_key_idx)
                            local_min_key_idx = i;
                    }
                }

                int wg_min_seg_end = sycl::reduce_over_group(group, local_min_key_idx, sycl::minimum<>());

                int item_offset = 0;
                for (int i = start; i < sycl::minimum{}(wg_min_seg_end, end); ++i)
                    __out_values[i] = __binary_op(agg_collector, __out_values[i]);
            });
        }).wait();
    }
};

} // end namespace internal
} // end namespace dpl
} // end namespace oneapi
#endif
#endif
