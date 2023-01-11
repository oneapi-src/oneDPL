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

#    include <oneapi/dpl/pstl/algorithm_fwd.h>
#    include <oneapi/dpl/pstl/parallel_backend.h>
#    include <oneapi/dpl/pstl/hetero/utils_hetero.h>

#    include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>
#    include <oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h>
#    include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>

#    include <array>

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename _NdItem, typename _LocalAcc, typename _IdxType, typename _ValueType, typename _BinaryOp>
inline _ValueType
wg_segmented_scan(_NdItem __item, _LocalAcc __local_acc, _IdxType __local_id, _IdxType __delta_local_id,
                  _ValueType __accumulator, _BinaryOp __binary_op, ::std::size_t __wgroup_size)
{
    _IdxType __first = 0;
    __local_acc[__local_id] = __accumulator;

    __dpl_sycl::__group_barrier(__item);

    for (::std::size_t __i = 1; __i < __wgroup_size; __i += __i)
    {
        if (__delta_local_id >= __i)
            __accumulator = __binary_op(__local_acc[__first + __local_id - __i], __accumulator);

        __first = __wgroup_size - __first;
        __local_acc[__first + __local_id] = __accumulator;

        __dpl_sycl::__group_barrier(__item);
    }

    return (__local_id ? __local_acc[__first + __local_id - 1] : 0);
}

enum class scan_type
{
    inclusive,
    exclusive
};

template <scan_type __scan_type>
struct sycl_scan_by_segment_impl
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
              typename _BinaryPredicate, typename _BinaryOperator, typename _T>
    void
    sycl_scan_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_values,
                         _BinaryPredicate __binary_pred, _BinaryOperator __binary_op, _T __init, _T __identity)
    {
        using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
        using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
        using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;
        using __flag_type = bool;

        const ::std::size_t __n = __keys.size();

        constexpr ::std::size_t __vals_per_item =
            4; // Assigning 4 elements per work item resulted in best performance on gpu.

        ::std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);

        // change __wgroup_size according to local memory limit
        __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size(::std::forward<_ExecutionPolicy>(__exec),
                                                                             2 * sizeof(__val_type), __wgroup_size);

        ::std::size_t __n_groups = 1 + ((__n - 1) / (__wgroup_size * __vals_per_item));

        auto __partials =
            oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups)
                .get_buffer();

        // the number of segment ends found in each work group
        auto __seg_ends =
            oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __flag_type>(__exec, __n_groups)
                .get_buffer();

        // 1. Work group reduction
        auto __wg_scan = __exec.queue().submit([&](sycl::handler& __cgh) {
            auto __partials_acc = __partials.template get_access<sycl::access_mode::write>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::write>(__cgh);

            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __values, __out_values);

            auto __loc_acc = sycl::accessor<__val_type, 1, sycl::access::mode::read_write, sycl::access::target::local>{
                2 * __wgroup_size, __cgh};

            __cgh.parallel_for(
                sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                    __val_type __accumulator = __identity;

                    auto __group = __item.get_group();
                    ::std::size_t __global_id = __item.get_global_id();
                    ::std::size_t __local_id = __group.get_local_id();

                    // 1a. Perform a serial scan within the work item over assigned elements. Store partial
                    // reductions in work item memory, and write the accumulated value and number of counted
                    // segments into work group memory.
                    ::std::size_t __start = __global_id * __vals_per_item;
                    ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);

                    // First work item must set their accumulator to the provided init
                    if (__global_id == 0 && __local_id == 0)
                    {
                        __accumulator = __init;
                    }

                    const int64_t NO_SEGMENT_BREAK = -1;
                    // signed to allow flag for no segment break found
                    int64_t __max_end = NO_SEGMENT_BREAK;

                    for (::std::size_t __i = __start; __i < __end; ++__i)
                    {
                        if constexpr (__scan_type == scan_type::inclusive)
                        {
                            __accumulator = __binary_op(__accumulator, __values[__i]);

                            __out_values[__i] = __accumulator;
                        }
                        else // exclusive scan
                        {
                            __out_values[__i] = __accumulator;
                            __accumulator = __binary_op(__accumulator, __values[__i]);
                        }

                        // reset the accumulator to init if we reach the end of a segment
                        // (init for inclusive scan is the identity)
                        if (__n - 1 == __i || !__binary_pred(__keys[__i], __keys[__i + 1]))
                        {
                            __accumulator = __init;
                            __max_end = __local_id;
                        }
                    }

                    // 1b. Perform a work group scan to find the carry in value to apply to each item.
                    int64_t __closest_seg_id = __dpl_sycl::__inclusive_scan_over_group(
                        __group, __max_end, __dpl_sycl::__maximum<decltype(__max_end)>());

                    __flag_type __group_has_segment_break = (__closest_seg_id != NO_SEGMENT_BREAK);

                    __closest_seg_id = ::std::max(0L, __closest_seg_id); //get rid of no segment end found flag
                    __val_type __carry_in = wg_segmented_scan(__item, __loc_acc, __local_id,
                                                              __local_id - __closest_seg_id, __accumulator, __binary_op,
                                                              __wgroup_size); // need to use exclusive scan delta

                    // 1c. Update local partial reductions and write to global memory.
                    for (::std::size_t __i = __start; __i < __end; ++__i)
                    {
                        __out_values[__i] = __binary_op(__carry_in, __out_values[__i]);

                        if (__i >= __n - 1 || !__binary_pred(__keys[__i], __keys[__i + 1]))
                            break;
                    }

                    if (__local_id == __wgroup_size - 1) // last work item writes the group's carry out
                    {
                        ::std::size_t __group_id = __group.get_group_id(0);

                        __seg_ends_acc[__group_id] = __group_has_segment_break;

                        if (__max_end == NO_SEGMENT_BREAK)
                        {
                            __partials_acc[__group_id] = __binary_op(__carry_in, __accumulator);
                        }
                        else
                        {
                            __partials_acc[__group_id] = __accumulator;
                        }
                    }
                });
        });

        // 2. Apply work group carry outs, calculate output indices, and load results into correct indices.
        __exec.queue()
            .submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_values);

                auto __partials_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
                auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);
                
                __cgh.depends_on(__wg_scan);

                auto __loc_partials_acc = sycl::accessor<__val_type, 1, sycl::access::mode::read_write, sycl::access::target::local>{
                     __wgroup_size, __cgh};
                
                auto __loc_seg_ends_acc = sycl::accessor<__flag_type, 1, sycl::access::mode::read_write, sycl::access::target::local>{
                     __wgroup_size, __cgh};

                __cgh.parallel_for(
                    sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
                        auto __group = __item.get_group();
                        ::std::size_t __group_id = __group.get_group_id(0);
                        ::std::size_t __global_id = __item.get_global_id();
                        ::std::size_t __local_id = __item.get_local_id();
                        ::std::size_t __start = __global_id * __vals_per_item;
                        ::std::size_t __end = __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n);

                        ::std::size_t __wg_agg_idx = __group_id - 1;
                        __val_type __agg_collector = __identity;

                        //TODO:  just launch with one fewer group and adjust indexing since group zero can skip phase
                        if (__group_id != 0)
                        {
                            // 2a. Calculate the work group's carry-in value.
                            bool __ag_exists = false;
                            if (__start < __n)
                            {
                                __ag_exists = true;
                            }
                            // local reductions followed by a downsweep
                            // TODO: Generalize this value
                            constexpr int64_t __vals_to_explore = 16;
                            __flag_type  __last_it = false;
                            __loc_seg_ends_acc[__local_id] = false;
                            __loc_partials_acc[__local_id] = __identity; 

                            for (int64_t __i = __wg_agg_idx - __vals_to_explore * __local_id; ; __i -= __wgroup_size * __vals_to_explore)
                            {
                                __val_type __local_collector = __identity; 
                                // exploration phase 
                                for (int64_t __j = __i; __j > __dpl_sycl::__maximum<int64_t>{}(-1L, __i - __vals_to_explore); --__j)
                                {
                                    __local_collector = __binary_op(__partials_acc[__j], __local_collector);
                                    if (__seg_ends_acc[__j] || __j == 0)
                                    {
                                        __loc_seg_ends_acc[__local_id] = true;
                                        break;
                                    }
                                }
                                __loc_partials_acc[__local_id] = __local_collector;
                                __dpl_sycl::__group_barrier(__item);
                                // serial aggregate collection and synchronization
                                if (__local_id == 0)
                                {
                                    for (int64_t __j = 0; __j < __wgroup_size; ++__j)
                                    {
                                        __agg_collector = __binary_op(__loc_partials_acc[__j], __agg_collector);
                                        if (__loc_seg_ends_acc[__j])
                                        {
                                            __last_it = true;
                                            break;
                                        }
                                    }
                                }
                                __agg_collector = __dpl_sycl::__group_broadcast(__item.get_group(), __agg_collector);
                                __last_it = __dpl_sycl::__group_broadcast(__item.get_group(), __last_it);
                                if (__last_it)
                                    break;
                            }
                            __ag_exists = __dpl_sycl::__any_of_group(__group, __ag_exists);

                            if (!__ag_exists)
                                return;

                            // 2c. Second pass over the keys, reidentifying end segments and applying work group
                            // aggregates if appropriate.

                            ::std::size_t __end_nm1_cap =
                                __dpl_sycl::__minimum<decltype(__n)>{}(__start + __vals_per_item, __n - 1);
                            ::std::size_t __local_min_key_idx = __n - 1;

                            // Find the smallest index in the work group
                            for (::std::size_t __i = __end_nm1_cap - 1; __i >= __start; --__i)
                            {
                                if (!__binary_pred(__keys[__i], __keys[__i + 1]))
                                {
                                    __local_min_key_idx = __i;
                                }
                            }

                            ::std::size_t __wg_min_seg_end = __dpl_sycl::__reduce_over_group(
                                __group, __local_min_key_idx, __dpl_sycl::__minimum<decltype(__local_min_key_idx)>());

                            // apply work group aggregates
                            for (::std::size_t __i = __start;
                                 __i < __dpl_sycl::__minimum<decltype(__end)>{}(__wg_min_seg_end + 1, __end); ++__i)
                            {
                                __out_values[__i] = __binary_op(__agg_collector, __out_values[__i]);
                            }
                        }
                    });
            })
            .wait();
    }

    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
              typename _BinaryPredicate, typename _BinaryOperator, typename T>
    void
    operator()(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_values,
               _BinaryPredicate __binary_pred, _BinaryOperator __binary_op, T __init, T __identity)
    {
        sycl_scan_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys),
                             ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_values), __binary_pred,
                             __binary_op, __init, __identity);
    }
}; // namespace internal

} // namespace internal
} // namespace dpl
} // namespace oneapi
#endif
#endif
