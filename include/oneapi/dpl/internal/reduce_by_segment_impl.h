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

#ifndef _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H
#define _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H

#include "../pstl/iterator_impl.h"
#include "function.h"
#include "by_segment_extension_defs.h"
#include "../pstl/utils.h"
#if _ONEDPL_BACKEND_SYCL
#    include "../pstl/utils_ranges.h"
#    include "../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#    include "../pstl/ranges_defs.h"
#    include "../pstl/glue_algorithm_ranges_defs.h"
#    include "../pstl/glue_algorithm_ranges_impl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename Name>
class Reduce1;
template <typename Name>
class Reduce2;
template <typename Name>
class Reduce3;
template <typename Name>
class Reduce4;

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_host_execution_policy<typename ::std::decay<Policy>::type,
                                                           ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_segment_impl(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                       OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred,
                       BinaryOperator binary_op)
{
    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    const auto n = ::std::distance(first1, last1);

    if (n <= 0)
        return ::std::make_pair(result1, result2);
    else if (n == 1)
    {
        *result1 = *first1;
        *result2 = *first2;
        return ::std::make_pair(result1 + 1, result2 + 1);
    }

    typedef uint64_t FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef uint64_t CountType;
    typedef typename ::std::decay<Policy>::type policy_type;

    // buffer that is used to store a flag indicating if the associated key is not equal to
    // the next key, and thus its associated sum should be part of the final result
    oneapi::dpl::__par_backend::__buffer<policy_type, FlagType> _mask(n + 1);
    auto mask = _mask.get();
    mask[0] = 1;

    // instead of copying mask, use shifted sequence:
    mask[n] = 1;

    // Identify where the first key in a sequence of equivalent keys is located
    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPred>(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    oneapi::dpl::__par_backend::__buffer<policy_type, ValueType> _scanned_values(n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    oneapi::dpl::__par_backend::__buffer<policy_type, FlagType> _scanned_tail_flags(n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    typename internal::rebind_policy<policy_type, Reduce1<policy_type>>::type policy1(policy);
    inclusive_scan(policy1, make_zip_iterator(first2, _mask.get()), make_zip_iterator(first2, _mask.get()) + n,
                   make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
                   internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op));

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indices each segment sum should be written
    typename internal::rebind_policy<policy_type, Reduce2<policy_type>>::type policy2(policy);
    oneapi::dpl::exclusive_scan(policy2, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                                ::std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    auto scanned_tail_flags = _scanned_tail_flags.get();
    auto scanned_values = _scanned_values.get();

    // number of unique segments
    CountType N = scanned_tail_flags[n - 1] + 1;

    // scatter the keys and accumulated values
    typename internal::rebind_policy<policy_type, Reduce3<policy_type>>::type policy3(policy);
    oneapi::dpl::for_each(policy3, make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1),
                          make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1) + n,
                          internal::scatter_and_accumulate_fun<OutputIterator1, OutputIterator2>(result1, result2));

    // for example: result1 = {1, 2, 3, 4, 1, 3, 1, 3, 0}
    // for example: result2 = {1, 2, 3, 4, 2, 6, 2, 6, 0}

    return ::std::make_pair(result1 + N, result2 + N);
}

#if _ONEDPL_BACKEND_SYCL

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range3>>
sycl_reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                       _Range4&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
    using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
    using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;
    using __flag_type = bool;

    const auto __n = __keys.size();

    constexpr int __vals_per_item = 2; // Each work item serially processes 2 items. Best observered performance on gpu

    ::std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__exec);

    // adjust __wgroup_size according to local memory limit
    __wgroup_size = oneapi::dpl::__internal::__max_local_allocation_size(
        ::std::forward<_ExecutionPolicy>(__exec), sizeof(__key_type) + sizeof(__val_type), __wgroup_size);

    int __n_groups = 1 + std::ceil(__n / (__wgroup_size * __vals_per_item));

    // intermediate reductions within a workgroup
    auto __partials =
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, __val_type>(__exec, __n_groups)
            .get_buffer();

    auto __end_idx =
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int>(__exec, 1).get_buffer();

    // the number of segment ends found in each work group
    auto __seg_ends =
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, int>(__exec, __n_groups).get_buffer();

    // 1. Count the segment ends in each workgroup
    auto __seg_end_identification = __exec.queue().submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __keys);
        auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::write>(__cgh);

        __cgh.parallel_for(sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
            auto __group = __item.get_group();
            auto __group_id = __group.get_group_id();
            int __local_id = __group.get_local_id();
            int __global_id = __item.get_global_id();

            int __start = __global_id * __vals_per_item;
            int __end = sycl::minimum{}(__start + __vals_per_item, __n);
            int __item_segments = 0;

            // 1a. Work item scan to identify segment ends
            for (int32_t __i = __start; __i < __end; ++__i)
                if (__n - 1 == __i || __keys[__i] != __keys[__i + 1])
                    ++__item_segments;

            // 1b. Work group reduction
            auto __num_segs = sycl::reduce_over_group(__group, __item_segments, sycl::plus<>());

            // 1c. First work item writes segment count to global memory
            if (__local_id == 0)
                __seg_ends_acc[__group_id] = __num_segs;
        });
    });

    // 2. Work group reduction
    auto __wg_reduce = __exec.queue().submit([&](sycl::handler& __cgh) {
        __cgh.depends_on(__seg_end_identification);
        oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_keys, __out_values, __values);

        auto __partials_acc = __partials.template get_access<sycl::access_mode::read_write>(__cgh);
        auto __end_idx_acc = __end_idx.template get_access<sycl::access_mode::write>(__cgh);
        auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);
        auto __loc_acc = sycl::accessor<__val_type, 1, sycl::access::mode::read_write, sycl::access::target::local>{
            2 * __wgroup_size, __cgh};

        __cgh.parallel_for(sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1> __item) {
            ::std::array<__val_type, __vals_per_item> __loc_partials;

            auto __group = __item.get_group();
            auto __group_id = __group.get_group_id();
            auto __local_id = __group.get_local_id();
            auto __global_id = __item.get_global_id();

            __val_type __accumulator{};

            // 2a. Compute the number of segments in prior workgroups. Do this collectively in
            // subgroups to eliminate barriers.
            auto __start_ptr = __seg_ends_acc.get_pointer();
            auto __end_ptr = __start_ptr + __group_id;

            int32_t __wg_num_prior_segs =
                sycl::joint_reduce(__item.get_sub_group(), __start_ptr, __end_ptr, sycl::plus<>());

            // 2b. Perform a serial scan within the work item over assigned elements. Store partial
            // reductions in work group local memory.
            int32_t __start = __global_id * __vals_per_item;
            int32_t __end = sycl::minimum{}(__start + __vals_per_item, __n);

            int32_t __max_end = 0;
            int32_t __item_segments = 0;

            bool __first = true;
            for (int32_t __i = __start; __i < __end; ++__i)
            {
                if (__first)
                {
                    __accumulator = __values[__i];
                    __first = false;
                }

                else
                    __accumulator = __binary_op(__accumulator, __values[__i]);

                // clear the accumulator if we reach end of segment
                if (__n - 1 == __i || __keys[__i] != __keys[__i + 1])
                {
                    __loc_partials[__i - __start] = __accumulator;
                    __accumulator = {};
                    ++__item_segments;

                    __max_end = __local_id;
                    __first = true;
                }
            }

            // 2c. Count the number of prior work segments cooperatively over group
            int __prior_segs_in_wg = sycl::exclusive_scan_over_group(__group, __item_segments, sycl::plus<>());
            auto __start_idx = __wg_num_prior_segs + __prior_segs_in_wg;

            // 2d. Find the greatest segment end less than the current index (inclusive)
            auto __closest_seg_id = sycl::inclusive_scan_over_group(__group, __max_end, sycl::maximum<>());

            __val_type __carry_in =
                oneapi::dpl::internal::wg_segmented_scan(__item, __loc_acc, __local_id, __local_id - __closest_seg_id,
                                                         __accumulator, __binary_op, __wgroup_size);

            // 2e. Update local partial reductions in first segment and write to global memory.
            // double check edge cases for applying partials
            bool __apply_aggs = true;
            int __item_offset = 0;

            // first item in group does not have any work-group aggregates to apply
            if (__local_id == 0)
                __apply_aggs = false;

            for (int32_t __i = __start; __i < __end; ++__i)
            {
                if (__i == __n - 1 || __keys[__i] != __keys[__i + 1])
                {
                    auto __idx = __start_idx + __item_offset;
                    if (__apply_aggs)
                    {
                        __out_values[__idx] = __binary_op(__carry_in, __loc_partials[__i - __start]);
                        __apply_aggs = false;
                    }
                    else
                        __out_values[__idx] = __loc_partials[__i - __start];

                    __out_keys[__idx] = __keys[__i];

                    // the last item must write the last index's position to return
                    if (__i == __n - 1)
                        __end_idx_acc[0] = __idx;

                    ++__item_offset;
                }
            }

            // 2e. Output the work group aggregate and total number of segments for use in phase 2.
            if (__local_id == __wgroup_size - 1) // last work item writes the group's carry out
            {
                // If no segment ends in the item, the aggregates from previous work groups must be applied.
                if (__max_end == 0)
                    __partials_acc[__group_id] =
                        __binary_op(__carry_in, __accumulator); // needs to be inclusive with last element

                else
                    __partials_acc[__group_id] = __accumulator;
            }
        });
    });

    // 3. Apply inter work-group aggregates
    __exec.queue()
        .submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys, __out_keys, __out_values);

            auto __partials_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
            auto __seg_ends_acc = __seg_ends.template get_access<sycl::access_mode::read>(__cgh);

            __cgh.depends_on(__wg_reduce);

            __cgh.parallel_for(sycl::nd_range<1>{__n_groups * __wgroup_size, __wgroup_size}, [=](sycl::nd_item<1>
                                                                                                     __item) {
                auto __group = __item.get_group();
                auto __group_id = __group.get_group_id(0);
                auto __global_id = __item.get_global_id();
                auto __local_id = __item.get_local_id();

                int32_t __start = __global_id * __vals_per_item;
                int32_t __end = sycl::minimum{}(__start + __vals_per_item, __n);
                int32_t __item_segments = 0;

                int32_t __wg_agg_idx = __group_id - 1;
                __val_type __agg_collector{};

                // 3a. Find the work group's carry-in value.
                bool __first = true;
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

                // 3b. count the segment ends
                for (auto __i = __start; __i < __end; ++__i)
                    if (__i == __n - 1 || __keys[__i] != __keys[__i + 1])
                        ++__item_segments;

                auto __prior_segs_in_wg = sycl::exclusive_scan_over_group(__group, __item_segments, sycl::plus<>());

                // 3c. Collectively perform a subgroup reduction to determine the first index
                // the work group will write to.
                auto __start_ptr = __seg_ends_acc.get_pointer();
                auto __end_ptr = __start_ptr + __group_id;

                auto __wg_num_prior_segs =
                    sycl::joint_reduce(__item.get_sub_group(), __start_ptr, __end_ptr, sycl::plus<>());

                // 3d. Second pass over the keys, reidentifying end segments and applying work group
                // aggregates if appropriate. Both the key and reduction value are written to the final output at the
                // computed index
                int __item_offset = 0;
                for (int32_t __i = __start; __i < __end; ++__i)
                {
                    if (__i == __n - 1 || __keys[__i] != __keys[__i + 1])
                    {
                        int __idx = __wg_num_prior_segs + __prior_segs_in_wg + __item_offset;

                        // apply the aggregate if it is the first segment end in the workgroup only
                        if (__prior_segs_in_wg == 0 && __item_offset == 0)
                            __out_values[__idx] = __binary_op(__agg_collector, __out_values[__idx]);

                        ++__item_offset;
                    }
                }
            });
        })
        .wait();

    return 1 + __end_idx.template get_access<sycl::access_mode::read>()[0];
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
typename ::std::enable_if<
    oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value,
    ::std::pair<OutputIterator1, OutputIterator2>>::type
reduce_by_segment_impl(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                       OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred,
                       BinaryOperator binary_op)
{
    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    typedef uint64_t CountType;

    namespace __bknd = __par_backend_hetero;

    const auto n = ::std::distance(first1, last1);

    if (n == 0)
        return ::std::make_pair(result1, result2);

    auto keep_keys = __ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto key_buf = keep_keys(first1, last1);
    auto keep_values = __ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(first2, first2 + n);
    auto keep_key_outputs = __ranges::__get_sycl_range<__bknd::access_mode::write, OutputIterator1>();
    auto key_output_buf = keep_key_outputs(result1, result1 + n);
    auto keep_value_outputs = __ranges::__get_sycl_range<__bknd::access_mode::write, OutputIterator2>();
    auto value_output_buf = keep_value_outputs(result2, result2 + n);

    // number of unique keys
    CountType N =
        sycl_reduce_by_segment(::std::forward<Policy>(policy), key_buf.all_view(), value_buf.all_view(),
                               key_output_buf.all_view(), value_output_buf.all_view(), binary_pred, binary_op);
    return ::std::make_pair(result1 + N, result2 + N);
}
#endif
} // namespace internal

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                  OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op)
{
    return internal::reduce_by_segment_impl(::std::forward<Policy>(policy), first1, last1, first2, result1, result2,
                                            binary_pred, binary_op);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2,
          typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_segment(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1,
                  OutputIt2 result2, BinaryPred binary_pred)
{
    typedef typename ::std::iterator_traits<InputIt2>::value_type T;

    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred,
                             ::std::plus<T>());
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_segment(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1,
                  OutputIt2 result2)
{
    typedef typename ::std::iterator_traits<InputIt1>::value_type T;

    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2,
                             ::std::equal_to<T>());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
              OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op)
{
    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred,
                             binary_op);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2,
          typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_key(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1, OutputIt2 result2,
              BinaryPred binary_pred)
{
    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIt1, OutputIt2>>
reduce_by_key(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1, OutputIt2 result2)
{
    return reduce_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result1, result2);
}
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H
