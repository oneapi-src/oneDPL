/*
 *  Copyright (c) 2019-2020 Intel Corporation
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

#ifndef DPSTD_REDUCE_BY_KEY
#define DPSTD_REDUCE_BY_KEY

#include "oneapi/dpl/iterator"
#include "function.h"
#include "by_segment_extension_defs.h"

namespace oneapi
{
namespace dpl
{
namespace internal
{

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
    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1, ::std::not2(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    oneapi::dpl::__par_backend::__buffer<policy_type, FlagType> _scanned_values(n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    oneapi::dpl::__par_backend::__buffer<policy_type, FlagType> _scanned_tail_flags(n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    typename internal::rebind_policy<policy_type, class ReduceByKey1>::type policy1(policy);
    inclusive_scan(policy1, make_zip_iterator(first2, _mask.get()), make_zip_iterator(first2, _mask.get()) + n,
                   make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
                   internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op));

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indicies each segment sum should be written
    typename internal::rebind_policy<policy_type, class ReduceByKey2>::type policy2(policy);
    ::std::exclusive_scan(policy2, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                          ::std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    auto scanned_tail_flags = _scanned_tail_flags.get();
    auto scanned_values = _scanned_values.get();

    // number of unique segments
    CountType N = scanned_tail_flags[n - 1] + 1;

    // scatter the keys and accumulated values
    typename internal::rebind_policy<policy_type, class ReduceByKey3>::type policy3(policy);
    oneapi::dpl::for_each(policy3, make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1),
                    make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1) + n,
                    internal::scatter_and_accumulate_fun<OutputIterator1, OutputIterator2>(result1, result2));

    // for example: result1 = {1, 2, 3, 4, 1, 3, 1, 3, 0}
    // for example: result2 = {1, 2, 3, 4, 2, 6, 2, 6, 0}

    return ::std::make_pair(result1 + N, result2 + N);
}

#if _PSTL_BACKEND_SYCL
template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
typename ::std::enable_if<
    oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
    !oneapi::dpl::internal::is_discard_iterator<OutputIterator1>::value &&
    !oneapi::dpl::internal::is_discard_iterator<OutputIterator2>::value,
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

    typedef uint64_t FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef uint64_t CountType;
    typedef typename ::std::decay<Policy>::type policy_type;

    ValueType initial_value;
    {
        auto first2_acc = internal::get_access<cl::sycl::access::mode::read>(first2);
        initial_value = first2_acc[0];
    }

    const auto n = ::std::distance(first1, last1);
    if (n <= 0)
        return ::std::make_pair(result1, result2);
    else if (n == 1)
    {
        auto result1_acc = internal::get_access<cl::sycl::access::mode::write>(result1);
        auto result2_acc = internal::get_access<cl::sycl::access::mode::write>(result2);
        auto first1_acc = internal::get_access<cl::sycl::access::mode::read>(first1);
        result1_acc[0] = first1_acc[0];
        result2_acc[0] = initial_value;
        return ::std::make_pair(result1 + 1, result2 + 1);
    }

    FlagType initial_mask = 1;

    // buffer that is used to store a flag indicating if the associated key is not equal to
    // the next key, and thus its associated sum should be part of the final result
    internal::__buffer<policy_type, FlagType> _mask(policy, n + 1);
    {
        auto mask_buf = _mask.get_buffer();
        auto mask = mask_buf.template get_access<cl::sycl::access::mode::write>();
        mask[0] = initial_mask;

        // instead of copying mask, use shifted sequence:
        mask[n] = initial_mask;
    }

    // Identify where the first key in a sequence of equivalent keys is located
    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1, ::std::not2(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    internal::__buffer<policy_type, FlagType> _scanned_values(policy, n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    internal::__buffer<policy_type, FlagType> _scanned_tail_flags(policy, n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    typename internal::rebind_policy<policy_type, class ReduceByKey1>::type policy1(policy);
    transform_inclusive_scan(policy1, make_zip_iterator(first2, _mask.get()),
                             make_zip_iterator(first2, _mask.get()) + n,
                             make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
                             internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op),
                             oneapi::dpl::__internal::__no_op(), ::std::make_tuple(initial_value, initial_mask));

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indicies each segment sum should be written
    typename internal::rebind_policy<policy_type, class ReduceByKey2>::type policy2(policy);
    oneapi::dpl::exclusive_scan(policy2, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                          ::std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    // number of unique keys
    CountType N = 0;

    // scoped sections enforce that there is only one access to the scanned_tail_flags
    // buffer alive at any point in time.
    {
        auto scanned_tail_flags_buf = _scanned_tail_flags.get_buffer();
        auto scanned_tail_flags = scanned_tail_flags_buf.template get_access<cl::sycl::access::mode::read>();

        N = scanned_tail_flags[n - 1] + 1;
    }

    // scatter the keys and accumulated values
    typename internal::rebind_policy<policy_type, class ReduceByKey3>::type policy3(policy);
    typename internal::rebind_policy<policy_type, class ReduceByKey4>::type policy4(policy);

    // permutation iterator reorders elements in result1 so the element at index
    // _scanned_tail_flags[i] is returned when index i of the iterator is accessed. The result
    // is that some elements of result are written multiple times.
    {
        auto permute_it1 = make_permutation_iterator(result1, _scanned_tail_flags.get());
        oneapi::dpl::for_each(policy3, make_zip_iterator(first1, _mask.get() + 1, permute_it1),
                        make_zip_iterator(last1, _mask.get() + 1 + n, permute_it1 + n),
                        internal::transform_if_stencil_fun<FlagType, identity>(identity()));
    }

    {
        auto permute_it2 = make_permutation_iterator(result2, _scanned_tail_flags.get());
        oneapi::dpl::for_each(policy4, make_zip_iterator(_scanned_values.get(), _mask.get() + 1, permute_it2),
                        make_zip_iterator(_scanned_values.get() + n, _mask.get() + 1 + n, permute_it2 + n),
                        internal::transform_if_stencil_fun<FlagType, identity>(identity()));
    }
    // for example: result1 = {1, 2, 3, 4, 1, 3, 1, 3, 0}
    // for example: result2 = {1, 2, 3, 4, 2, 6, 2, 6, 0}

    return ::std::make_pair(result1 + N, result2 + N);
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
typename ::std::enable_if<
    oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
    oneapi::dpl::internal::is_discard_iterator<OutputIterator1>::value &&
    !oneapi::dpl::internal::is_discard_iterator<OutputIterator2>::value,
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

    typedef uint64_t FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef uint64_t CountType;
    typedef typename ::std::decay<Policy>::type policy_type;

    ValueType initial_value;
    {
        auto first2_acc = internal::get_access<cl::sycl::access::mode::read>(first2);
        initial_value = first2_acc[0];
    }

    const auto n = ::std::distance(first1, last1);
    if (n <= 0)
        return ::std::make_pair(result1, result2);
    else if (n == 1)
    {
        auto result1_acc = internal::get_access<cl::sycl::access::mode::write>(result1);
        auto result2_acc = internal::get_access<cl::sycl::access::mode::write>(result2);
        auto first1_acc = internal::get_access<cl::sycl::access::mode::read>(first1);
        result1_acc[0] = first1_acc[0];
        result2_acc[0] = initial_value;
        return ::std::make_pair(result1 + 1, result2 + 1);
    }

    FlagType initial_mask = 1;

    // buffer that is used to store a flag indicating if the associated key is not equal to
    // the next key, and thus its associated sum should be part of the final result
    internal::__buffer<policy_type, FlagType> _mask(policy, n + 1);
    {
        auto mask_buf = _mask.get_buffer();
        auto mask = mask_buf.template get_access<cl::sycl::access::mode::write>();
        mask[0] = initial_mask;

        // instead of copying mask, use shifted sequence:
        mask[n] = initial_mask;
    }

    // Identify where the first key in a sequence of equivalent keys is located
    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1, ::std::not2(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    internal::__buffer<policy_type, FlagType> _scanned_values(policy, n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    internal::__buffer<policy_type, FlagType> _scanned_tail_flags(policy, n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    typename internal::rebind_policy<policy_type, class ReduceByKey1>::type policy1(policy);
    transform_inclusive_scan(policy1, make_zip_iterator(first2, _mask.get()),
                             make_zip_iterator(first2, _mask.get()) + n,
                             make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
                             internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op),
                             oneapi::dpl::__internal::__no_op(), ::std::make_tuple(initial_value, initial_mask));

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indicies each segment sum should be written
    typename internal::rebind_policy<policy_type, class ReduceByKey2>::type policy2(policy);
    oneapi::dpl::exclusive_scan(policy2, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                          ::std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    // number of unique keys
    CountType N = 0;

    // scoped sections enforce that there is only one access to the scanned_tail_flags
    // buffer alive at any point in time.
    {
        auto scanned_tail_flags_buf = _scanned_tail_flags.get_buffer();
        auto scanned_tail_flags = scanned_tail_flags_buf.template get_access<cl::sycl::access::mode::read>();

        N = scanned_tail_flags[n - 1] + 1;
    }

    // scatter the keys and accumulated values
    typename internal::rebind_policy<policy_type, class ReduceByKey3>::type policy3(policy);
    typename internal::rebind_policy<policy_type, class ReduceByKey4>::type policy4(policy);

    // result1 is a discard_iterator instance so we omit the write to it.

    // permutation iterator reorders elements in result2 so the element at index
    // _scanned_tail_flags[i] is returned when index i of the iterator is accessed. The result
    // is that some elements of result are written multiple times.
    {
        auto permute_it2 = make_permutation_iterator(result2, _scanned_tail_flags.get());
        oneapi::dpl::for_each(policy4, make_zip_iterator(_scanned_values.get(), _mask.get() + 1, permute_it2),
                        make_zip_iterator(_scanned_values.get() + n, _mask.get() + 1 + n, permute_it2 + n),
                        internal::transform_if_stencil_fun<FlagType, identity>(identity()));
    }
    // for example: result2 = {1, 2, 3, 4, 2, 6, 2, 6, 0}

    return ::std::make_pair(result1 + N, result2 + N);
}
template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
typename ::std::enable_if<
    oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
    !oneapi::dpl::internal::is_discard_iterator<OutputIterator1>::value &&
    oneapi::dpl::internal::is_discard_iterator<OutputIterator2>::value,
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

    typedef uint64_t FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef uint64_t CountType;
    typedef typename ::std::decay<Policy>::type policy_type;

    ValueType initial_value;
    {
        auto first2_acc = internal::get_access<cl::sycl::access::mode::read>(first2);
        initial_value = first2_acc[0];
    }

    const auto n = ::std::distance(first1, last1);
    if (n <= 0)
        return ::std::make_pair(result1, result2);
    else if (n == 1)
    {
        auto result1_acc = internal::get_access<cl::sycl::access::mode::write>(result1);
        auto result2_acc = internal::get_access<cl::sycl::access::mode::write>(result2);
        auto first1_acc = internal::get_access<cl::sycl::access::mode::read>(first1);
        result1_acc[0] = first1_acc[0];
        result2_acc[0] = initial_value;
        return ::std::make_pair(result1 + 1, result2 + 1);
    }

    FlagType initial_mask = 1;

    // buffer that is used to store a flag indicating if the associated key is not equal to
    // the next key, and thus its associated sum should be part of the final result
    internal::__buffer<policy_type, FlagType> _mask(policy, n + 1);
    {
        auto mask_buf = _mask.get_buffer();
        auto mask = mask_buf.template get_access<cl::sycl::access::mode::write>();
        mask[0] = initial_mask;

        // instead of copying mask, use shifted sequence:
        mask[n] = initial_mask;
    }

    // Identify where the first key in a sequence of equivalent keys is located
    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1, ::std::not2(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    internal::__buffer<policy_type, FlagType> _scanned_values(policy, n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    internal::__buffer<policy_type, FlagType> _scanned_tail_flags(policy, n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    typename internal::rebind_policy<policy_type, class ReduceByKey1>::type policy1(policy);
    transform_inclusive_scan(policy1, make_zip_iterator(first2, _mask.get()),
                             make_zip_iterator(first2, _mask.get()) + n,
                             make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
                             internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op),
                             oneapi::dpl::__internal::__no_op(), ::std::make_tuple(initial_value, initial_mask));

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indicies each segment sum should be written
    typename internal::rebind_policy<policy_type, class ReduceByKey2>::type policy2(policy);
    oneapi::dpl::exclusive_scan(policy2, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                          ::std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    // number of unique keys
    CountType N = 0;

    // scoped sections enforce that there is only one access to the scanned_tail_flags
    // buffer alive at any point in time.
    {
        auto scanned_tail_flags_buf = _scanned_tail_flags.get_buffer();
        auto scanned_tail_flags = scanned_tail_flags_buf.template get_access<cl::sycl::access::mode::read>();

        N = scanned_tail_flags[n - 1] + 1;
    }

    // scatter the keys and accumulated values
    typename internal::rebind_policy<policy_type, class ReduceByKey3>::type policy3(policy);
    typename internal::rebind_policy<policy_type, class ReduceByKey4>::type policy4(policy);

    // permutation iterator reorders elements in result1 so the element at index
    // _scanned_tail_flags[i] is returned when index i of the iterator is accessed. The result
    // is that some elements of result are written multiple times.
    {
        auto permute_it1 = make_permutation_iterator(result1, _scanned_tail_flags.get());
        oneapi::dpl::for_each(policy3, make_zip_iterator(first1, _mask.get() + 1, permute_it1),
                        make_zip_iterator(last1, _mask.get() + 1 + n, permute_it1 + n),
                        internal::transform_if_stencil_fun<FlagType, identity>(identity()));
    }

    // result2 is a discard_iterator instance so we omit the write to it.

    // for example: result1 = {1, 2, 3, 4, 1, 3, 1, 3, 0}

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

namespace dpstd
{
using oneapi::dpl::reduce_by_key;
using oneapi::dpl::reduce_by_segment;
} // end namespace dpstd
#endif
