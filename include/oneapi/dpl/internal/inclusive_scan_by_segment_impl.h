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

#ifndef _ONEDPL_INCLUSIVE_SCAN_BY_SEGMENT_IMPL_H
#define _ONEDPL_INCLUSIVE_SCAN_BY_SEGMENT_IMPL_H

#include "by_segment_extension_defs.h"
#include "../pstl/glue_numeric_impl.h"
#include "../pstl/parallel_backend.h"
#include "function.h"
#include "../pstl/utils.h"

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename Name>
class InclusiveScan1;

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_host_execution_policy<typename ::std::decay<Policy>::type, OutputIterator>
inclusive_scan_by_segment_impl(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                               OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op)
{
    const auto n = ::std::distance(first1, last1);

    // Check for empty and single element ranges
    if (n <= 0)
        return result;
    if (n == 1)
    {
        *result = *first2;
        return result + 1;
    }

    typedef unsigned int FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef typename ::std::decay<Policy>::type policy_type;

    oneapi::dpl::__par_backend::__buffer<policy_type, FlagType> _mask(n);
    auto mask = _mask.get();

    mask[0] = 1;

    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPredicate>(binary_pred));

    typename internal::rebind_policy<policy_type, InclusiveScan1<policy_type>>::type policy1(policy);
    inclusive_scan(policy1, make_zip_iterator(first2, _mask.get()), make_zip_iterator(first2, _mask.get()) + n,
                   make_zip_iterator(result, _mask.get()),
                   internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op));

    return result + n;
}

#if _ONEDPL_BACKEND_SYCL
template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<typename ::std::decay<Policy>::type, OutputIterator>
inclusive_scan_by_segment_impl(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                               OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op)
{

    const auto n = ::std::distance(first1, last1);

    // Check for empty and single element ranges
    if (n <= 0)
        return result;

    namespace __bknd = oneapi::dpl::__par_backend_hetero;

    auto keep_keys = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto key_buf = keep_keys(first1, last1);
    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(first2, first2 + n);
    auto keep_value_outputs = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, OutputIterator>();
    auto value_output_buf = keep_value_outputs(result, result + n);
    auto buf_view = key_buf.all_view();

    auto identity = oneapi::dpl::__internal::__value_t<decltype(value_buf.all_view())>{};

    sycl_scan_by_segment_impl<scan_type::inclusive> scan;

    scan(::std::forward<Policy>(policy), key_buf.all_view(), value_buf.all_view(), value_output_buf.all_view(),
         binary_pred, binary_op, identity);

    return result + n;
}

#endif
} // namespace internal

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op)
{
    return internal::inclusive_scan_by_segment_impl(::std::forward<Policy>(policy), first1, last1, first2, result,
                                                    binary_pred, binary_op);
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter, typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_segment(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result,
                          BinaryPredicate binary_pred)
{
    using T = typename ::std::iterator_traits<InputIter2>::value_type;

    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred,
                                     ::std::plus<T>());
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_segment(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result)
{
    using T = typename ::std::iterator_traits<InputIter1>::value_type;

    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result,
                                     ::std::equal_to<T>());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                      OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op)
{
    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred,
                                     binary_op);
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter, typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_key(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result,
                      BinaryPredicate binary_pred)
{
    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred);
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_key(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result)
{
    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result);
}
} // end namespace dpl
} // end namespace oneapi

#endif
