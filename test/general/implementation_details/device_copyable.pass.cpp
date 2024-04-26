// -*- C++ -*-
//===-- device_copyable.pass.cpp ------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"
using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT

#    include "support/utils_device_copyable.h"
#    include "support/utils_sycl_defs.h"

void
test_device_copyable()
{
    //check that our testing types are non-trivially copyable but device copyable
    static_assert(!std::is_trivially_copy_constructible_v<int_device_copyable>,
                  "int_device_copyable is not trivially copy constructible");
    static_assert(!std::is_trivially_copy_constructible_v<noop_device_copyable>,
                  "noop_device_copyable is not trivially copy constructible");
    static_assert(!std::is_trivially_copy_constructible_v<constant_iterator_device_copyable>,
                  "constant_iterator_device_copyable is not trivially copy constructible");

    static_assert(sycl::is_device_copyable_v<int_device_copyable>, "int_device_copyable is not device copyable");
    static_assert(sycl::is_device_copyable_v<noop_device_copyable>, "noop_device_copyable is not device copyable");
    static_assert(sycl::is_device_copyable_v<constant_iterator_device_copyable>,
                  "constant_iterator_device_copyable is not device copyable");

    //custom_brick
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::internal::custom_brick<
                      noop_device_copyable, int_device_copyable, oneapi::dpl::internal::search_algorithm::lower_bound>>,
                  "custom_brick is not device copyable with device copyable types");
    //replace_if_fun
    static_assert(
        sycl::is_device_copyable_v<oneapi::dpl::internal::replace_if_fun<int_device_copyable, noop_device_copyable>>,
        "replace_if_fun is not device copyable with device copyable types");
    //scan_by_key_fun
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::internal::scan_by_key_fun<int_device_copyable, int_device_copyable, noop_device_copyable>>,
        "scan_by_key_fun is not device copyable with device copyable types");
    //segmented_scan_fun
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::internal::segmented_scan_fun<
                      int_non_device_copyable, int_device_copyable, noop_device_copyable>>,
                  "segmented_scan_fun is not device copyable with device copyable types");
    //scatter_and_accumulate_fun
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::internal::scatter_and_accumulate_fun<int_device_copyable, int_device_copyable>>,
                  "scatter_and_accumulate_fun is not device copyable with device copyable types");
    //transform_if_stencil_fun
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::internal::transform_if_stencil_fun<
                      int_device_copyable, noop_device_copyable, noop_device_copyable>>,
                  "transform_if_stencil_fun is not device copyable with device copyable types");

    using policy_non_device_copyable = oneapi::dpl::execution::device_policy<oneapi::dpl::execution::DefaultKernelName>;
    //walk_n
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::walk_n<policy_non_device_copyable, noop_device_copyable>>,
                  "walk_n is not device copyable with device copyable types");
    //walk_adjacent_difference
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::walk_adjacent_difference<policy_non_device_copyable, noop_device_copyable>>,
        "walk_adjacent_difference is not device copyable with device copyable types");
    //transform_reduce
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::transform_reduce<policy_non_device_copyable, 8, noop_device_copyable,
                                                         noop_device_copyable, int_device_copyable, std::true_type>>,
        "transform_reduce is not device copyable with device copyable types");
    //reduce_over_group
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::reduce_over_group<
                      policy_non_device_copyable, noop_device_copyable, int_device_copyable>>,
                  "reduce_over_group is not device copyable with device copyable types");
    //single_match_pred_by_idx
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::single_match_pred_by_idx<policy_non_device_copyable, noop_device_copyable>>,
        "single_match_pred_by_idx is not device copyable with device copyable types");
    //multiple_match_pred
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::multiple_match_pred<policy_non_device_copyable, noop_device_copyable>>,
        "multiple_match_pred is not device copyable with device copyable types");
    //n_elem_match_pred
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::n_elem_match_pred<
                      policy_non_device_copyable, noop_device_copyable, int_device_copyable, int_device_copyable>>,
                  "n_elem_match_pred is not device copyable with device copyable types");
    //first_match_pred
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::first_match_pred<policy_non_device_copyable, noop_device_copyable>>,
                  "first_match_pred is not device copyable with device copyable types");
    //__create_mask
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::__create_mask<noop_device_copyable, int_device_copyable>>,
                  "__create_mask is not device copyable with device copyable types");
    //__copy_by_mask
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::__copy_by_mask<noop_device_copyable, noop_device_copyable, std::true_type, 10>>,
        "__copy_by_mask is not device copyable with device copyable types");
    // __partition_by_mask
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::__partition_by_mask<noop_device_copyable, std::true_type>>,
                  "__partition_by_mask is not device copyable with device copyable types");
    // __global_scan_functor
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__global_scan_functor<
                      std::true_type, noop_device_copyable, int_device_copyable>>,
                  "__global_scan_functor is not device copyable with device copyable types");
    // __scan
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__scan<
                      std::true_type, policy_non_device_copyable, noop_device_copyable, noop_device_copyable,
                      noop_device_copyable, noop_device_copyable, noop_device_copyable,
                      oneapi::dpl::unseq_backend::__init_value<int_device_copyable>>>,
                  "__scan is not device copyable with device copyable types");
    // __brick_includes
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_includes<
                      policy_non_device_copyable, noop_device_copyable, int_device_copyable, int_device_copyable>>,
                  "__brick_includes is not device copyable with device copyable types");
    // __brick_set_op
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::__brick_set_op<policy_non_device_copyable, noop_device_copyable,
                                                       int_device_copyable, int_device_copyable, std::true_type>>,
        "__brick_set_op is not device copyable with device copyable types");
    // __brick_reduce_idx
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::__brick_reduce_idx<noop_device_copyable, int_device_copyable>>,
                  "__brick_reduce_idx is not device copyable with device copyable types");

    // __early_exit_find_or
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::__par_backend_hetero::__early_exit_find_or<policy_non_device_copyable, noop_device_copyable>>,
        "__early_exit_find_or is not device copyable with device copyable types");

    //__not_pred
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__not_pred<noop_device_copyable>>,
                  "__not_pred is not device copyable with device copyable types");
    //__reorder_pred
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__reorder_pred<noop_device_copyable>>,
                  "__reorder_pred is not device copyable with device copyable types");
    //__equal_value_by_pred
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__equal_value_by_pred<int_device_copyable, noop_device_copyable>>,
                  "__equal_value_by_pred is not device copyable with device copyable types");
    //__equal_value
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__equal_value<int_device_copyable>>,
                  "__equal_value is not device copyable with device copyable types");
    //__not_equal_value
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__not_equal_value<int_device_copyable>>,
                  "__not_equal_value is not device copyable with device copyable types");
    //__transform_functor
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__transform_functor<noop_device_copyable>>,
                  "__transform_functor is not device copyable with device copyable types");
    //__transform_if_unary_functor
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__transform_if_unary_functor<noop_device_copyable, noop_device_copyable>>,
        "__transform_if_unary_functor is not device copyable with device copyable types");
    //__transform_if_binary_functor
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__transform_if_binary_functor<noop_device_copyable, noop_device_copyable>>,
        "__transform_if_binary_functor is not device copyable with device copyable types");
    //__replace_functor
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__replace_functor<int_device_copyable, noop_device_copyable>>,
                  "__replace_functor is not device copyable with device copyable types");
    //__replace_copy_functor
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__replace_copy_functor<int_device_copyable, noop_device_copyable>>,
                  "__replace_copy_functor is not device copyable with device copyable types");
    //fill_functor
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::fill_functor<int_device_copyable>>,
                  "fill_functor is not device copyable with device copyable types");
    //generate_functor
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::generate_functor<int_device_copyable>>,
                  "generate_functor is not device copyable with device copyable types");

    using hetero_device_tag = oneapi::dpl::__internal::__hetero_tag<oneapi::dpl::__internal::__device_backend_tag>;
    //__brick_fill
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__brick_fill<hetero_device_tag, policy_non_device_copyable, int_device_copyable>>,
        "__brick_fill is not device copyable with device copyable types");
    //__brick_fill_n
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__brick_fill_n<
                      hetero_device_tag, policy_non_device_copyable, int_device_copyable>>,
                  "__brick_fill_n is not device copyable with device copyable types");
    //__search_n_unary_predicate
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__search_n_unary_predicate<int_device_copyable, noop_device_copyable>>,
                  "__search_n_unary_predicate is not device copyable with device copyable types");
    //__is_heap_check
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::__is_heap_check<noop_device_copyable>>,
                  "__is_heap_check is not device copyable with device copyable types");
    //equal_predicate
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::equal_predicate<noop_device_copyable>>,
                  "equal_predicate is not device copyable with device copyable types");
    //adjacent_find_fn
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::adjacent_find_fn<noop_device_copyable>>,
                  "adjacent_find_fn is not device copyable with device copyable types");
    //__create_mask_unique_copy
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__create_mask_unique_copy<noop_device_copyable, int_device_copyable>>,
                  "__create_mask_unique_copy is not device copyable with device copyable types");
    //tuple
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::__internal::tuple<int_device_copyable, int_device_copyable>>,
                  "tuple is not device copyable with device copyable types");
    static_assert(
        sycl::is_device_copyable_v<oneapi::dpl::__internal::tuple<std::tuple<int_device_copyable, int_device_copyable>,
                                                                  int_device_copyable, int_device_copyable>>,
        "tuple is not device copyable with device copyable types");
}

void
test_non_device_copyable()
{
    //first check that our testing types defined as non-device copyable are in fact non-device copyable
    static_assert(!sycl::is_device_copyable_v<noop_non_device_copyable>, "functor is device copyable");
    static_assert(!sycl::is_device_copyable_v<int_non_device_copyable>, "struct is device copyable");
    static_assert(!sycl::is_device_copyable_v<constant_iterator_non_device_copyable>, "iterator is device copyable");

    //custom_brick
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::internal::custom_brick<
            noop_device_copyable, int_non_device_copyable, oneapi::dpl::internal::search_algorithm::lower_bound>>,
        "custom_brick is device copyable with non device copyable types");
    //replace_if_fun
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::internal::replace_if_fun<int_device_copyable, noop_non_device_copyable>>,
                  "replace_if_fun is device copyable with non device copyable types");
    //scan_by_key_fun
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::internal::scan_by_key_fun<int_non_device_copyable, int_device_copyable,
                                                                           noop_non_device_copyable>>,
        "scan_by_key_fun is device copyable with non device copyable types");
    //segmented_scan_fun
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::internal::segmented_scan_fun<int_device_copyable, int_device_copyable,
                                                                              noop_non_device_copyable>>,
        "segmented_scan_fun is device copyable with non device copyable types");
    //scatter_and_accumulate_fun
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::internal::scatter_and_accumulate_fun<int_non_device_copyable, int_device_copyable>>,
                  "scatter_and_accumulate_fun is device copyable with non device copyable types");
    //transform_if_stencil_fun
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::internal::transform_if_stencil_fun<
                      int_device_copyable, noop_non_device_copyable, noop_device_copyable>>,
                  "transform_if_stencil_fun is device copyable with non device copyable types");

    using policy_non_device_copyable = oneapi::dpl::execution::device_policy<oneapi::dpl::execution::DefaultKernelName>;
    //walk_n
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::walk_n<policy_non_device_copyable, noop_non_device_copyable>>,
                  "walk_n is device copyable with non device copyable types");
    //walk_adjacent_difference
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::walk_adjacent_difference<policy_non_device_copyable, noop_non_device_copyable>>,
        "walk_adjacent_difference is device copyable with non device copyable types");
    //transform_reduce
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::transform_reduce<policy_non_device_copyable, 8, noop_non_device_copyable,
                                                         noop_device_copyable, int_device_copyable, std::true_type>>,
        "transform_reduce is device copyable with non device copyable types");
    //reduce_over_group
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::reduce_over_group<
                      policy_non_device_copyable, noop_non_device_copyable, int_device_copyable>>,
                  "reduce_over_group is device copyable with non device copyable types");
    //single_match_pred_by_idx
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::single_match_pred_by_idx<policy_non_device_copyable, noop_non_device_copyable>>,
        "single_match_pred_by_idx is device copyable with non device copyable types");
    //multiple_match_pred
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::multiple_match_pred<policy_non_device_copyable, noop_non_device_copyable>>,
        "multiple_match_pred is device copyable with non device copyable types");
    //n_elem_match_pred
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::n_elem_match_pred<
                      policy_non_device_copyable, noop_device_copyable, int_non_device_copyable, int_device_copyable>>,
                  "n_elem_match_pred is device copyable with non device copyable types");
    //first_match_pred
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::first_match_pred<policy_non_device_copyable, noop_non_device_copyable>>,
        "first_match_pred is device copyable with non device copyable types");
    //__create_mask
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::__create_mask<noop_device_copyable, int_non_device_copyable>>,
                  "__create_mask is device copyable with non device copyable types");
    //__copy_by_mask
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__copy_by_mask<
                      noop_device_copyable, noop_non_device_copyable, std::true_type, 10>>,
                  "__copy_by_mask is device copyable with non device copyable types");
    //__partition_by_mask
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::__partition_by_mask<noop_non_device_copyable, std::true_type>>,
                  "__partition_by_mask is device copyable with non device copyable types");
    //__global_scan_functor
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__global_scan_functor<
                      std::true_type, noop_non_device_copyable, int_device_copyable>>,
                  "__global_scan_functor is device copyable with non device copyable types");
    //__scan
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__scan<
                      std::true_type, policy_non_device_copyable, noop_non_device_copyable, noop_device_copyable,
                      noop_device_copyable, noop_device_copyable, noop_device_copyable,
                      oneapi::dpl::unseq_backend::__init_value<int_device_copyable>>>,
                  "__scan is device copyable with non device copyable types");
    //__brick_includes
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_includes<
                      policy_non_device_copyable, noop_non_device_copyable, int_device_copyable, int_device_copyable>>,
                  "__brick_includes is device copyable with non device copyable types");
    //__brick_set_op
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::unseq_backend::__brick_set_op<policy_non_device_copyable, noop_non_device_copyable,
                                                       int_device_copyable, int_device_copyable, std::true_type>>,
        "__brick_set_op is device copyable with non device copyable types");
    //__brick_reduce_idx
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::unseq_backend::__brick_reduce_idx<noop_device_copyable, int_non_device_copyable>>,
                  "__brick_reduce_idx is device copyable with non device copyable types");

    // __early_exit_find_or
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::__par_backend_hetero::__early_exit_find_or<policy_non_device_copyable,
                                                                                            noop_non_device_copyable>>,
        "__early_exit_find_or is device copyable with non device copyable types");

    //__not_pred
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__not_pred<noop_non_device_copyable>>,
                  "__not_pred is device copyable with non device copyable types");
    //__reorder_pred
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__reorder_pred<noop_non_device_copyable>>,
                  "__reorder_pred is device copyable with non device copyable types");

    //__equal_value_by_pred
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__equal_value_by_pred<int_device_copyable, noop_non_device_copyable>>,
                  "__equal_value_by_pred is device copyable with non device copyable types");

    //__equal_value
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__equal_value<int_non_device_copyable>>,
                  "__equal_value is device copyable with non device copyable types");

    //__not_equal_value
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__not_equal_value<int_non_device_copyable>>,
                  "__not_equal_value is device copyable with non device copyable types");

    //__transform_functor
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__transform_functor<noop_non_device_copyable>>,
                  "__transform_functor is device copyable with non device copyable types");

    //__transform_if_unary_functor
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__transform_if_unary_functor<noop_non_device_copyable, noop_non_device_copyable>>,
        "__transform_if_unary_functor is device copyable with non device copyable types");

    //__transform_if_binary_functor
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__transform_if_binary_functor<noop_non_device_copyable, noop_non_device_copyable>>,
        "__transform_if_binary_functor is device copyable with non device copyable types");

    //__replace_functor
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__replace_functor<int_device_copyable, noop_non_device_copyable>>,
                  "__replace_functor is device copyable with non device copyable types");

    //__replace_copy_functor
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::__internal::__replace_copy_functor<int_device_copyable, noop_non_device_copyable>>,
                  "__replace_copy_functor is device copyable with non device copyable types");

    //fill_functor
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::fill_functor<int_non_device_copyable>>,
                  "fill_functor is device copyable with non device copyable types");

    //generate_functor
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::generate_functor<int_non_device_copyable>>,
                  "generate_functor is device copyable with non device copyable types");

    using hetero_device_tag = oneapi::dpl::__internal::__hetero_tag<oneapi::dpl::__internal::__device_backend_tag>;
    //__brick_fill
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::__internal::__brick_fill<hetero_device_tag, policy_non_device_copyable,
                                                                          int_non_device_copyable>>,
        "__brick_fill is device copyable with non device copyable types");

    //__brick_fill_n
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__brick_fill_n<
                      hetero_device_tag, policy_non_device_copyable, int_non_device_copyable>>,
                  "__brick_fill_n is device copyable with non device copyable types");

    //__search_n_unary_predicate
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__search_n_unary_predicate<int_device_copyable, noop_non_device_copyable>>,
        "__search_n_unary_predicate is device copyable with non device copyable types");

    //__is_heap_check
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::__is_heap_check<noop_non_device_copyable>>,
                  "__is_heap_check is device copyable with non device copyable types");

    //equal_predicate
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::equal_predicate<noop_non_device_copyable>>,
                  "equal_predicate is device copyable with non device copyable types");

    //adjacent_find_fn
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::adjacent_find_fn<noop_non_device_copyable>>,
                  "adjacent_find_fn is device copyable with non device copyable types");

    //__create_mask_unique_copy
    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::__internal::__create_mask_unique_copy<noop_non_device_copyable, int_non_device_copyable>>,
        "__create_mask_unique_copy is device copyable with non device copyable types");
    //tuple
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::__internal::tuple<int_non_device_copyable, int_device_copyable>>,
        "tuple is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::tuple<
                      std::tuple<int_non_device_copyable, int_device_copyable>, int_device_copyable>>,
                  "tuple is device copyable with non device copyable types");
}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_device_copyable();
    test_non_device_copyable();
#endif // TEST_DPCPP_BACKEND_PRESENT
    return done(TEST_DPCPP_BACKEND_PRESENT);
}
