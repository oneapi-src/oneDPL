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

#    include "sycl.hpp"

struct noop_device_copyable
{
    noop_device_copyable(const noop_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
};

template <>
struct sycl::is_device_copyable<noop_device_copyable> : std::true_type
{
};

struct noop_non_device_copyable
{
    noop_non_device_copyable(const noop_non_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
};

struct int_device_copyable
{
    int i;
    int_device_copyable(const int_device_copyable& other) : i(other.i) { std::cout << "non trivial copy ctor\n"; }
};

template <>
struct sycl::is_device_copyable<int_device_copyable> : std::true_type
{
};

struct int_non_device_copyable
{
    int i;
    int_non_device_copyable(const int_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }
};

//constant iterator which has custom copy constructor and is not device copyable
struct constant_iterator_non_device_copyable
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    int i;
    constant_iterator_non_device_copyable(int __i) : i(__i) {}

    constant_iterator_non_device_copyable(const constant_iterator_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }

    reference operator*() const { return i; }

    constant_iterator_non_device_copyable& operator++() { return *this; }
    constant_iterator_non_device_copyable operator++(int) {return *this; }

    constant_iterator_non_device_copyable& operator--() {  return *this; }
    constant_iterator_non_device_copyable operator--(int) { return *this; }

    constant_iterator_non_device_copyable& operator+=(difference_type n) {return *this; }
    constant_iterator_non_device_copyable operator+(difference_type n) const { return constant_iterator_non_device_copyable(i); }
    friend constant_iterator_non_device_copyable operator+(difference_type n, const constant_iterator_non_device_copyable& it) { return constant_iterator_non_device_copyable(it.i); }

    constant_iterator_non_device_copyable& operator-=(difference_type n) { return *this; }
    constant_iterator_non_device_copyable operator-(difference_type n) const { return constant_iterator_non_device_copyable(i); }
    difference_type operator-(const constant_iterator_non_device_copyable& other) const { return 0; }

    reference operator[](difference_type n) const { return i; }

    bool operator==(const constant_iterator_non_device_copyable& other) const { return true; }
    bool operator!=(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator<(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator>(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator<=(const constant_iterator_non_device_copyable& other) const { return true; }
    bool operator>=(const constant_iterator_non_device_copyable& other) const { return true; }
};


struct constant_iterator_device_copyable
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    int i;
    constant_iterator_device_copyable(int __i) : i(__i) {}

    constant_iterator_device_copyable(const constant_iterator_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }

    reference operator*() const { return i; }

    constant_iterator_device_copyable& operator++() { return *this; }
    constant_iterator_device_copyable operator++(int) {return *this; }

    constant_iterator_device_copyable& operator--() {  return *this; }
    constant_iterator_device_copyable operator--(int) { return *this; }

    constant_iterator_device_copyable& operator+=(difference_type n) {return *this; }
    constant_iterator_device_copyable operator+(difference_type n) const { return constant_iterator_device_copyable(i); }
    friend constant_iterator_device_copyable operator+(difference_type n, const constant_iterator_device_copyable& it) { return constant_iterator_device_copyable(it.i); }

    constant_iterator_device_copyable& operator-=(difference_type n) { return *this; }
    constant_iterator_device_copyable operator-(difference_type n) const { return constant_iterator_device_copyable(i); }
    difference_type operator-(const constant_iterator_device_copyable& other) const { return 0; }

    reference operator[](difference_type n) const { return i; }

    bool operator==(const constant_iterator_device_copyable& other) const { return true; }
    bool operator!=(const constant_iterator_device_copyable& other) const { return false; }
    bool operator<(const constant_iterator_device_copyable& other) const { return false; }
    bool operator>(const constant_iterator_device_copyable& other) const { return false; }
    bool operator<=(const constant_iterator_device_copyable& other) const { return true; }
    bool operator>=(const constant_iterator_device_copyable& other) const { return true; }
};

template <>
struct sycl::is_device_copyable<constant_iterator_device_copyable> : std::true_type
{
};

void
test_device_copyable()
{
    //check that our testing types are non-trivially copyable and device copyable
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

    //now check that our custom types can be device copyable with these test types
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::transform_iterator<constant_iterator_device_copyable, noop_device_copyable>>,
                  "transform_iterator is not device copyable with device copyable types");
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<constant_iterator_device_copyable, int*>>,
                  "zip_iterator is not device copyable with device copyable types");
    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::permutation_iterator<constant_iterator_device_copyable, constant_iterator_device_copyable>>,
        "permutation_iterator is not device copyable with device copyable types");
    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::permutation_iterator<constant_iterator_device_copyable, noop_device_copyable>>,
                  "permutation_iterator is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::internal::custom_brick<
                      noop_device_copyable, int_device_copyable, oneapi::dpl::internal::search_algorithm::lower_bound>>,
                  "custom_brick is not device copyable with device copyable types");

    static_assert(
        sycl::is_device_copyable_v<oneapi::dpl::internal::replace_if_fun<int_device_copyable, noop_device_copyable>>,
        "replace_if_fun is not device copyable with device copyable types");

    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::internal::scan_by_key_fun<int_device_copyable, int_device_copyable, noop_device_copyable>>,
        "scan_by_key_fun is not device copyable with device copyable types");

    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::internal::segmented_scan_fun<int_non_device_copyable, int_device_copyable, noop_device_copyable>>,
        "segmented_scan_fun is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::internal::scatter_and_accumulate_fun<int_device_copyable, int_device_copyable>>,
                  "scatter_and_accumulate_fun is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::internal::transform_if_stencil_fun<
                      int_device_copyable, noop_device_copyable, noop_device_copyable>>,
                  "transform_if_stencil_fun is not device copyable with device copyable types");

    using policy_non_device_copyable = oneapi::dpl::execution::device_policy<oneapi::dpl::execution::DefaultKernelName>;

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::walk_n<policy_non_device_copyable, noop_device_copyable>>,
                  "walk_n is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::walk_adjacent_difference<policy_non_device_copyable, noop_device_copyable>>,
                  "walk_adjacent_difference is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::transform_reduce<policy_non_device_copyable, 8,  noop_device_copyable, noop_device_copyable, int_device_copyable, std::true_type>>,
                  "transform_reduce is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::reduce_over_group<policy_non_device_copyable, noop_device_copyable, int_device_copyable>>,
                  "reduce_over_group is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::single_match_pred_by_idx<policy_non_device_copyable, noop_device_copyable>>,
                  "single_match_pred_by_idx is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::multiple_match_pred<policy_non_device_copyable, noop_device_copyable>>,
                  "multiple_match_pred is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::n_elem_match_pred<policy_non_device_copyable, noop_device_copyable, int_device_copyable, int_device_copyable>>,
                  "n_elem_match_pred is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::first_match_pred<policy_non_device_copyable, noop_device_copyable>>,
                  "first_match_pred is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__create_mask<noop_device_copyable, int_device_copyable>>,
                  "__create_mask is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__copy_by_mask<noop_device_copyable,noop_device_copyable, std::true_type, 10>>,
                  "__copy_by_mask is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__partition_by_mask<noop_device_copyable, std::true_type>>,
                  "__partition_by_mask is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__global_scan_functor<std::true_type, noop_device_copyable, int_device_copyable>>,
                  "__global_scan_functor is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__scan<std::true_type, policy_non_device_copyable, noop_device_copyable, noop_device_copyable, noop_device_copyable, noop_device_copyable, noop_device_copyable, oneapi::dpl::unseq_backend::__init_value<int_device_copyable>>>,
                  "__scan is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_includes<policy_non_device_copyable, noop_device_copyable, int_device_copyable, int_device_copyable>>,
                  "__brick_includes is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_set_op<policy_non_device_copyable, noop_device_copyable, int_device_copyable, int_device_copyable, std::true_type>>,
                  "__brick_set_op is not device copyable with device copyable types");

    static_assert(sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_reduce_idx<noop_device_copyable, int_device_copyable>>,
                  "__brick_reduce_idx is not device copyable with device copyable types");

}

void
test_non_device_copyable()
{
    //first check that our testing types defined as non-device copyable are in fact non-device copyable
    static_assert(!sycl::is_device_copyable_v<noop_non_device_copyable>, "functor is device copyable");
    static_assert(!sycl::is_device_copyable_v<int_non_device_copyable>, "struct is device copyable");
    static_assert(!sycl::is_device_copyable_v<constant_iterator_non_device_copyable>, "iterator is device copyable");

    //test custom iterators

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::transform_iterator<int*, noop_non_device_copyable>>,
                  "transform_iterator is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::transform_iterator<constant_iterator_non_device_copyable, noop_device_copyable>>,
                  "transform_iterator is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<constant_iterator_non_device_copyable, int*>>,
                  "zip_iterator is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<int*, constant_iterator_non_device_copyable>>,
                  "zip_iterator is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::permutation_iterator<constant_iterator_non_device_copyable, noop_device_copyable>>,
                  "permutation_iterator is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::permutation_iterator<int*, noop_non_device_copyable>>,
                  "permutation_iterator is device copyable with non device copyable types");
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::permutation_iterator<int*, constant_iterator_non_device_copyable>>,
        "permutation_iterator is device copyable with non device copyable types");


    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::internal::custom_brick<
                      noop_device_copyable, int_non_device_copyable, oneapi::dpl::internal::search_algorithm::lower_bound>>,
                  "custom_brick is device copyable with non device copyable types");

    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::internal::replace_if_fun<int_device_copyable, noop_non_device_copyable>>,
        "replace_if_fun is device copyable with non device copyable types");

    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::internal::scan_by_key_fun<int_non_device_copyable, int_device_copyable, noop_non_device_copyable>>,
        "scan_by_key_fun is device copyable with non device copyable types");

    static_assert(
        !sycl::is_device_copyable_v<
            oneapi::dpl::internal::segmented_scan_fun<int_device_copyable, int_device_copyable, noop_non_device_copyable>>,
        "segmented_scan_fun is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::internal::scatter_and_accumulate_fun<int_non_device_copyable, int_device_copyable>>,
                  "scatter_and_accumulate_fun is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::internal::transform_if_stencil_fun<
                      int_device_copyable, noop_non_device_copyable, noop_device_copyable>>,
                  "transform_if_stencil_fun is device copyable with non device copyable types");

    using policy_non_device_copyable = oneapi::dpl::execution::device_policy<oneapi::dpl::execution::DefaultKernelName>;

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::walk_n<policy_non_device_copyable, noop_non_device_copyable>>,
                  "walk_n is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::walk_adjacent_difference<policy_non_device_copyable, noop_non_device_copyable>>,
                  "walk_adjacent_difference is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::transform_reduce<policy_non_device_copyable, 8,  noop_non_device_copyable, noop_device_copyable, int_device_copyable, std::true_type>>,
                  "transform_reduce is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::reduce_over_group<policy_non_device_copyable, noop_non_device_copyable, int_device_copyable>>,
                  "reduce_over_group is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::single_match_pred_by_idx<policy_non_device_copyable, noop_non_device_copyable>>,
                  "single_match_pred_by_idx is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::multiple_match_pred<policy_non_device_copyable, noop_non_device_copyable>>,
                  "multiple_match_pred is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::n_elem_match_pred<policy_non_device_copyable, noop_device_copyable, int_non_device_copyable, int_device_copyable>>,
                  "n_elem_match_pred is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::first_match_pred<policy_non_device_copyable, noop_non_device_copyable>>,
                  "first_match_pred is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__create_mask<noop_device_copyable, int_non_device_copyable>>,
                  "__create_mask is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__copy_by_mask<noop_device_copyable,noop_non_device_copyable, std::true_type, 10>>,
                  "__copy_by_mask is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__partition_by_mask<noop_non_device_copyable, std::true_type>>,
                  "__partition_by_mask is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__global_scan_functor<std::true_type, noop_non_device_copyable, int_device_copyable>>,
                  "__global_scan_functor is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__scan<std::true_type, policy_non_device_copyable, noop_non_device_copyable, noop_device_copyable, noop_device_copyable, noop_device_copyable, noop_device_copyable, oneapi::dpl::unseq_backend::__init_value<int_device_copyable>>>,
                  "__scan is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_includes<policy_non_device_copyable, noop_non_device_copyable, int_device_copyable, int_device_copyable>>,
                  "__brick_includes is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_set_op<policy_non_device_copyable, noop_non_device_copyable, int_device_copyable, int_device_copyable, std::true_type>>,
                  "__brick_set_op is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::unseq_backend::__brick_reduce_idx<noop_device_copyable, int_non_device_copyable>>,
                  "__brick_reduce_idx is device copyable with non device copyable types");

}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_device_copyable();
    test_non_device_copyable();
#endif // TEST_DPCPP_BACKEND_PRESENT
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
