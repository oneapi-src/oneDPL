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

struct noop
{
    int
    operator()(int a) const
    {
        return a;
    }
};

struct noop_non_device_copyable
{
    noop_non_device_copyable(const noop_non_device_copyable& other)
    {
        std::cout<<"non trivial copy ctor\n";
    }
    int
    operator()(int a) const
    {
        return a;
    }
};

struct int_non_device_copyable
{
    int i;
    int_non_device_copyable(const int_non_device_copyable& other) : i(other.i)
    {
        std::cout<<"non trivial copy ctor\n";
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
        std::cout<<"non trivial copy ctor\n";
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


void
test_device_copyable()
{
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::transform_iterator<int*, noop>>,
                  "transform_iterator is not device copyable with device copyable types");
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<int*, double*>>,
                  "zip_iterator is not device copyable with device copyable types");
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::permutation_iterator<int*, int*>>,
                  "permutation_iterator is not device copyable with device copyable types");
}

void
test_non_device_copyable()
{
    //first check that our testing types defined as non-device copyable are in fact non-device copyable
    static_assert(!sycl::is_device_copyable_v<noop_non_device_copyable>, "functor is device copyable");
    static_assert(!sycl::is_device_copyable_v<int_non_device_copyable>, "struct is device copyable");
    static_assert(!sycl::is_device_copyable_v<constant_iterator_non_device_copyable>, "iterator is device copyable");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::transform_iterator<int*, noop_non_device_copyable>>,
                  "transform_iterator is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<constant_iterator_non_device_copyable, int*>>,
                  "zip_iterator is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<int*, constant_iterator_non_device_copyable>>,
                  "zip_iterator is device copyable with non device copyable types");

    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::permutation_iterator<constant_iterator_non_device_copyable, noop>>,
        "permutation_iterator is device copyable with non device copyable types");
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::permutation_iterator<int*, noop_non_device_copyable>>,
                  "permutation_iterator is device copyable with non device copyable types");
    static_assert(
        !sycl::is_device_copyable_v<oneapi::dpl::permutation_iterator<int*, constant_iterator_non_device_copyable>>,
        "permutation_iterator is device copyable with non device copyable types");

    static_assert(!std::is_trivially_copyable_v<constant_iterator_non_device_copyable>,
                  "iterator is trivially copyable");
    static_assert(
        !std::is_trivially_copyable_v<oneapi::dpl::permutation_iterator<int*, constant_iterator_non_device_copyable>>,
        "permutation_iterator with non trivially copyable index is trivially copyable");
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
