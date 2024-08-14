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
#ifndef _UTILS_DEVICE_COPYABLE_H
#define _UTILS_DEVICE_COPYABLE_H

#if TEST_DPCPP_BACKEND_PRESENT
#include "utils_sycl_defs.h"
#include <iostream>
#include <type_traits>
#include <cstddef>
#include <iterator>

namespace TestUtils
{

// Device copyable noop functor used in testing as surrogate for predicates, binary ops, unary functors
// Intentionally non-trivially copyable to test that device_copyable speciailzation works and we are not
// relying on trivial copyability
struct noop_device_copyable
{
    noop_device_copyable(const noop_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
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

// Device copyable int wrapper struct used in testing as surrogate for values, value types, etc.
// Intentionally non-trivially copyable to test that device_copyable speciailzation works and we are not
// relying on trivial copyability
struct int_device_copyable
{
    int i;
    int_device_copyable(const int_device_copyable& other) : i(other.i) { std::cout << "non trivial copy ctor\n"; }
};

struct int_non_device_copyable
{
    int i;
    int_non_device_copyable(const int_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }
};

// Device copyable iterator used in testing as surrogate for iterators.
// Intentionally non-trivially copyable to test that device_copyable speciailzation works and we are not
// relying on trivial copyability
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

} /* namespace TestUtils */

template <>
struct sycl::is_device_copyable<TestUtils::noop_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::int_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::constant_iterator_device_copyable> : std::true_type
{
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // _UTILS_DEVICE_COPYABLE_H
