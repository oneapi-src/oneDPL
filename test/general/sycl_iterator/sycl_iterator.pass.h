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

#ifndef _TEST_SYCL_ITERATOR_PASS_H
#define _TEST_SYCL_ITERATOR_PASS_H

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(memory)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"
#include "oneapi/dpl/pstl/utils.h"

#include <cmath>
#include <type_traits>

using namespace TestUtils;

//This macro is required for the tests to work correctly in CI with tbb-backend.
#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"

struct Flip
{
    std::int32_t val;
    Flip(std::int32_t y) : val(y) {}
    template <typename T>
    T
    operator()(const T& x) const
    {
        return val - x;
    }
};
struct Plus
{
    template <typename T, typename U>
    T
    operator()(const T x, const U y) const
    {
        return x + y;
    }
};

struct Inc
{
    template <typename T>
    void
    operator()(T& x) const
    {
        ++x;
    }
};

template <typename T>
struct Generator_count
{
    T def_val;
    Generator_count(const T& val) : def_val(val) {}
    T
    operator()() const
    {
        return def_val;
    }
    T
    default_value() const
    {
        return def_val;
    }
};

// created just to check destroy and destroy_n correctness
template <typename T>
struct SyclTypeWrapper
{
    T __value;

    explicit SyclTypeWrapper(const T& value = T{4}) : __value(value) {}
    ~SyclTypeWrapper() { __value = -2; }
    bool
    operator==(const SyclTypeWrapper& other) const
    {
        return __value == other.__value;
    }
};

// this wrapper is needed to take into account not only kernel name,
// but also other types (for example, iterator's value type)
template<typename... T>
struct policy_name_wrapper{};

using namespace oneapi::dpl::execution;

template <typename Policy>
void
wait_and_throw(Policy&& exec)
{
#if _PSTL_SYCL_TEST_USM
    exec.queue().wait_and_throw();
#endif // _PSTL_SYCL_TEST_USM
}

inline constexpr int a[] = {0, 0, 1, 1, 2, 6, 6, 9, 9};
inline constexpr int b[] = {0, 1, 1, 6, 6, 9};
inline constexpr int c[] = {0, 1, 6, 6, 6, 9, 9};
inline constexpr int d[] = {7, 7, 7, 8};
inline constexpr int e[] = {11, 11, 12, 16, 19};
inline constexpr auto a_size = sizeof(a) / sizeof(a[0]);
inline constexpr auto b_size = sizeof(b) / sizeof(b[0]);
inline constexpr auto c_size = sizeof(c) / sizeof(c[0]);
inline constexpr auto d_size = sizeof(d) / sizeof(d[0]);

template <typename Size>
Size get_size(Size n)
{
    return n + a_size + b_size + c_size + d_size;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // _TEST_SYCL_ITERATOR_PASS_H
