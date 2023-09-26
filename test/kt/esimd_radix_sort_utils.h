
// -*- C++ -*-
//===-- esimd_radix_sort_test_utils.h -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ESIMD_RADIX_SORT_TEST_UTILS_H
#define _ESIMD_RADIX_SORT_TEST_UTILS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#ifndef LOG_TEST_INFO
#    define LOG_TEST_INFO 0
#endif

#include <string>
#include <tuple>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <cstdint>

template <typename T, bool Order>
struct Compare : public std::less<T>
{
};

template <typename T>
struct Compare<T, false> : public std::greater<T>
{
};

template<bool Order>
struct CompareKey
{
    template<typename T, typename U>
    bool operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) < std::get<0>(rhs);
    }
};

template<>
struct CompareKey<false>
{
    template<typename T, typename U>
    bool operator()(const T& lhs, const U& rhs) const
    {
        return std::get<0>(lhs) > std::get<0>(rhs);
    }
};

constexpr bool Ascending = true;
constexpr bool Descending = false;
constexpr std::uint8_t TestRadixBits = 8;

#if LOG_TEST_INFO
struct TypeInfo
{
    template <typename T>
    const std::string&
    name()
    {
        static const std::string TypeName = "unknown type name";
        return TypeName;
    }

    template <>
    const std::string&
    name<char>()
    {
        static const std::string TypeName = "char";
        return TypeName;
    }

    template <>
    const std::string&
    name<int8_t>()
    {
        static const std::string TypeName = "int8_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint8_t>()
    {
        static const std::string TypeName = "uint8_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<int16_t>()
    {
        static const std::string TypeName = "int16_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint16_t>()
    {
        static const std::string TypeName = "uint16_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint32_t>()
    {
        static const std::string TypeName = "uint32_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<uint64_t>()
    {
        static const std::string TypeName = "uint64_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<int64_t>()
    {
        static const std::string TypeName = "int64_t";
        return TypeName;
    }

    template <>
    const std::string&
    name<int>()
    {
        static const std::string TypeName = "int";
        return TypeName;
    }

    template <>
    const std::string&
    name<float>()
    {
        static const std::string TypeName = "float";
        return TypeName;
    }

    template <>
    const std::string&
    name<double>()
    {
        static const std::string TypeName = "double";
        return TypeName;
    }
};

struct USMAllocPresentation
{
    template <sycl::usm::alloc>
    const std::string&
    name()
    {
        static const std::string USMAllocTypeName = "unknown";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::host>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::host";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::device>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::device";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::shared>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::shared";
        return USMAllocTypeName;
    }

    template <>
    const std::string&
    name<sycl::usm::alloc::unknown>()
    {
        static const std::string USMAllocTypeName = "sycl::usm::alloc::unknown";
        return USMAllocTypeName;
    }
};
#endif // LOG_TEST_INFO

template <typename T>
typename ::std::enable_if_t<std::is_arithmetic_v<T>, void>
generate_data(T* input, std::size_t size, std::uint32_t seed)
{
    std::default_random_engine gen{seed};
    std::size_t unique_threshold = 75 * size / 100;
    if constexpr (sizeof(T) < sizeof(short)) // no uniform_int_distribution for chars
    {
        std::uniform_int_distribution<int> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
        std::generate(input, input + unique_threshold, [&] { return T(dist(gen)); });
    }
    else if constexpr (std::is_integral_v<T>)
    {
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
        std::generate(input, input + unique_threshold, [&] { return dist(gen); });
    }
    else
    {
        std::uniform_real_distribution<T> dist_real(std::numeric_limits<T>::min(), log2(1e12));
        std::uniform_int_distribution<int> dist_binary(0, 1);
        auto randomly_signed_real = [&dist_real, &dist_binary, &gen]()
        {
            auto v = exp2(dist_real(gen));
            return dist_binary(gen) == 0 ? v : -v;
        };
        std::generate(input, input + unique_threshold, [&] { return randomly_signed_real(); });
    }
    for (uint32_t i = 0, j = unique_threshold; j < size; ++i, ++j)
    {
        input[j] = input[i];
    }
}

template <typename Container1, typename Container2>
void
print_data(const Container1& expected, const Container2& actual, std::size_t first, std::size_t n = 0)
{
    if (expected.size() <= first)
        return;
    if (n == 0 || expected.size() < first + n)
        n = expected.size() - first;

    if constexpr (std::is_floating_point_v<typename Container1::value_type>)
        std::cout << std::hexfloat;
    else
        std::cout << std::hex;

    for (std::size_t i = first; i < first + n; ++i)
    {
        std::cout << actual[i] << " --- " << expected[i] << std::endl;
    }

    if constexpr (std::is_floating_point_v<typename Container1::value_type>)
        std::cout << std::defaultfloat << std::endl;
    else
        std::cout << std::dec << std::endl;
}

#endif // _ESIMD_RADIX_SORT_TEST_UTILS_H
