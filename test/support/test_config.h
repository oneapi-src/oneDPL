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

#ifndef _TEST_CONFIG_H
#define _TEST_CONFIG_H

// Any include from standard library required to have correct state of _GLIBCXX_RELEASE
#if __has_include(<version>)
#   include <version>
#else
#   include <ciso646>
#endif

#define _PSTL_TEST_STRING(X) _PSTL_TEST_STRING_AUX(oneapi/dpl/X)
#define _PSTL_TEST_STRING_AUX(X) #X
//to support the optional including: <algorithm>, <memory>, <numeric> or <pstl/algorithm>, <pstl/memory>, <pstl/numeric>
#define _PSTL_TEST_HEADER(HEADER_ID) _PSTL_TEST_STRING(HEADER_ID)

#if defined(_MSC_VER) && defined(_DEBUG)
#define _SCL_SECURE_NO_WARNINGS //to prevent the compilation warning. Microsoft STL implementation has specific checking of an iterator range in DEBUG mode for the containers from the standard library.
#endif

// ICC 18 (Windows) has encountered an unexpected problem on some tests
#define _PSTL_ICC_18_VC141_TEST_SIMD_LAMBDA_RELEASE_BROKEN                                                            \
    (!_DEBUG && __INTEL_COMPILER >= 1800 && __INTEL_COMPILER < 1900 && _MSC_VER == 1910)
// ICC 18 doesn't vectorize the loop
#define _PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN (!_DEBUG && __INTEL_COMPILER && __INTEL_COMPILER == 1800)
// ICC 18 generates wrong result with omp simd early_exit
#define _PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN                                                               \
    (!_DEBUG && __INTEL_COMPILER == 1800 && __AVX__ && !__AVX2__ && !__AVX512__)
// ICC 19 has encountered an unexpected problem: Segmentation violation signal raised
#define _PSTL_ICC_19_TEST_IS_PARTITIONED_RELEASE_BROKEN                                                               \
    (!PSTL_USE_DEBUG && (__linux__ || __APPLE__) && __INTEL_COMPILER == 1900)
// ICC 19 generates wrong result with UDS on Windows
#define _PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN (__INTEL_COMPILER == 1900 && _MSC_VER && !_DEBUG)
// ICPC compiler generates wrong "openMP simd" code for a user defined scan operation(UDS) for MacOS, Linuxand Windows
#define _PSTL_ICC_TEST_SIMD_UDS_BROKEN                                                                                \
    (__INTEL_COMPILER && __INTEL_COMPILER_BUILD_DATE < 20211123)
// ICC 18,19 generate wrong result
#define _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN													  \
    ((__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900) && _MSC_VER && !_DEBUG)
// ICC 18,19 generate wrong result with for_loop_strided and reverse iterators
#define _PSTL_ICC_18_19_TEST_REVERSE_ITERATOR_WITH_STRIDE_BROKEN                                                      \
    (__i386__ && (__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900))
// VC14 uninitialized_fill with no policy has broken implementation
#define _PSTL_STD_UNINITIALIZED_FILL_BROKEN (_MSC_VER == 1900)
// GCC10 produces wrong answer calling exclusive_scan using vectorized polices
#define TEST_GCC10_EXCLUSIVE_SCAN_BROKEN (_GLIBCXX_RELEASE == 10)
// GCC7 std::get doesn't return const rvalue reference from const rvalue reference of tuple
#define _PSTL_TEST_GCC7_RVALUE_TUPLE_GET_BROKEN (_GLIBCXX_RELEASE > 0 && _GLIBCXX_RELEASE < 8)
// Array swap broken on Windows because Microsoft implementation of std::swap function for std::array
// call some internal function which is not declared as SYCL external and we have compile error
#if defined(_MSC_VER)
#   define TEST_XPU_ARRAY_SWAP_BROKEN (_MSC_VER <= 1937)
#else
#   define TEST_XPU_ARRAY_SWAP_BROKEN 0
#endif

#define _PSTL_SYCL_TEST_USM 1

// Enable test when the DPC++ backend is available
#if ((defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) &&                                         \
     (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))) &&                                             \
    (!defined(ONEDPL_USE_DPCPP_BACKEND) || ONEDPL_USE_DPCPP_BACKEND != 0)
#define TEST_DPCPP_BACKEND_PRESENT 1
#else
#define TEST_DPCPP_BACKEND_PRESENT 0
#endif

#ifdef __SYCL_UNNAMED_LAMBDA__
#define TEST_UNNAMED_LAMBDAS 1
#else
#define TEST_UNNAMED_LAMBDAS 0
#endif

// The TEST_EXPLICIT_KERNEL_NAMES macro may be defined on CMake level in CMakeLists.txt
// so we should check here if it is defined or not
#ifndef TEST_EXPLICIT_KERNEL_NAMES
#    if __SYCL_UNNAMED_LAMBDA__
#        define TEST_EXPLICIT_KERNEL_NAMES 0
#    else
#        define TEST_EXPLICIT_KERNEL_NAMES 1
#    endif // __SYCL_UNNAMED_LAMBDA__
#endif // !TEST_EXPLICIT_KERNEL_NAMES

// Enables full scope of testing
#ifndef TEST_LONG_RUN
#define TEST_LONG_RUN 0
#endif

// Enable test when the TBB backend is available
#if !defined(ONEDPL_USE_TBB_BACKEND) || ONEDPL_USE_TBB_BACKEND
#define TEST_TBB_BACKEND_PRESENT 1
#endif

// Check for C++ standard and standard library for the use of ranges API
#if !defined(_ENABLE_RANGES_TESTING)
#define _TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY TEST_DPCPP_BACKEND_PRESENT
#if defined(_GLIBCXX_RELEASE)
#    define _ENABLE_RANGES_TESTING (_TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY && _GLIBCXX_RELEASE >= 8 && __GLIBCXX__ >= 20180502)
#elif defined(_LIBCPP_VERSION)
#    define _ENABLE_RANGES_TESTING (_TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY && _LIBCPP_VERSION >= 7000)
#else
#    define _ENABLE_RANGES_TESTING (_TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY)
#endif
#endif //!defined(_ENABLE_RANGES_TESTING)

#define TEST_HAS_NO_INT128
#define _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE (_MSVC_STL_VERSION < 143)

#define _PSTL_TEST_COMPLEX_OP_BROKEN_INTEL_LLVM_COMPILER __INTEL_LLVM_COMPILER
#define _PSTL_TEST_COMPLEX_OP_BROKEN_CLANG __clang__
#define _PSTL_TEST_COMPLEX_OP_BROKEN_MSVC_STL (_MSVC_STL_VERSION && _MSVC_STL_VERSION <= 143)
#define _PSTL_TEST_COMPLEX_OP_BROKEN_MSVC_STL_LLVM_COMPILER (_MSVC_STL_VERSION && __INTEL_LLVM_COMPILER)
#define _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN (__GLIBCXX__ >= 7)
#define _PSTL_MSVC_LESS_THAN_CPP20_COMPLEX_CONSTEXPR_BROKEN (_MSC_VER && __cplusplus < 202002L && _MSVC_LANG < 202002L)

#define _PSTL_ICC_TEST_UNDERLYING_TYPE_BROKEN (_GLIBCXX_RELEASE && _GLIBCXX_RELEASE < 9)

// Known limitation:
// Due to specifics of Microsoft* Visual C++, some standard floating-point math functions require device support for double precision.
#define _PSTL_ICC_TEST_COMPLEX_MSVC_MATH_DOUBLE_REQ _MSC_VER

#define TEST_DYNAMIC_SELECTION_AVAILABLE (TEST_DPCPP_BACKEND_PRESENT && __INTEL_LLVM_COMPILER >= 20230000)

// oneAPI DPC++ compiler in 2023.2 release build crashes during optimization of reduce_by_segment.pass.cpp
// with TBB backend.
#if !PSTL_USE_DEBUG && TEST_TBB_BACKEND_PRESENT && defined(__INTEL_LLVM_COMPILER)
#   define _PSTL_ICPX_TEST_RED_BY_SEG_OPTIMIZER_CRASH ((__INTEL_LLVM_COMPILER >= 20230200) && (__INTEL_LLVM_COMPILER <= 20240100))
#else
#   define _PSTL_ICPX_TEST_RED_BY_SEG_OPTIMIZER_CRASH 0
#endif

// If the workaround macro for the 64-bit type bug is not defined by the user, then exclude 64-bit type testing
// in reduce_by_segment.pass.cpp.
// TODO: When a driver fix is provided to resolve this issue, consider altering this macro or checking the driver version at runtime
// of the underlying sycl::device to determine whether to include or exclude 64-bit type tests.
#if !PSTL_USE_DEBUG && defined(__INTEL_LLVM_COMPILER)
#    define _PSTL_ICPX_TEST_RED_BY_SEG_BROKEN_64BIT_TYPES 1
#endif

// Group reduction produces wrong results with multiplication of 64-bit for certain driver versions
// TODO: When a driver fix is provided to resolve this issue, consider altering this macro or checking the driver version at runtime
// of the underlying sycl::device to determine whether to include or exclude 64-bit type tests.
#define _PSTL_GROUP_REDUCTION_MULT_INT64_BROKEN 1

// oneAPI DPC++ compiler 2022.2 an below show an internal compiler error during the backend code generation of
// minmax_element.pass.cpp
#define _PSTL_ICPX_TEST_MINMAX_ELEMENT_BROKEN                                                                         \
    (TEST_DPCPP_BACKEND_PRESENT && defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20220300)

#endif // _TEST_CONFIG_H
