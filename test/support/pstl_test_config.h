// -*- C++ -*-
//===-- pstl_test_config.h ------------------------------------------------===//
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

#ifndef _PSTL_TEST_config_H
#define _PSTL_TEST_config_H

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
// clang 900.0.38 produces fatal error: error in backend: Section too large
#define _PSTL_CLANG_TEST_BIG_OBJ_DEBUG_32_BROKEN                                                                      \
    (__i386__ && PSTL_USE_DEBUG && __clang__ && _PSTL_CLANG_VERSION <= 90000)
// ICC 18 generates wrong result with omp simd early_exit
#define _PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN                                                               \
    (!_DEBUG && __INTEL_COMPILER == 1800 && __AVX__ && !__AVX2__ && !__AVX512__)
// ICC 19 has encountered an unexpected problem: Segmentation violation signal raised
#define _PSTL_ICC_19_TEST_IS_PARTITIONED_RELEASE_BROKEN                                                               \
    (!PSTL_USE_DEBUG && (__linux__ || __APPLE__) && __INTEL_COMPILER == 1900)
// ICC 19 has encountered an unexpected problem: Segmentation violation signal raised
#define _PSTL_ICL_19_VC14_VC141_TEST_SCAN_RELEASE_BROKEN                                                              \
    (__INTEL_COMPILER == 1900 && _MSC_VER >= 1900 && _MSC_VER <= 1910)
// ICC 19 generates wrong result with UDS on Windows
#define _PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN (__INTEL_COMPILER == 1900 && _MSC_VER && !_DEBUG)
// ICC 18,19 generate wrong result
#define _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN													  \
    ((__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900) && _MSC_VER && !_DEBUG)
// ICC 18,19 generate wrong result with for_loop_strided and reverse iterators
#define _PSTL_ICC_18_19_TEST_REVERSE_ITERATOR_WITH_STRIDE_BROKEN                                                      \
    (__i386__ && (__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900))
// VC14 uninitialized_fill with no policy has broken implementation
#define _PSTL_STD_UNINITIALIZED_FILL_BROKEN (_MSC_VER == 1900)

#define _PSTL_SYCL_TEST_USM 1

// Check for C++ standard and standard library for the use of ranges API
#define _ONEDPL_USE_RANGES                                                                                            \
    (__cplusplus >= 201703L && ((_GLIBCXX_RELEASE >= 8 && __GLIBCXX__ >= 20180502) || _LIBCPP_VERSION >= 7000))

#endif /* _PSTL_TEST_config_H */
