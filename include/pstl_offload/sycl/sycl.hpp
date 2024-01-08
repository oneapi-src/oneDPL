// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_SYCL_SYCL_HPP

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_SYCL_SYCL_HPP
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

#include_next <sycl/sycl.hpp>

#ifdef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_SYCL_SYCL_HPP
#    include "internal/usm_memory_replacement.h"
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_SYCL_SYCL_HPP
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_SYCL_SYCL_HPP

#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_SYCL_SYCL_HPP
