// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_CONCURRENT_VECTOR_H

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_CONCURRENT_VECTOR_H
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

#include_next <oneapi/tbb/concurrent_vector.h>

#ifdef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_CONCURRENT_VECTOR_H
#    include "internal/usm_memory_replacement.h"
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_CONCURRENT_VECTOR_H
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_CONCURRENT_VECTOR_H

#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_CONCURRENT_VECTOR_H
