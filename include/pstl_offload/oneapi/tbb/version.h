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

#ifndef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_VERSION_H
#    define _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL

#include_next <oneapi/tbb/version.h>

#ifdef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_VERSION_H
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_VERSION_H
#    include "internal/usm_memory_replacement.h"
#    undef _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL
#endif // _ONEDPL_PSTL_OFFLOAD_TOP_LEVEL_ONEAPI_TBB_VERSION_H
