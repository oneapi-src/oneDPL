// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

#ifndef _PSTL_PARALLEL_BACKEND_H
#define _PSTL_PARALLEL_BACKEND_H

#include "pstl_config.h"

#if defined(_PSTL_PAR_BACKEND_SERIAL)
#    include "parallel_backend_serial.h"
namespace pstl
{
namespace __par_backend = __serial_backend;
}
#elif defined(_PSTL_PAR_BACKEND_TBB)
#    include "parallel_backend_tbb.h"
namespace pstl
{
namespace __par_backend = __tbb_backend;
}
#else
_PSTL_PRAGMA_MESSAGE("Parallel backend was not specified");
#endif

#endif /* _PSTL_PARALLEL_BACKEND_H */
