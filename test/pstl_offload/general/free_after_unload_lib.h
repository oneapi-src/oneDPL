// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _FREE_AFTER_UNLOAD_LIB_HEADERS_H
#define _FREE_AFTER_UNLOAD_LIB_HEADERS_H

struct pointers
{
    void *p1;
    int *p2;
    void *p3;
};

#if free_after_unload_lib_EXPORTS && _WIN64
__declspec(dllexport)
#endif
void register_mem_to_later_release(pointers *);

#endif // _FREE_AFTER_UNLOAD_LIB_HEADERS_H
