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
    void* malloc_allocated;
    int* aligned_new_allocated;
    void* aligned_alloc_allocated;
};

#if free_after_unload_lib_EXPORTS && _WIN64
__declspec(dllexport)
#endif
void register_mem_to_later_release(pointers*);

#endif // _FREE_AFTER_UNLOAD_LIB_HEADERS_H
