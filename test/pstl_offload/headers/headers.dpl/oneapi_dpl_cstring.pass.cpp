// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/cstring>
#include "support/utils.h"

int main() {
    void* i = ::operator new(sizeof(int));
    oneapi::dpl::memset(i, 0, 1);
    ::operator delete(i);
    return TestUtils::done();
}
