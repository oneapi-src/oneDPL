// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <new>
#include "support/utils.h"

int main() {
    void* ptr = operator new(1);
    operator delete(ptr);
    return TestUtils::done();
}
