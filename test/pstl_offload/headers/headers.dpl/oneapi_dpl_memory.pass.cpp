// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/memory>
#include "support/utils.h"

int main() {
    int* i = reinterpret_cast<int*>(::operator new(sizeof(int)));
    oneapi::dpl::uninitialized_default_construct_n(oneapi::dpl::execution::seq, i, 1);
    ::operator delete(i);
    return TestUtils::done();
}
