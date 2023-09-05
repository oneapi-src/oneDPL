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

#include "set_common.h"

#ifdef _PSTL_TEST_SET_UNION

template <typename T>
struct test_non_const_set_union
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_union(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(::std::less<T>()));
    }
};

#endif

int
main()
{
#ifdef _PSTL_TEST_SET_UNION
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const_set_union<std::int32_t>>());
#endif

    return TestUtils::done(_PSTL_TEST_SET_UNION);
}
