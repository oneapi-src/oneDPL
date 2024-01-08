// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "support/test_dynamic_selection_one_policy.h"

int
test_no_customizations()
{
    int trace = 0;
    one_with_no_customizations p(trace);

    trace = 0;
    auto s = oneapi::dpl::experimental::select(p);
    if (trace != t_select)
    {
        std::cout << "ERROR: unexpected trace of select " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit(s, [](int i) { return i; });
    if (trace != t_submit_selection)
    {
        std::cout << "ERROR: unexpected trace of submit selection: " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    if (trace != (t_select | t_submit_selection))
    {
        std::cout << "ERROR: unexpected trace of submit function " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(s, [](int i) { return i; });
    if (trace != (t_submit_selection | t_wait))
    {
        std::cout << "ERROR: unexpected trace of submit_and_wait selection " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    if (trace != (t_select | t_submit_selection | t_wait))
    {
        std::cout << "ERROR: unexpected trace of submit_and_wait function " << trace << "\n";
        return 1;
    }
    std::cout << "test_no_customizations: OK\n";

    return 0;
}

int
test_all_customizations()
{
    int trace = 0;
    one_with_all_customizations p(trace);

    trace = 0;
    auto s = oneapi::dpl::experimental::select(p);
    if (trace != t_select)
    {
        std::cout << "ERROR: unexpected trace of select " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit(s, [](int i) { return i; });
    if (trace != t_submit_selection)
    {
        std::cout << "ERROR: unexpected trace of submit selection: " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    if (trace != (t_submit_function))
    {
        std::cout << "ERROR: unexpected trace of submit function " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(s, [](int i) { return i; });
    if (trace != (t_submit_and_wait_selection))
    {
        std::cout << "ERROR: unexpected trace of submit_and_wait selection " << trace << "\n";
        return 1;
    }

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    if (trace != (t_submit_and_wait_function))
    {
        std::cout << "ERROR: unexpected trace of submit_and_wait function " << trace << "\n";
        return 1;
    }

    std::cout << "test_all_customizations: OK\n";
    return 0;
}

int
main()
{
    if (test_no_customizations() || test_all_customizations())
    {
        std::cout << "FAIL\n";
        return 1;
    }
    else
    {
        std::cout << "PASS\n";
        return 0;
    }
}
