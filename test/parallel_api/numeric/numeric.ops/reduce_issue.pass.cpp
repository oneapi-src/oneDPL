// -*- C++ -*-
//===-- reduce.pass.cpp ---------------------------------------------------===//
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

#include "support/test_config.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

// to check __has_known_identity
#include <oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h>

#include "support/utils.h"

#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

int
main(int argc, const char* argv[])
{
    sycl::queue q{sycl::default_selector_v};
    const sycl::device& d = q.get_device();

    const std::string& name = d.get_info<sycl::info::device::name>();
    const std::string& driver_version = d.get_info<sycl::info::device::driver_version>();

    std::cout << "Device " << name << " [" << driver_version << "]" << std::endl;
    std::cout << "_USE_GROUP_ALGOS = " << _USE_GROUP_ALGOS << std::endl;
    std::cout << "_ONEDPL_SYCL_INTEL_COMPILER = " << _ONEDPL_SYCL_INTEL_COMPILER << std::endl;
    std::cout << "_ONEDPL_SYCL2020_COLLECTIVES_PRESENT = " << _ONEDPL_SYCL2020_COLLECTIVES_PRESENT << std::endl;

    constexpr size_t N = 6;
    using T = std::int64_t;

    using __has_known_identity_t = typename oneapi::dpl::unseq_backend::__has_known_identity<std::multiplies<T>, T>::type;
    std::cout << "__has_known_identity_t = " << __has_known_identity_t{} << std::endl;

    T* data = sycl::malloc_shared<T>(N + 1, q);

    for (int i = 0; i < N; ++i)
    {
        data[i] = T(i + 1);
        std::cout << data[i] << " ";
    }

    const T init = T(2);
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    T acc = std::reduce(policy, data, data + N, init, std::multiplies<T>());

    std::cout << std::endl << "Result: " << acc << std::endl;
    sycl::free(data, q);

    return 0;
}