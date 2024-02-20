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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

void
test_backend_tags()
{
    using namespace oneapi::dpl::__internal;

    // template <class _IsVector>
    // struct __serial_tag;
    static_assert(__is_backend_tag_v<__serial_tag<std::true_type>>);
    static_assert(__is_backend_tag_v<__serial_tag<std::false_type>>);

    // template <class _IsVector>
    // struct __parallel_tag
    static_assert(__is_backend_tag_v<__parallel_tag<std::true_type>>);
    static_assert(__is_backend_tag_v<__parallel_tag<std::false_type>>);

    // struct __parallel_forward_tag
    static_assert(__is_backend_tag_v<__parallel_forward_tag>);


#if TEST_DPCPP_BACKEND_PRESENT

    // template <typename _BackendTag>
    // struct __hetero_tag
    static_assert(__is_backend_tag_v<__hetero_tag<__device_backend_tag>>);

#if ONEDPL_FPGA_DEVICE

    // template <typename _BackendTag>
    // struct __hetero_tag
    static_assert(__is_backend_tag_v<__hetero_tag<__fpga_backend_tag>>);

#endif // ONEDPL_FPGA_DEVICE
#endif // TEST_DPCPP_BACKEND_PRESENT
}

void
test_backend_tags_serial()
{
    using namespace oneapi::dpl::__internal;

    // template <class _IsVector>
    // struct __serial_tag;
    static_assert(__is_backend_tag_serial_v<__serial_tag<std::true_type>>);
    static_assert(__is_backend_tag_serial_v<__serial_tag<std::false_type>>);

    // struct __parallel_forward_tag
    static_assert(!__is_backend_tag_serial_v<__parallel_forward_tag>);

    // template <class _IsVector>
    // struct __parallel_tag
    static_assert(!__is_backend_tag_serial_v<__parallel_tag<std::true_type>>);
    static_assert(!__is_backend_tag_serial_v<__parallel_tag<std::false_type>>);

#if TEST_DPCPP_BACKEND_PRESENT

    // template <typename _BackendTag>
    // struct __hetero_tag
    static_assert(!__is_backend_tag_serial_v<__hetero_tag<__device_backend_tag>>);

#if ONEDPL_FPGA_DEVICE

    // template <typename _BackendTag>
    // struct __hetero_tag
    static_assert(!__is_backend_tag_serial_v<__hetero_tag<__fpga_backend_tag>>);

#endif // ONEDPL_FPGA_DEVICE
#endif // TEST_DPCPP_BACKEND_PRESENT
}

std::int32_t
main()
{
    test_backend_tags();
    test_backend_tags_serial();

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}

