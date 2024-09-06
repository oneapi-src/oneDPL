// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is a simple standalone SYCL test which is meant to prove that the SYCL installation is correct.
// If this test fails, it means that the SYCL environment has not be configured properly.

#include <cstdlib>
#include <iostream>

#if ((defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) &&                                         \
     (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))) &&                                             \
    (!defined(ONEDPL_USE_DPCPP_BACKEND) || ONEDPL_USE_DPCPP_BACKEND != 0)
#define TEST_DPCPP_BACKEND_PRESENT 1
#else
#define TEST_DPCPP_BACKEND_PRESENT 0
#endif

#if TEST_DPCPP_BACKEND_PRESENT

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif


// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define TEST_LIBSYCL_VERSION                                                                                    \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define TEST_LIBSYCL_VERSION 0
#endif

#define _SKIP_RETURN_CODE 77

inline auto default_selector =
#    if TEST_LIBSYCL_VERSION >= 60000
        sycl::default_selector_v;
#    else
        sycl::default_selector{};
#    endif

class canary_test_name;

void
test()
{
    sycl::queue q(default_selector);
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<canary_test_name>([=]() {});
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test();
#endif
    if (std::getenv("_ONEDPL_SKIP_SYCL_CANARY_TEST") != nullptr)
    {
        // This environment variable allows our main CI run to skip this test and not count it toward oneDPL's test
        // statistics, while still allowing non-ci test runs to have this as a environment health indicater.
        std::cout << "Skipped\n";
        return _SKIP_RETURN_CODE;
    }
    else
        return 0;
}