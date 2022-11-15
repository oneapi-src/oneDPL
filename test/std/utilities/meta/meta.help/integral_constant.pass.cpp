//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// integral_constant
#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::cl_bool ret = false;
    typedef s::integral_constant<int, 5> _5;
    static_assert(_5::value == 5, "");
    static_assert((s::is_same<_5::value_type, int>::value), "");
    static_assert((s::is_same<_5::type, _5>::value), "");
#if TEST_STD_VER >= 11
    static_assert((_5() == 5), "");
#endif
    ret = (_5() == 5);

#if TEST_STD_VER > 11
    static_assert(_5{}() == 5, "");
    static_assert(s::true_type{}(), "");
#endif

    static_assert(s::false_type::value == false, "");
    static_assert((s::is_same<s::false_type::value_type, bool>::value), "");
    static_assert((s::is_same<s::false_type::type, s::false_type>::value), "");

    static_assert(s::true_type::value == true, "");
    static_assert((s::is_same<s::true_type::value_type, bool>::value), "");
    static_assert((s::is_same<s::true_type::type, s::true_type>::value), "");

    s::false_type f1;
    s::false_type f2 = f1;
    ret &= (!f2);

    s::true_type t1;
    s::true_type t2 = t1;
    ret &= (t2);

    return ret;
}

class KernelTest;
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
