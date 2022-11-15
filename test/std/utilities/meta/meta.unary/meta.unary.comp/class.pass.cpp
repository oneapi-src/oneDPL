//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// class

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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T>
void
test_class_imp()
{
    static_assert(!s::is_reference<T>::value, "");
    static_assert(!s::is_arithmetic<T>::value, "");
    static_assert(!s::is_fundamental<T>::value, "");
    static_assert(s::is_object<T>::value, "");
    static_assert(!s::is_scalar<T>::value, "");
    static_assert(s::is_compound<T>::value, "");
    static_assert(!s::is_member_pointer<T>::value, "");
}

template <class T>
void
test_class()
{
    test_class_imp<T>();
    test_class_imp<const T>();
    test_class_imp<volatile T>();
    test_class_imp<const volatile T>();
}

class Class
{
};

class incomplete_type;

cl::sycl::cl_bool
kernel_test()
{
    test_class<Class>();
    test_class<incomplete_type>();
    return true;
}

int
main(int, char**)
{
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
    return 0;
}
