//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_object

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <type_traits>
#    include <cstddef> // for std::nullptr_t
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <class T>
void
test_is_object()
{
    static_assert(s::is_object<T>::value, "");
    static_assert(s::is_object<const T>::value, "");
    static_assert(s::is_object<volatile T>::value, "");
    static_assert(s::is_object<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(s::is_object_v<T>, "");
    static_assert(s::is_object_v<const T>, "");
    static_assert(s::is_object_v<volatile T>, "");
    static_assert(s::is_object_v<const volatile T>, "");
#endif
}

template <class T>
void
test_is_not_object()
{
    static_assert(!s::is_object<T>::value, "");
    static_assert(!s::is_object<const T>::value, "");
    static_assert(!s::is_object<volatile T>::value, "");
    static_assert(!s::is_object<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!s::is_object_v<T>, "");
    static_assert(!s::is_object_v<const T>, "");
    static_assert(!s::is_object_v<volatile T>, "");
    static_assert(!s::is_object_v<const volatile T>, "");
#endif
}

class incomplete_type;

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
};

union Union {
};

struct bit_zero
{
    int : 0;
};

class Abstract
{
    virtual ~Abstract() = 0;
};

enum Enum
{
    zero,
    one
};

typedef void (*FunctionPtr)();

cl::sycl::cl_bool
kernel_test()
{
    // An object type is a (possibly cv-qualified) type that is not a function
    // type, not a reference type, and not a void type.
    test_is_object<s::nullptr_t>();
    test_is_object<void*>();
    test_is_object<char[3]>();
    test_is_object<char[]>();
    test_is_object<int>();
    test_is_object<int*>();
    test_is_object<Union>();
    test_is_object<int*>();
    test_is_object<const int*>();
    test_is_object<Enum>();
    test_is_object<incomplete_type>();
    test_is_object<bit_zero>();
    test_is_object<NotEmpty>();
    test_is_object<Abstract>();
    test_is_object<FunctionPtr>();
    test_is_object<int Empty::*>();
    test_is_object<void (Empty::*)(int)>();

    test_is_not_object<void>();
    test_is_not_object<int&>();
    test_is_not_object<int&&>();
    test_is_not_object<int(int)>();
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
