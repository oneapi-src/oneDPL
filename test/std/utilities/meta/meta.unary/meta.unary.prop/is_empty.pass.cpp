//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_empty

// T is a non-union class type with:
//  no non-static data members,
//  no unnamed bit-fields of non-zero length,
//  no virtual member functions,
//  no virtual base classes,
//  and no base class B for which is_empty_v<B> is false.

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

template <class T>
void
test_is_empty(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::is_empty<T>::value, "");
            static_assert(s::is_empty<const T>::value, "");
            static_assert(s::is_empty<volatile T>::value, "");
            static_assert(s::is_empty<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_empty_v<T>, "");
            static_assert(s::is_empty_v<const T>, "");
            static_assert(s::is_empty_v<volatile T>, "");
            static_assert(s::is_empty_v<const volatile T>, "");
#endif
        });
    });
}

template <class T>
void
test_is_not_empty(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_empty<T>::value, "");
            static_assert(!s::is_empty<const T>::value, "");
            static_assert(!s::is_empty<volatile T>::value, "");
            static_assert(!s::is_empty<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_empty_v<T>, "");
            static_assert(!s::is_empty_v<const T>, "");
            static_assert(!s::is_empty_v<volatile T>, "");
            static_assert(!s::is_empty_v<const volatile T>, "");
#endif
        });
    });
}

class Empty
{
};
struct NotEmpty
{
    int foo;
};

union Union {
};

struct EmptyBase : public Empty
{
};
struct NotEmptyBase : public NotEmpty
{
};

struct NonStaticMember
{
    int foo;
};

struct bit_zero
{
    int : 0;
};

struct bit_one
{
    int : 1;
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_not_empty<void>(deviceQueue);
    test_is_not_empty<int&>(deviceQueue);
    test_is_not_empty<int>(deviceQueue);
    test_is_not_empty<int*>(deviceQueue);
    test_is_not_empty<const int*>(deviceQueue);
    test_is_not_empty<char[3]>(deviceQueue);
    test_is_not_empty<char[]>(deviceQueue);
    test_is_not_empty<Union>(deviceQueue);
    test_is_not_empty<NotEmpty>(deviceQueue);
    test_is_not_empty<NotEmptyBase>(deviceQueue);
    test_is_not_empty<NonStaticMember>(deviceQueue);

    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_not_empty<double>(deviceQueue);
    }

    test_is_empty<Empty>(deviceQueue);
    test_is_empty<EmptyBase>(deviceQueue);
    test_is_empty<bit_zero>(deviceQueue);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    std::cout << "Pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
