//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// underlying_type
//  As of C++20, std::underlying_type is SFINAE-friendly; if you hand it
//  a non-enumeration, it returns an empty struct.

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

#if TEST_STD_VER > 17
template <class, class = s::void_t<>>
struct has_type_member : s::false_type
{
};

template <class T>
struct has_type_member<T, s::void_t<typename s::underlying_type<T>::type>> : s::true_type
{
};

struct S
{
};
union U {
    int i;
    float f;
};
#endif

template <typename T, typename Expected>
void
check()
{
    ASSERT_SAME_TYPE(Expected, typename s::underlying_type<T>::type);
#if TEST_STD_VER > 11
    ASSERT_SAME_TYPE(Expected, typename s::underlying_type_t<T>);
#endif
}

enum E
{
    V = INT_MIN
};

enum G : char
{
};
enum class H
{
    red,
    green = 20,
    blue
};
enum class I : long
{
    red,
    green = 20,
    blue
};
enum struct J
{
    red,
    green = 20,
    blue
};
enum struct K : short
{
    red,
    green = 20,
    blue
};

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            //  Basic tests
            check<E, int>();

            //  Class enums and enums with specified underlying type
            check<G, char>();
            check<H, int>();
            check<I, long>();
            check<J, int>();
            check<K, short>();

//  SFINAE-able underlying_type
#if TEST_STD_VER > 17
            static_assert(has_type_member<E>::value, "");
            static_assert(has_type_member<F>::value, "");
            static_assert(has_type_member<G>::value, "");

            static_assert(!has_type_member<void>::value, "");
            static_assert(!has_type_member<int>::value, "");
            static_assert(!has_type_member<int[]>::value, "");
            static_assert(!has_type_member<S>::value, "");
            static_assert(!has_type_member<void (S::*)(int)>::value, "");
            static_assert(!has_type_member<void (S::*)(int, ...)>::value, "");
            static_assert(!has_type_member<U>::value, "");
            static_assert(!has_type_member<void(int)>::value, "");
            static_assert(!has_type_member<void(int, ...)>::value, "");
            static_assert(!has_type_member<int&>::value, "");
            static_assert(!has_type_member<int&&>::value, "");
            static_assert(!has_type_member<int*>::value, "");
            static_assert(!has_type_member<s::nullptr_t>::value, "");
#endif
        });
    });
}

void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
#if TEST_STD_VER > 17
            static_assert(!has_type_member<double>::value, "");
#endif
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    std::cout << "Pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
