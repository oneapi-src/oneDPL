//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_destructible

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
test_is_nothrow_destructible(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::is_nothrow_destructible<T>::value, "");
            static_assert(s::is_nothrow_destructible<const T>::value, "");
            static_assert(s::is_nothrow_destructible<volatile T>::value, "");
            static_assert(s::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_nothrow_destructible_v<T>, "");
            static_assert(s::is_nothrow_destructible_v<const T>, "");
            static_assert(s::is_nothrow_destructible_v<volatile T>, "");
            static_assert(s::is_nothrow_destructible_v<const volatile T>, "");
#endif
        });
    });
}

template <class T>
void
test_is_not_nothrow_destructible(cl::sycl::queue deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_nothrow_destructible<T>::value, "");
            static_assert(!s::is_nothrow_destructible<const T>::value, "");
            static_assert(!s::is_nothrow_destructible<volatile T>::value, "");
            static_assert(!s::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_nothrow_destructible_v<T>, "");
            static_assert(!s::is_nothrow_destructible_v<const T>, "");
            static_assert(!s::is_nothrow_destructible_v<volatile T>, "");
            static_assert(!s::is_nothrow_destructible_v<const volatile T>, "");
#endif
        });
    });
}

struct PublicDestructor
{
  public:
    ~PublicDestructor() {}
};
struct ProtectedDestructor
{
  protected:
    ~ProtectedDestructor() {}
};
struct PrivateDestructor
{
  private:
    ~PrivateDestructor() {}
};

class Empty
{
};

union Union {
};

struct bit_zero
{
    int : 0;
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_not_nothrow_destructible<void>(deviceQueue);
    test_is_not_nothrow_destructible<char[]>(deviceQueue);
    test_is_not_nothrow_destructible<char[][3]>(deviceQueue);

    test_is_nothrow_destructible<int&>(deviceQueue);
    test_is_nothrow_destructible<int>(deviceQueue);
    test_is_nothrow_destructible<int*>(deviceQueue);
    test_is_nothrow_destructible<const int*>(deviceQueue);
    test_is_nothrow_destructible<char[3]>(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_nothrow_destructible<double>(deviceQueue);
    }

    // requires noexcept. These are all destructible.
    test_is_nothrow_destructible<PublicDestructor>(deviceQueue);
    test_is_nothrow_destructible<bit_zero>(deviceQueue);
    test_is_nothrow_destructible<Empty>(deviceQueue);
    test_is_nothrow_destructible<Union>(deviceQueue);

    // requires access control
    test_is_not_nothrow_destructible<ProtectedDestructor>(deviceQueue);
    test_is_not_nothrow_destructible<PrivateDestructor>(deviceQueue);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
