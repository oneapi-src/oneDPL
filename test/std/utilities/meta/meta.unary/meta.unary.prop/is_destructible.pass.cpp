//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_destructible

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
test_is_destructible(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(s::is_destructible<T>::value, "");
            static_assert(s::is_destructible<const T>::value, "");
            static_assert(s::is_destructible<volatile T>::value, "");
            static_assert(s::is_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(s::is_destructible_v<T>, "");
            static_assert(s::is_destructible_v<const T>, "");
            static_assert(s::is_destructible_v<volatile T>, "");
            static_assert(s::is_destructible_v<const volatile T>, "");
#endif
        });
    });
}

template <class T>
void
test_is_not_destructible(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!s::is_destructible<T>::value, "");
            static_assert(!s::is_destructible<const T>::value, "");
            static_assert(!s::is_destructible<volatile T>::value, "");
            static_assert(!s::is_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
            static_assert(!s::is_destructible_v<T>, "");
            static_assert(!s::is_destructible_v<const T>, "");
            static_assert(!s::is_destructible_v<volatile T>, "");
            static_assert(!s::is_destructible_v<const volatile T>, "");
#endif
        });
    });
}

class Empty
{
};

union Union {
};

struct bit_zero
{
    int : 0;
};

typedef void(Function)();

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

struct DeletedPublicDestructor
{
  public:
    ~DeletedPublicDestructor() = delete;
};
struct DeletedProtectedDestructor
{
  protected:
    ~DeletedProtectedDestructor() = delete;
};
struct DeletedPrivateDestructor
{
  private:
    ~DeletedPrivateDestructor() = delete;
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    test_is_destructible<int&>(deviceQueue);
    test_is_destructible<Union>(deviceQueue);
    test_is_destructible<Empty>(deviceQueue);
    test_is_destructible<int>(deviceQueue);
    test_is_destructible<int*>(deviceQueue);
    test_is_destructible<const int*>(deviceQueue);
    test_is_destructible<char[3]>(deviceQueue);
    test_is_destructible<bit_zero>(deviceQueue);
    test_is_destructible<int[3]>(deviceQueue);
    test_is_destructible<PublicDestructor>(deviceQueue);

    test_is_not_destructible<int[]>(deviceQueue);
    test_is_not_destructible<void>(deviceQueue);
    test_is_not_destructible<Function>(deviceQueue);

    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        test_is_destructible<double>(deviceQueue);
    }

    // Test access controlled destructors
    test_is_not_destructible<ProtectedDestructor>(deviceQueue);
    test_is_not_destructible<PrivateDestructor>(deviceQueue);

    // Test deleted constructors
    test_is_not_destructible<DeletedPublicDestructor>(deviceQueue);
    test_is_not_destructible<DeletedProtectedDestructor>(deviceQueue);
    test_is_not_destructible<DeletedPrivateDestructor>(deviceQueue);
}

int
main(int, char**)
{
    kernel_test();
    std::cout << "Pass" << std::endl;
    return 0;
}
