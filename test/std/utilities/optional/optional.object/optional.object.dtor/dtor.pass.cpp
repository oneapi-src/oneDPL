//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// ~optional();

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(optional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <optional>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
using s::optional;

struct PODType
{
    int value;
    int value2;
};

class X
{
  public:
    bool dtor_called = false;
    X() = default;
    ~X() { dtor_called = true; }
};

bool
kernel_test()
{
    cl::sycl::queue q;
    bool ret = true;
    cl::sycl::range<1> numOfItems1{1};
    {
        cl::sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);

        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    typedef int T;
                    static_assert(s::is_trivially_destructible<T>::value, "");
                    static_assert(s::is_trivially_destructible<optional<T>>::value, "");
                }
                {
                    typedef double T;
                    static_assert(s::is_trivially_destructible<T>::value, "");
                    static_assert(s::is_trivially_destructible<optional<T>>::value, "");
                }
                {
                    typedef PODType T;
                    static_assert(s::is_trivially_destructible<T>::value, "");
                    static_assert(s::is_trivially_destructible<optional<T>>::value, "");
                }
                {
                    typedef X T;
                    static_assert(!s::is_trivially_destructible<T>::value, "");
                    static_assert(!s::is_trivially_destructible<optional<T>>::value, "");
                    {
                        X x;
                        optional<X> opt{x};
                        ret_access[0] &= (opt->dtor_called == false);
                    }
                }
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
