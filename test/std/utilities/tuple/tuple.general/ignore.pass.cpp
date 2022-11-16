// <tuple>

// constexpr unspecified ignore;

// UNSUPPORTED: c++98, c++03

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <functional>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelIgnoreTest;

bool __attribute__((always_inline)) test_ignore_constexpr()
{
    bool ret = false;
    { // Test that s::ignore provides constexpr converting assignment.
        auto& res = (s::ignore = 42);
        ret = (&res == &s::ignore);
    }
    { // Test that s::ignore provides constexpr copy/move constructors
        auto copy = s::ignore;
        auto moved = s::move(copy);
        ((void)moved);
    }
    { // Test that s::ignore provides constexpr copy/move assignment
        auto copy = s::ignore;
        copy = s::ignore;
        auto moved = s::ignore;
        moved = s::move(copy);
    }
    return ret;
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelIgnoreTest>([=]() {
            {
                constexpr auto& ignore_v = s::ignore;
                ((void)ignore_v);
            }

            {
                ret_access[0] = test_ignore_constexpr();
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
