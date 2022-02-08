#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct S
{
    int i_;

    template <typename T>
    S(T&& i)
    {
        i_ = i;
    }

    S() : i_{0} {};
};

int
main()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() { s::tuple<S&&>(s::forward_as_tuple(S{})); });
        });
    }

    std::cout << "pass" << std::endl;
    return 0;
}
