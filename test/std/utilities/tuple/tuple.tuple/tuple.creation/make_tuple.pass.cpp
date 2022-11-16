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
// This code is used to test std::make_tuple
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelMakeTupleTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelMakeTupleTest>([=]() {
            int i = 0;
            float j = 0.f;
            s::tuple<int, int&, float&> t = s::make_tuple(1, s::ref(i), s::ref(j));

            ret_access[0] = (s::get<0>(t) == 1);
            ret_access[0] &= (s::get<1>(t) == 0);
            ret_access[0] &= (s::get<2>(t) == 0.f);

            i = 2;
            j = 3.5f;
            ret_access[0] &= (s::get<0>(t) == 1);
            ret_access[0] &= (s::get<1>(t) == 2);
            ret_access[0] &= (s::get<2>(t) == 3.5f);

            s::get<1>(t) = 0;
            s::get<2>(t) = 0.f;
            ret_access[0] &= (i == 0);
            ret_access[0] &= (j == 0.f);
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
