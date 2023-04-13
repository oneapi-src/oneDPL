#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class KernelMinusTest;

void
kernel_test(sycl::queue deviceQueue)
{
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelMinusTest>([=]() {
            typedef s::minus<int> Fint;
            const Fint f1 = Fint();
            ret_access[0] = (f1(3, 7) == -4);

            typedef s::minus<float> Ffloat;
            const Ffloat f2 = Ffloat();
            ret_access[0] &= (f2(3, 2.5) == 0.5);
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    int is_done = 0;

#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test(deviceQueue);
        is_done = 1;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(is_done);
}
