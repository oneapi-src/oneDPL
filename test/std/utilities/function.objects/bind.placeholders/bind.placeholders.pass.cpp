#include "oneapi_std_test_config.h"

#include <iostream>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class Object
{
  public:
    void
    operator()(int a, int b, int c, int& i)
    {
        i += p;
        i += a;
        i *= b;
        i -= c;
    }

  private:
    int p = 10;
};

class KernelTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::cl_int result = 10;
    {
        cl::sycl::buffer<cl::sycl::cl_int, 1> buffer1(&result, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto res_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                Object instance;
                auto bf = s::bind(instance, 1, 2, 3, s::ref(res_access[0]));
                bf();
            });
        });
    }
    TestUtils::exitOnError(result == 39);
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
