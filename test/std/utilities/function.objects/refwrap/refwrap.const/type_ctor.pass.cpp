#include "oneapi_std_test_config.h"

#include <functional>
#include <iostream>

// <functional>

// reference_wrapper

// reference_wrapper(T& t);

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelTypeCtorTest;

class functor1
{
};

template <class T>
bool
test(T& t)
{
    std::reference_wrapper<T> r(t);
    return (&r.get() == &t);
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTypeCtorTest>([=]() {
                functor1 f1;
                ret_access[0] = test(f1);

                int i = 0;
                ret_access[0] &= test(i);
                const int j = 0;
                ret_access[0] &= test(j);
            });
        });
    }

    TestUtils::exitOnError(ret);
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
