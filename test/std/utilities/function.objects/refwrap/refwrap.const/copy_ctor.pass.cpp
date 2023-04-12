#include "oneapi_std_test_config.h"

#include <functional>
#include <iostream>

// <functional>

// reference_wrapper

// reference_wrapper(const reference_wrapper<T>& x);

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class KernelCopyCtorTest;

class functor1
{
};

template <class T>
bool
test(T& t)
{
    std::reference_wrapper<T> r(t);
    std::reference_wrapper<T> r2 = r;
    return (&r2.get() == &t);
}

void
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelCopyCtorTest>([=]() {
            functor1 f1;
            ret_access[0] = test(f1);
            int i = 0;
            ret_access[0] &= test(i);
            const int j = 0;
            ret_access[0] &= test(j);
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
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
