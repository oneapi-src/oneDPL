// <utility>

// template <class T1, class T2> struct pair

// void swap(pair& p);

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct S
{
    int i;
    S() : i(0) {}
    S(int j) : i(j) {}
    S* operator&()
    {
        assert(false);
        return this;
    }
    S const* operator&() const
    {
        assert(false);
        return this;
    }
    bool
    operator==(int x) const
    {
        return i == x;
    }
};

class KernelPairTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef s::pair<int, short> P1;
                P1 p1(3, static_cast<short>(4));
                P1 p2(5, static_cast<short>(6));
                p1.swap(p2);
                ret_access[0] = (p1.first == 5);
                ret_access[0] &= (p1.second == 6);
                ret_access[0] &= (p2.first == 3);
                ret_access[0] &= (p2.second == 4);
            }
            {
                typedef s::pair<int, S> P1;
                P1 p1(3, S(4));
                P1 p2(5, S(6));
                p1.swap(p2);
                ret_access[0] &= (p1.first == 5);
                ret_access[0] &= (p1.second == 6);
                ret_access[0] &= (p2.first == 3);
                ret_access[0] &= (p2.second == 4);
            }
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
