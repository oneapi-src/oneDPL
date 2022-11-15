// <utility>

// template <class T1, class T2> struct pair

// template <class T1, class T2> bool operator==(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator!=(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator< (const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator> (const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator>=(const pair<T1,T2>&, const pair<T1,T2>&);
// template <class T1, class T2> bool operator<=(const pair<T1,T2>&, const pair<T1,T2>&);

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
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelPairTest;
void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelPairTest>([=]() {
            {
                typedef s::pair<int, short> P;
                P p1(3, static_cast<short>(4));
                P p2(3, static_cast<short>(4));
                ret_access[0] = ((p1 == p2));
                ret_access[0] &= (!(p1 != p2));
                ret_access[0] &= (!(p1 < p2));
                ret_access[0] &= ((p1 <= p2));
                ret_access[0] &= (!(p1 > p2));
                ret_access[0] &= ((p1 >= p2));
            }

            {
                typedef s::pair<int, short> P;
                P p1(2, static_cast<short>(4));
                P p2(3, static_cast<short>(4));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= ((p1 < p2));
                ret_access[0] &= ((p1 <= p2));
                ret_access[0] &= (!(p1 > p2));
                ret_access[0] &= (!(p1 >= p2));
            }

            {
                typedef s::pair<int, short> P;
                P p1(3, static_cast<short>(2));
                P p2(3, static_cast<short>(4));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= ((p1 < p2));
                ret_access[0] &= ((p1 <= p2));
                ret_access[0] &= (!(p1 > p2));
                ret_access[0] &= (!(p1 >= p2));
            }

            {
                typedef s::pair<int, short> P;
                P p1(3, static_cast<short>(4));
                P p2(2, static_cast<short>(4));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= (!(p1 < p2));
                ret_access[0] &= (!(p1 <= p2));
                ret_access[0] &= ((p1 > p2));
                ret_access[0] &= ((p1 >= p2));
            }

            {
                typedef s::pair<int, short> P;
                P p1(3, static_cast<short>(4));
                P p2(3, static_cast<short>(2));
                ret_access[0] &= (!(p1 == p2));
                ret_access[0] &= ((p1 != p2));
                ret_access[0] &= (!(p1 < p2));
                ret_access[0] &= (!(p1 <= p2));
                ret_access[0] &= ((p1 > p2));
                ret_access[0] &= ((p1 >= p2));
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
