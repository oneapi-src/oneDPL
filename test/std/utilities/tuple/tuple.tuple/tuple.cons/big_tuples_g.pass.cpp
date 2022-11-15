// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

// A simple class without conversions to check some things
struct foo
{
};

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItem{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                using s::ignore;
                //test construction
                typedef s::tuple<int, int, int, int, int, int, int, int, int, int> type1;
                type1 a(0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
                type1 b(0, 0, 0, 0, 0, 0, 0, 0, 0, 2);
                type1 c(a);
                typedef s::tuple<int, int, int, int, int, int, int, int, int, char> type2;
                type2 d(0, 0, 0, 0, 0, 0, 0, 0, 0, 3);
                type1 e(d);
                typedef s::tuple<foo, int, int, int, int, int, int, int, int, foo> type3;
                // get
                ret_access[0] &= (s::get<9>(a) == 1 && s::get<9>(b) == 2);
                // comparisons
                ret_access[0] &= (a == a && !(a != a) && a <= a && a >= a && !(a < a) && !(a > a));
                ret_access[0] &= (!(a == b) && a != b && a <= b && a < b && !(a >= b) && !(a > b));
                //tie
                {
                    int i = 0;
                    tie(ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, i) = a;
                    ret_access[0] &= (i == 1);
                }
                //test_assignment
                a = d;
                a = b;
                //make_tuple
                s::make_tuple(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

                //s::tuple_size
                ret_access[0] &= (s::tuple_size<type3>::value == 10);
                //s::tuple_element
                {
                    foo q1;
                    s::tuple_element<0, type3>::type q2(q1);
                    s::tuple_element<9, type3>::type q3(q1);
                }
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(void)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
