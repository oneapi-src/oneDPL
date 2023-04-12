// Tuple
#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct foo
{
};
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                foo q1;
                s::tuple_element<0, s::tuple<foo, void, int>>::type q2(q1);
                s::tuple_element<2, s::tuple<void, int, foo>>::type q3(q1);
                s::tuple_element<0, const s::tuple<foo, void, int>>::type q4(q1);
                s::tuple_element<2, const s::tuple<void, int, foo>>::type q5(q1);
                s::tuple_element<0, volatile s::tuple<foo, void, int>>::type q6(q1);
                s::tuple_element<2, volatile s::tuple<void, int, foo>>::type q7(q1);
                s::tuple_element<0, const volatile s::tuple<foo, void, int>>::type q8(q1);
                s::tuple_element<2, const volatile s::tuple<void, int, foo>>::type q9(q1);
                ret_access[0] = true;
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
