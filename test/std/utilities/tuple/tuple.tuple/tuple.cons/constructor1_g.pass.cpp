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
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using namespace std;
sycl::cl_bool
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int x1 = 0, x2 = 0;
                const int& z1 = x1;

                // Test empty constructor
                s::tuple<> ta __attribute__((unused));
                s::tuple<int, int> tb;
                // Test construction from values
                s::tuple<int, int> tc(x1, x2);
                s::tuple<int, int&> td(x1, x2);
                s::tuple<const int&> te(z1);
                x1 = 1;
                x2 = 1;
                ret_access[0] = (get<0>(td) == 0 && get<1>(td) == 1 && get<0>(te) == 1);

                // Test identical s::tuple copy constructor
                s::tuple<int, int> tf(tc);
                s::tuple<int, int> tg(td);
                s::tuple<const int&> th(te);
                // Test different s::tuple copy constructor
                s::tuple<int, float> ti(tc);
                s::tuple<int, float> tj(td);
                // s::tuple<int&, int&> tk(tc);
                s::tuple<const int&, const int&> tl(tc);
                s::tuple<const int&, const int&> tm(tl);
                // Test constructing from a pair
                pair<int, int> pair1(1, 1);
                const pair<int, int> pair2(pair1);
                s::tuple<int, int> tn(pair1);
                s::tuple<int, const int&> to(pair1);
                s::tuple<int, int> tp(pair2);
                s::tuple<int, const int&> tq(pair2);
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test1();
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
