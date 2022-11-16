#include "oneapi_std_test_config.h"
#include "testsuite_struct.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef s::tuple<int> tt1;
                typedef s::tuple<int, float> tt2;
                typedef s::tuple<short, float, int> tt3;
                typedef s::tuple<short, NoexceptMoveConsClass, float> tt4;
                typedef s::tuple<NoexceptMoveConsClass, NoexceptMoveConsClass, float> tt5;
                typedef s::tuple<NoexceptMoveConsClass, NoexceptMoveConsClass, NoexceptMoveConsClass> tt6;

                static_assert(s::is_nothrow_move_constructible<tt1>::value, "Error");
                static_assert(s::is_nothrow_move_constructible<tt2>::value, "Error");
                static_assert(s::is_nothrow_move_constructible<tt3>::value, "Error");
                static_assert(s::is_nothrow_move_constructible<tt4>::value, "Error");
                static_assert(s::is_nothrow_move_constructible<tt5>::value, "Error");
                static_assert(s::is_nothrow_move_constructible<tt6>::value, "Error");
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    std::cout << "pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
