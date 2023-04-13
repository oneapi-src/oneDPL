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
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

void
kernel_test1(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            typedef s::tuple<int> tt1;
            typedef s::tuple<int, float> tt2;
            typedef s::tuple<short, float, int> tt3;
            typedef s::tuple<short, NoexceptMoveAssignClass, float> tt4;
            typedef s::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, float> tt5;
            typedef s::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, NoexceptMoveAssignClass> tt6;

            static_assert(s::is_nothrow_move_assignable<tt1>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt2>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt3>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt4>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt5>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt6>::value, "Error");
        });
    });
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            typedef s::tuple<int> tt1;
            typedef s::tuple<int, double> tt2;
            typedef s::tuple<short, double, int> tt3;
            typedef s::tuple<short, NoexceptMoveAssignClass, double> tt4;
            typedef s::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, double> tt5;
            typedef s::tuple<NoexceptMoveAssignClass, NoexceptMoveAssignClass, NoexceptMoveAssignClass> tt6;

            static_assert(s::is_nothrow_move_assignable<tt1>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt2>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt3>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt4>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt5>::value, "Error");
            static_assert(s::is_nothrow_move_assignable<tt6>::value, "Error");
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
