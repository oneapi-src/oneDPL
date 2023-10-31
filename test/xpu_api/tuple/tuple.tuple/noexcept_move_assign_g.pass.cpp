#include "oneapi_std_test_config.h"
#include "testsuite_struct.h"
#include <CL/sycl.hpp>
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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
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
kernel_test2(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
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

int
main()
{
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    std::cout << "pass" << std::endl;
    return 0;
}
