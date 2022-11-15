// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest1>([=]() {
            {
                static_assert(
                    s::is_same<s::tuple_element<0, const s::tuple<float, void, int>>::type, const float>::value,
                    "Error");
                static_assert(
                    s::is_same<s::tuple_element<1, volatile s::tuple<short, void>>::type, volatile void>::value,
                    "Error");
                static_assert(s::is_same<s::tuple_element<2, const volatile s::tuple<float, char, int>>::type,
                                         const volatile int>::value,
                              "Error");
            }
        });
    });
}
void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task<class KernelTest2>([=]() {
            {
                static_assert(
                    s::is_same<s::tuple_element<0, const s::tuple<double, void, int>>::type, const double>::value,
                    "Error");
            }
        });
    });
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    std::cout << "pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
