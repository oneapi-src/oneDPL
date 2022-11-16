#include "oneapi_std_test_config.h"
#include "testsuite_struct.h"

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

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef s::pair<int, int> tt1;
                typedef s::pair<int, float> tt2;
                typedef s::pair<short, NoexceptMoveAssignClass> tt3;
                typedef s::pair<short, NoexceptMoveConsClass> tt4;
                typedef s::pair<NoexceptMoveConsClass, NoexceptMoveConsClass> tt5;
                typedef s::pair<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass> tt6;
                typedef s::pair<NoexceptMoveConsNoexceptMoveAssignClass, float> tt7;
                typedef s::pair<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass> tt8;

                static_assert(noexcept(s::declval<tt1&>().swap(s::declval<tt1&>())), "Error");
                static_assert(noexcept(s::declval<tt2&>().swap(s::declval<tt2&>())), "Error");
                static_assert(noexcept(s::declval<tt3&>().swap(s::declval<tt3&>())), "Error");
                static_assert(noexcept(s::declval<tt4&>().swap(s::declval<tt4&>())), "Error");
                static_assert(noexcept(s::declval<tt5&>().swap(s::declval<tt5&>())), "Error");
                static_assert(noexcept(s::declval<tt6&>().swap(s::declval<tt6&>())), "Error");
                static_assert(noexcept(s::declval<tt7&>().swap(s::declval<tt7&>())), "Error");
                static_assert(noexcept(s::declval<tt8&>().swap(s::declval<tt8&>())), "Error");
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
