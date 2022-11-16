#include "oneapi_std_test_config.h"
#include "testsuite_struct.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
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
                typedef s::tuple<short, NoexceptMoveAssignClass, float> tt4;
                typedef s::tuple<short, NoexceptMoveConsClass, float> tt5;
                typedef s::tuple<NoexceptMoveConsClass> tt6;
                typedef s::tuple<NoexceptMoveConsNoexceptMoveAssignClass> tt7;
                typedef s::tuple<NoexceptMoveConsNoexceptMoveAssignClass, float> tt8;
                typedef s::tuple<float, NoexceptMoveConsNoexceptMoveAssignClass, short> tt9;
                typedef s::tuple<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass, char>
                    tt10;
                typedef s::tuple<NoexceptMoveConsNoexceptMoveAssignClass, NoexceptMoveConsNoexceptMoveAssignClass,
                                 NoexceptMoveConsNoexceptMoveAssignClass>
                    tt11;

                static_assert(noexcept(s::declval<tt1&>().swap(s::declval<tt1&>())), "Error");
                static_assert(noexcept(s::declval<tt2&>().swap(s::declval<tt2&>())), "Error");
                static_assert(noexcept(s::declval<tt3&>().swap(s::declval<tt3&>())), "Error");
                static_assert(noexcept(s::declval<tt4&>().swap(s::declval<tt4&>())), "Error");
                static_assert(noexcept(s::declval<tt5&>().swap(s::declval<tt5&>())), "Error");
                static_assert(noexcept(s::declval<tt6&>().swap(s::declval<tt6&>())), "Error");
                static_assert(noexcept(s::declval<tt7&>().swap(s::declval<tt7&>())), "Error");
                static_assert(noexcept(s::declval<tt8&>().swap(s::declval<tt8&>())), "Error");
                static_assert(noexcept(s::declval<tt9&>().swap(s::declval<tt9&>())), "Error");
                static_assert(noexcept(s::declval<tt10&>().swap(s::declval<tt10&>())), "Error");
                static_assert(noexcept(s::declval<tt11&>().swap(s::declval<tt11&>())), "Error");
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
