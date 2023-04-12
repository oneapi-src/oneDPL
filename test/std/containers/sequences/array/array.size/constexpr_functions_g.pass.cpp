#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct constexpr_member_functions
{
    template <typename _Ttesttype>
    void
    operator()()
    {
        struct _Concept
        {
            void
            __constraint()
            {
                constexpr _Ttesttype a = {};
                constexpr auto v1 __attribute__((unused)) = a.size();
                constexpr auto v2 __attribute__((unused)) = a.max_size();
                constexpr auto v3 __attribute__((unused)) = a.empty();
            }
        };

        _Concept c;
        c.__constraint();
    }
};

sycl::cl_bool
kernel_test()
{
    constexpr_member_functions test;
    test.operator()<s::array<long, 60>>();
    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::cl_bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }
    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
