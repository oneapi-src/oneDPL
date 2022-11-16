#include "oneapi_std_test_config.h"
#include "test_macros.h"

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

struct X
{
    explicit X(int, int) {}

  private:
    X(const X&) = delete;
};

struct move_only
{
    move_only() {}
    move_only(move_only&&) {}

  private:
    move_only(const move_only&) = delete;
};

cl::sycl::cl_bool
kernel_test()
{
    int* ip = 0;
    int X::*mp = 0;

    s::pair<int*, int*> p1(0, 0);
    s::pair<int*, int*> p2(ip, 0);
    s::pair<int*, int*> p3(0, ip);
    s::pair<int*, int*> p4(ip, ip);

    s::pair<int X::*, int*> p5(0, 0);
    s::pair<int X::*, int X::*> p6(mp, 0);
    s::pair<int X::*, int X::*> p7(0, mp);
    s::pair<int X::*, int X::*> p8(mp, mp);
    s::pair<int*, move_only> p9(ip, move_only());
    s::pair<int X::*, move_only> p10(mp, move_only());
    s::pair<move_only, int*> p11(move_only(), ip);
    s::pair<move_only, int X::*> p12(move_only(), mp);
    s::pair<move_only, move_only> p13{move_only(), move_only()};
    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }
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
