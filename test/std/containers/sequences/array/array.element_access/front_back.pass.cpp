// <array>

// reference front();       // constexpr in C++17
// reference back();        // constexpr in C++17
// const_reference front(); // constexpr in C++14
// const_reference back();  // constexpr in C++14

#include "oneapi_std_test_config.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <type_traits>
namespace s = std;
#endif

bool
kernel_test()
{
    cl::sycl::queue myQueue;
    auto ret = true;
    {
        cl::sycl::buffer<bool, 1> buf1(&ret, cl::sycl::range<1>(1));
        myQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buf1.get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.single_task<class KernelFrontBackTest>([=]() {
                typedef int T;
                typedef s::array<T, 3> C;
                {
                    C c = {1, 2, 35};

                    C::reference r1 = c.front();
                    ret_access[0] &= (r1 == 1);
                    r1 = 55;
                    ret_access[0] &= (c[0] == 55);

                    C::reference r2 = c.back();
                    ret_access[0] &= (r2 == 35);
                    r2 = 75;
                    ret_access[0] &= (c[2] == 75);
                }
                {
                    const C c = {1, 2, 35};
                    C::const_reference r1 = c.front();
                    ret_access[0] &= (r1 == 1);

                    C::const_reference r2 = c.back();
                    ret_access[0] &= (r2 == 35);
                }
                {

                    C c = {};
                    C const& cc = c;
                    ret_access[0] &= (s::is_same<decltype(c.back()), typename C::reference>::value == true);
                    ret_access[0] &= (s::is_same<decltype(cc.back()), typename C::const_reference>::value == true);
                    (void)noexcept(c.back());
                    (void)noexcept(cc.back());
                    ret_access[0] &= (s::is_same<decltype(c.front()), typename C::reference>::value == true);
                    ret_access[0] &= (s::is_same<decltype(cc.front()), typename C::const_reference>::value == true);
                    (void)noexcept(c.back());
                    (void)noexcept(cc.back());
                }
                {
                    constexpr C c = {1, 2, 35};
                    constexpr T t1 = c.front();
                    ret_access[0] &= (t1 == 1);
                    constexpr T t2 = c.back();
                    ret_access[0] &= (t2 == 35);
                }
            });
        });
    }
    return ret;
}

int
main(int, char**)
{
    auto ret = kernel_test();
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
    return 0;
}
