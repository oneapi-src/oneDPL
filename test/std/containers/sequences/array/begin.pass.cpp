#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
namespace s = std;
#endif

int
main(int, char**)
{
    {
        auto ret = true;
        {
            cl::sycl::queue myQueue;
            cl::sycl::buffer<bool, 1> buf1(&ret, cl::sycl::range<1>(1));

            myQueue.submit([&](cl::sycl::handler& cgh) {
                auto ret_access = buf1.get_access<cl::sycl::access::mode::read_write>(cgh);

                cgh.single_task<class KernelBeginTest>([=]() {
                    typedef int T;
                    typedef s::array<T, 3> C;
                    C c = {1, 2, 35};
                    C::iterator i;
                    i = c.begin();
                    ret_access[0] &= (*i == 1);
                    ret_access[0] &= (&*i == c.data());
                    *i = 55;
                    ret_access[0] &= (c[0] == 55);
                });
            });
        }

        if (ret)
        {
            std::cout << "Pass" << std::endl;
        }
        else
        {
            std::cout << "Fail" << std::endl;
        }
    }
    return 0;
}
