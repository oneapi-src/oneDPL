#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <cstddef>
namespace s = std;
#endif

class Test1;

int
main(int, char**)
{
    bool ret = false;
    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{1});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<Test1>([=]() {
#ifndef NULL
#    error NULL not defined
#else
                acc[0] = true;
#endif
            });
        });
    }

    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
