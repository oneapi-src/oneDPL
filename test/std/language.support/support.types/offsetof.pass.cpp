//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03
//
//===----------------------------------------------------------------------===//

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

#ifndef offsetof
#    error offsetof not defined
#endif

struct A
{
    int x;
};

int
main(int, char**)
{
    {
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() { static_assert(noexcept(offsetof(A, x)), ""); });
        });
    }
    std::cout << "Pass" << std::endl;
    return 0;
}
