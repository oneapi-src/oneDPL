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

#if TEST_DPCPP_BACKEND_PRESENT
struct A
{
    int x;
};
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    {
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() { static_assert(noexcept(offsetof(A, x)), ""); });
        });
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
