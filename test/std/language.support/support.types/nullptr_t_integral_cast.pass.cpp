//===----------------------------------------------------------------------===//
//
// NOTE: nullptr_t emulation cannot handle a reinterpret_cast to an
// integral type
// XFAIL: c++98, c++03
//
// typedef decltype(nullptr) nullptr_t;
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

int
main(int, char**)
{
    const s::size_t N = 1;
    bool ret = true;
    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{N});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                s::ptrdiff_t i = reinterpret_cast<s::ptrdiff_t>(nullptr);
                acc[0] &= (i == 0);
            });
        });
    }

    if (ret)

        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;

    return 0;
}
