//===----------------------------------------------------------------------===//
//
// template<class E> class initializer_list;
//
// initializer_list();
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <initializer_list>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
namespace s = oneapi_cpp_ns;
#else
#    include <cstddef>
namespace s = std;
#endif

struct A
{
};

int
main(int, char**)
{
    const std::size_t N = 1;
    bool rs[N] = {false};

    {
        cl::sycl::buffer<bool, 1> buf(rs, cl::sycl::range<1>{N});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                std::initializer_list<A> il;
                acc[0] = (il.size() == 0);
            });
        });
    }

    for (std::size_t i = 0; i < N; ++i)
    {
        if (!rs[i])
        {
            std::cout << "Fail" << std::endl;
            return -1;
        }
    }

    std::cout << "Pass" << std::endl;

    return 0;
}
