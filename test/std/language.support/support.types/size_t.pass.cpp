//===----------------------------------------------------------------------===//
//
// size_t should:
//
//  1. be in namespace std.
//  2. be the same sizeof as void*.
//  3. be an unsigned integral.
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <cstddef>
#    include <type_traits>
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
                static_assert(sizeof(s::size_t) == sizeof(void*), "sizeof(s::size_t) == sizeof(void*)");
                static_assert(s::is_unsigned<s::size_t>::value, "s::is_unsigned<s::size_t>::value");
                static_assert(s::is_integral<s::size_t>::value, "s::is_integral<s::size_t>::value");
                acc[0] &= (sizeof(s::size_t) == sizeof(void*));
            });
        });
    }

    if (ret)

        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;

    return 0;
}
