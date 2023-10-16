//===----------------------------------------------------------------------===//
//
// <array>
//
// template <class T, size_t N> constexpr size_type array<T,N>::size();
//
//===----------------------------------------------------------------------===//

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
    bool ret = true;
    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{1});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                {
                    typedef float T;
                    typedef s::array<T, 3> C;
                    C c = {1.f, 2.f, 3.5f};
                    ret_acc[0] &= (c.size() == 3);
                    ret_acc[0] &= (c.max_size() == 3);
                    ret_acc[0] &= (!c.empty());
                }
                {
                    typedef float T;
                    typedef s::array<T, 0> C;
                    C c = {};
                    ret_acc[0] &= (c.size() == 0);
                    ret_acc[0] &= (c.max_size() == 0);
                    ret_acc[0] &= (c.empty());
                }
                {
                    typedef int T;
                    typedef s::array<T, 3> C;
                    constexpr C c = {1, 2, 35};
                    static_assert(c.size() == 3, "");
                    static_assert(c.max_size() == 3, "");
                    static_assert(!c.empty(), "");
                }
                {
                    typedef int T;
                    typedef s::array<T, 0> C;
                    constexpr C c = {};
                    static_assert(c.size() == 0, "");
                    static_assert(c.max_size() == 0, "");
                    static_assert(c.empty(), "");
                }
            });
        });
    }

    if (ret)
        std::cout << "Pass" << std::endl;
    else
        std::cout << "Fail" << std::endl;
    return 0;
}
