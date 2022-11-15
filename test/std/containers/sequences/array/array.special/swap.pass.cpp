//===----------------------------------------------------------------------===//
//
// <array>
//
// template <class T, size_t N> void swap(array<T,N>& x, array<T,N>& y);
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"

#include <iostream>

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
#if TEST_DPCPP_BACKEND_PRESENT
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
                    C c1 = {1.f, 2.f, 3.5f};
                    C c2 = {4.f, 5.f, 6.5f};
                    swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 3);
                    ret_acc[0] &= (c1[0] == 4.f);
                    ret_acc[0] &= (c1[1] == 5.f);
                    ret_acc[0] &= (c1[2] == 6.5f);
                    ret_acc[0] &= (c2.size() == 3);
                    ret_acc[0] &= (c2[0] == 1.f);
                    ret_acc[0] &= (c2[1] == 2.f);
                    ret_acc[0] &= (c2[2] == 3.5f);
                }
                {
                    typedef float T;
                    typedef s::array<T, 0> C;
                    C c1 = {};
                    C c2 = {};
                    swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 0);
                    ret_acc[0] &= (c2.size() == 0);
                }
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
