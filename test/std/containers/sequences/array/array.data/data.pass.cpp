//===----------------------------------------------------------------------===//
//
// <array>
//
// T *data();
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

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

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

#if TEST_DPCPP_BACKEND_PRESENT
class Test1;

struct NoDefault
{
    NoDefault() {}
    NoDefault(int) {}
};
#endif // TEST_DPCPP_BACKEND_PRESENT

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
            cgh.single_task<Test1>([=]() {
                {
                    typedef float T;
                    typedef s::array<T, 3> C;
                    C c = {1.f, 2.f, 3.5f};
                    T* p = c.data();
                    ret_acc[0] &= (p[0] == 1.f);
                    ret_acc[0] &= (p[1] == 2.f);
                    ret_acc[0] &= (p[2] == 3.5f);
                }
                {
                    typedef float T;
                    typedef s::array<T, 0> C;
                    C c = {};
                    T* p = c.data();
                }
                {
                    typedef float T;
                    typedef s::array<const T, 0> C;
                    C c = {{}};
                    const T* p = c.data();
                    static_assert((s::is_same<decltype(c.data()), const T*>::value), "");
                }
                {
                    typedef NoDefault T;
                    typedef s::array<T, 0> C;
                    C c = {};
                    T* p = c.data();
                }
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
