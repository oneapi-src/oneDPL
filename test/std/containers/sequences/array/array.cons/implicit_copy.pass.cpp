//===----------------------------------------------------------------------===//
//
// <array>
//
// implicitly generated array constructors / assignment operators
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"

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

#if TEST_DPCPP_BACKEND_PRESENT
#define TEST_NOT_COPY_ASSIGNABLE(T) static_assert(!s::is_copy_assignable<T>::value, "")

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
    {
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() {
                {
                    typedef float T;
                    typedef s::array<T, 3> C;
                    C c = {1.1f, 2.2f, 3.3f};
                    C c2 = c;
                    c2 = c;
                    static_assert(s::is_copy_constructible<C>::value, "");
                    static_assert(s::is_copy_assignable<C>::value, "");
                }
                {
                    typedef float T;
                    typedef s::array<const T, 3> C;
                    C c = {1.1f, 2.2f, 3.3f};
                    C c2 = c;
                    ((void)c2);
                    static_assert(s::is_copy_constructible<C>::value, "");
                    TEST_NOT_COPY_ASSIGNABLE(C);
                }
                {
                    typedef float T;
                    typedef s::array<T, 0> C;
                    C c = {};
                    C c2 = c;
                    c2 = c;
                    static_assert(s::is_copy_constructible<C>::value, "");
                    static_assert(s::is_copy_assignable<C>::value, "");
                }
                {
                    // const arrays of size 0 should disable the implicit copy assignment
                    // operator.
                    typedef float T;
                    typedef s::array<const T, 0> C;
                    const C c = {{}};
                    C c2 = c;
                    ((void)c2);
                    static_assert(s::is_copy_constructible<C>::value, "");
                }
                {
                    typedef NoDefault T;
                    typedef s::array<T, 0> C;
                    C c = {};
                    C c2 = c;
                    c2 = c;
                    static_assert(s::is_copy_constructible<C>::value, "");
                    static_assert(s::is_copy_assignable<C>::value, "");
                }
                {
                    typedef NoDefault T;
                    typedef s::array<const T, 0> C;
                    C c = {{}};
                    C c2 = c;
                    ((void)c2);
                    static_assert(s::is_copy_constructible<C>::value, "");
                }
            });
        });
        TestUtils::exitOnError(true);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
