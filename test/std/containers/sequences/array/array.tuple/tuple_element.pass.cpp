//===----------------------------------------------------------------------===//
//
// <array>
//
// tuple_element<I, array<T, N> >::type
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
class KernelTest1;

template <class T>
void
test()
{
    {
        typedef T Exp;
        typedef s::array<T, 3> C;
        static_assert((s::is_same<typename s::tuple_element<0, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<2, C>::type, Exp>::value), "");
    }
    {
        typedef T const Exp;
        typedef s::array<T, 3> const C;
        static_assert((s::is_same<typename s::tuple_element<0, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<2, C>::type, Exp>::value), "");
    }
    {
        typedef T volatile Exp;
        typedef s::array<T, 3> volatile C;
        static_assert((s::is_same<typename s::tuple_element<0, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<2, C>::type, Exp>::value), "");
    }
    {
        typedef T const volatile Exp;
        typedef s::array<T, 3> const volatile C;
        static_assert((s::is_same<typename s::tuple_element<0, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<1, C>::type, Exp>::value), "");
        static_assert((s::is_same<typename s::tuple_element<2, C>::type, Exp>::value), "");
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    bool ret = false;
    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{1});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest1>([=]() {
                test<float>();
                test<int>();
                ret_acc[0] = true;
            });
        });
    }

    if (ret)
    {
        std::cout << "Pass" << std::endl;
    }
    else
        std::cout << "Fail" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
