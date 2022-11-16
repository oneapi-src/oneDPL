#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename T>
typename s::decay<T>::type
copy(T&& x)
{
    return s::forward<T>(x);
}

template <typename... Args1, typename... Args2>
bool
check_tuple_cat(s::tuple<Args1...> t1, s::tuple<Args2...> t2)
{
    typedef s::tuple<Args1..., Args2...> concatenated;

    auto cat1 = s::tuple_cat(t1, t2);
    auto cat2 = s::tuple_cat(copy(t1), t2);
    auto cat3 = s::tuple_cat(t1, copy(t2));
    auto cat4 = s::tuple_cat(copy(t1), copy(t2));

    static_assert(s::is_same<decltype(cat1), concatenated>::value, "");
    static_assert(s::is_same<decltype(cat2), concatenated>::value, "");
    static_assert(s::is_same<decltype(cat3), concatenated>::value, "");
    static_assert(s::is_same<decltype(cat4), concatenated>::value, "");

    auto ret = (cat1 == cat2);
    ret &= (cat1 == cat3);
    ret &= (cat1 == cat4);
    return ret;
}

cl::sycl::cl_bool
kernel_test()
{
    int i = 0;
    s::tuple<> t0;
    s::tuple<int&> t1(i);
    s::tuple<int&, int> t2(i, 0);
    s::tuple<int const&, int, float> t3(i, 0, 0.f);

    auto ret = check_tuple_cat(t0, t0);
    ret &= check_tuple_cat(t0, t1);
    ret &= check_tuple_cat(t0, t2);
    ret &= check_tuple_cat(t0, t3);

    ret &= check_tuple_cat(t1, t0);
    ret &= check_tuple_cat(t1, t1);
    ret &= check_tuple_cat(t1, t2);
    ret &= check_tuple_cat(t1, t3);

    ret &= check_tuple_cat(t2, t0);
    ret &= check_tuple_cat(t2, t1);
    ret &= check_tuple_cat(t2, t2);
    ret &= check_tuple_cat(t2, t3);

    ret &= check_tuple_cat(t3, t0);
    ret &= check_tuple_cat(t3, t1);
    ret &= check_tuple_cat(t3, t2);
    ret &= check_tuple_cat(t3, t3);
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
