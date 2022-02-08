// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(array)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

void
kernel_test1(cl::sycl::queue& deviceQueue)
{
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() {
                static_assert(s::is_same<decltype(s::tuple_cat()), s::tuple<>>::value, "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<>>())), s::tuple<>>::value, "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<>&>())), s::tuple<>>::value,
                              "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<const s::tuple<>>())), s::tuple<>>::value,
                              "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<const s::tuple<>&>())), s::tuple<>>::value,
                              "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::pair<int, bool>>())), s::tuple<int, bool>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::pair<int, bool>&>())), s::tuple<int, bool>>::value,
                    "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<const s::pair<int, bool>>())),
                                         s::tuple<int, bool>>::value,
                              "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<const s::pair<int, bool>&>())),
                                         s::tuple<int, bool>>::value,
                              "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::array<int, 3>>())), s::tuple<int, int, int>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::array<int, 3>&>())), s::tuple<int, int, int>>::value,
                    "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<const s::array<int, 3>>())),
                                         s::tuple<int, int, int>>::value,
                              "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<const s::array<int, 3>&>())),
                                         s::tuple<int, int, int>>::value,
                              "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<>>(), s::declval<s::tuple<>>())),
                                         s::tuple<>>::value,
                              "Error");
                static_assert(s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<>>(), s::declval<s::tuple<>>(),
                                                               s::declval<s::tuple<>>())),
                                         s::tuple<>>::value,
                              "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<>>(), s::declval<s::array<char, 0>>(),
                                                     s::declval<s::array<int, 0>>(), s::declval<s::tuple<>>())),
                               s::tuple<>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<int>>(), s::declval<s::tuple<float>>())),
                               s::tuple<int, float>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<int>>(), s::declval<s::tuple<float>>(),
                                                     s::declval<s::tuple<const long&>>())),
                               s::tuple<int, float, const long&>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::array<wchar_t, 3>&>(), s::declval<s::tuple<float>>(),
                                                     s::declval<s::tuple<>>(), s::declval<s::tuple<unsigned&>>(),
                                                     s::declval<s::pair<bool, std::nullptr_t>>())),
                               s::tuple<wchar_t, wchar_t, wchar_t, float, unsigned&, bool, std::nullptr_t>>::value,
                    "Error");

                s::array<int, 3> a3;
                s::pair<float, bool> pdb;
                s::tuple<unsigned, float, std::nullptr_t, void*> t;
                int i{};
                float d{};
                int* pi{};
                s::tuple<int&, float&, int*&> to{i, d, pi};

                static_assert(s::is_same<decltype(s::tuple_cat(a3, pdb, t, a3, pdb, t)),
                                         s::tuple<int, int, int, float, bool, unsigned, float, std::nullptr_t, void*, int,
                                                  int, int, float, bool, unsigned, float, std::nullptr_t, void*>>::value,
                              "Error");

                s::tuple_cat(s::tuple<int, char, void*>{}, to, a3, s::tuple<>{}, s::pair<float, std::nullptr_t>{}, pdb,
                             to);
            });
        });
    }
}

void
kernel_test2(cl::sycl::queue& deviceQueue)
{
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest2>([=]() {
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<int>>(), s::declval<s::tuple<double>>())),
                               s::tuple<int, double>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::tuple<int>>(), s::declval<s::tuple<double>>(),
                                                     s::declval<s::tuple<const long&>>())),
                               s::tuple<int, double, const long&>>::value,
                    "Error");
                static_assert(
                    s::is_same<decltype(s::tuple_cat(s::declval<s::array<wchar_t, 3>&>(),
                                                     s::declval<s::tuple<double>>(), s::declval<s::tuple<>>(),
                                                     s::declval<s::tuple<unsigned&>>(),
                                                     s::declval<s::pair<bool, std::nullptr_t>>())),
                               s::tuple<wchar_t, wchar_t, wchar_t, double, unsigned&, bool, std::nullptr_t>>::value,
                    "Error");

                s::array<int, 3> a3;
                s::pair<double, bool> pdb;
                s::tuple<unsigned, float, std::nullptr_t, void*> t;
                static_assert(
                    s::is_same<decltype(s::tuple_cat(a3, pdb, t, a3, pdb, t)),
                               s::tuple<int, int, int, double, bool, unsigned, float, std::nullptr_t, void*, int, int,
                                        int, double, bool, unsigned, float, std::nullptr_t, void*>>::value,
                    "Error");
            });
        });
    }
}

int
main()
{
    cl::sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    if (deviceQueue.get_device().has_extension("cl_khr_fp64"))
    {
        kernel_test2(deviceQueue);
    }
    std::cout << "pass" << std::endl;
    return 0;
}
