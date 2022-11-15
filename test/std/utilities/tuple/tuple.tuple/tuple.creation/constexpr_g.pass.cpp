#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

// make_tuple
void
test_make_tuple()
{
    {
        typedef s::tuple<int, float> tuple_type;
        constexpr tuple_type p1 __attribute__((unused)) = s::make_tuple(22, 22.222f);
    }

    {
        typedef s::tuple<int, float, int> tuple_type;
        constexpr tuple_type p1 __attribute__((unused)) = s::make_tuple(22, 22.222f, 77799);
    }
}

// forward_as_tuple
void
test_forward_as_tuple()
{
    {
        static const int i(22);
        static const float f(22.222f);
        typedef s::tuple<const int&, const float&&> tuple_type;
        constexpr tuple_type p1 __attribute__((unused)) = s::forward_as_tuple(i, s::move(f));
    }

    {
        static const int i(22);
        static const float f(22.222f);
        static const int ii(77799);

        typedef s::tuple<const int&, const float&, const int&&> tuple_type;
        constexpr tuple_type p1 __attribute__((unused)) = s::forward_as_tuple(i, f, s::move(ii));
    }
}

// tie
void
test_tie()
{
    {
        static const int i(22);
        static const float f(22.222f);
        typedef s::tuple<const int&, const float&> tuple_type;
        constexpr tuple_type p1 __attribute__((unused)) = s::tie(i, f);
    }

    {
        static const int i(22);
        static const float f(22.222f);
        static const int ii(77799);

        typedef s::tuple<const int&, const float&, const int&> tuple_type;
        constexpr tuple_type p1 __attribute__((unused)) = s::tie(i, f, ii);
    }
}

// get
void
test_get()
{
    {
        typedef s::tuple<int, float> tuple_type;
        constexpr tuple_type t1{55, 77.77f};
        constexpr auto var __attribute__((unused)) = s::get<1>(t1);
    }

    {
        typedef s::tuple<int, float, int> tuple_type;
        constexpr tuple_type t1{55, 77.77f, 99};
        constexpr auto var __attribute__((unused)) = s::get<2>(t1);
    }
}

// tuple_cat
void
test_tuple_cat()
{
    typedef s::tuple<int, float> tuple_type1;
    typedef s::tuple<int, int, float> tuple_type2;

    constexpr tuple_type1 t1{55, 77.77f};
    constexpr tuple_type2 t2{55, 99, 77.77f};
    constexpr auto cat1 __attribute__((unused)) = s::tuple_cat(t1, t2);
}

namespace
{

template <class T>
constexpr int zero_from_anything(T)
{
    return 0;
}

} // namespace

// ignore
void
test_ignore()
{
    constexpr auto ign1 __attribute__((unused)) = s::ignore;
    constexpr auto ign2 __attribute__((unused)) = s::make_tuple(s::ignore);
    constexpr int ign3 __attribute__((unused)) = zero_from_anything(s::ignore);
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test_make_tuple();
                test_forward_as_tuple();
                test_tie();
                test_get();
                test_tuple_cat();
                test_ignore();
            });
        });
    }
}

int
main()
{
    kernel_test();
    std::cout << "pass" << std::endl;
    return 0;
}
