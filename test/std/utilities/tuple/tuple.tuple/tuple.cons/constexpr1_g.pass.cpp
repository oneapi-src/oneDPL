#include "oneapi_std_test_config.h"
#include "testsuite_common_types.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(tuple)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <tuple>
#    include <utility>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef s::tuple<int, int> tuple_type;

                // 01: default ctor
                constexpr_default_constructible test1;
                test1.operator()<tuple_type>();

                // 02: default copy ctor
                constexpr_single_value_constructible test2;
                test2.operator()<tuple_type, tuple_type>();

                // 03: element move ctor, single element
                const int i1(415);
                constexpr tuple_type t2{44, s::move(i1)};

                // 04: element move ctor, two element
                const int i2(510);
                const int i3(408);
                constexpr tuple_type t4{s::move(i2), s::move(i3)};

                // 05: value-type conversion constructor
                const int i4(650);
                const int i5(310);
                constexpr tuple_type t8(i4, i5);

                // 06: pair conversion ctor
                test2.operator()<tuple_type, s::pair<int, int>>();
                test2.operator()<s::tuple<short, short>, s::pair<int, int>>();
                test2.operator()<tuple_type, s::pair<short, short>>();

                // 07: different-tuple-type conversion constructor
                test2.operator()<tuple_type, s::tuple<short, short>>();
                test2.operator()<s::tuple<short, short>, tuple_type>();
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
