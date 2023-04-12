#include "oneapi_std_test_config.h"
#include "test_macros.h"

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

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct MoveOnly
{
    MoveOnly() {}

    MoveOnly(MoveOnly&&) {}

    MoveOnly&
    operator=(MoveOnly&&)
    {
        return *this;
    }

    MoveOnly(MoveOnly const&) = delete;
    MoveOnly&
    operator=(MoveOnly const&) = delete;
};

MoveOnly
make_move_only()
{
    return MoveOnly();
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                typedef s::tuple<MoveOnly> move_only_tuple;

                move_only_tuple t1(make_move_only());
                move_only_tuple t2(s::move(t1));
                move_only_tuple t3 = s::move(t2);
                t1 = s::move(t3);

                typedef s::tuple<MoveOnly, MoveOnly> move_only_tuple2;

                move_only_tuple2 t4(make_move_only(), make_move_only());
                move_only_tuple2 t5(s::move(t4));
                move_only_tuple2 t6 = s::move(t5);
                t4 = s::move(t6);
            });
        });
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
    std::cout << "pass" << std::endl;
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
