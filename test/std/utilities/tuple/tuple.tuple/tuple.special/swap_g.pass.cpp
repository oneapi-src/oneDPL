#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
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

struct MoveOnly
{
    explicit MoveOnly(int j) : i(j) {}

    MoveOnly(MoveOnly&& m) : i(m.i) {}

    MoveOnly&
    operator=(MoveOnly&& m)
    {
        i = m.i;
        return *this;
    }

    MoveOnly(MoveOnly const&) = delete;
    MoveOnly&
    operator=(MoveOnly const&) = delete;

    bool
    operator==(MoveOnly const& m)
    {
        return i == m.i;
    }

    void
    swap(MoveOnly& m)
    {
        s::swap(m.i, i);
    }

    int i;
};

void
swap(MoveOnly& m1, MoveOnly& m2)
{
    m1.swap(m2);
}

MoveOnly
make_move_only(int i)
{
    return MoveOnly(i);
}

cl::sycl::cl_bool
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> numOfItem{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                {
                    s::tuple<> t1, t2;
                    s::swap(t1, t2);

                    ret_acc[0] &= (t1 == t2);
                }
                {
                    s::tuple<int> t1(1), t2(2);
                    s::swap(t1, t2);

                    ret_acc[0] &= (s::get<0>(t1) == 2 && s::get<0>(t2) == 1);
                }
                {
                    s::tuple<int, float> t1(1, 1.0f), t2(2, 2.0f);
                    s::swap(t1, t2);

                    ret_acc[0] &= (s::get<0>(t1) == 2 && s::get<0>(t2) == 1);
                    ret_acc[0] &= (s::get<1>(t1) == 2.0f && s::get<1>(t2) == 1.0f);
                }
                {
                    s::tuple<int, float, MoveOnly> t1(1, 1.0f, make_move_only(1)), t2(2, 2.0f, make_move_only(2));

                    s::swap(t1, t2);

                    ret_acc[0] &= (s::get<0>(t1) == 2 && s::get<0>(t2) == 1);
                    ret_acc[0] &= (s::get<1>(t1) == 2.0f && s::get<1>(t2) == 1.0f);
                    ret_acc[0] &= (s::get<2>(t1) == make_move_only(2) && s::get<2>(t2) == make_move_only(1));
                }
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    if (ret)
        std::cout << "pass" << std::endl;
    else
        std::cout << "pass" << std::endl;
    return 0;
}
