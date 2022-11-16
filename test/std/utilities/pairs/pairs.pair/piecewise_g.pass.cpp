// Tuple

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
#    include _ONEAPI_STD_TEST_HEADER(tuple)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
#    include <tuple>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

struct type_zero
{
    type_zero() : n_(757) {}

    type_zero(const type_zero&) = delete;
    type_zero(type_zero&& other) : n_(other.n_) {}

    int
    get() const
    {
        return n_;
    }

  private:
    int n_;
};

struct type_one
{
    type_one(int n) : n_(n) {}

    type_one(const type_one&) = delete;
    type_one(type_one&& other) : n_(other.n_) {}

    int
    get() const
    {
        return n_;
    }

  private:
    int n_;
};

struct type_two
{
    type_two(int n1, int n2) : n1_(n1), n2_(n2) {}

    type_two(const type_two&) = delete;
    type_two(type_two&& other) : n1_(other.n1_), n2_(other.n2_) {}

    int
    get1() const
    {
        return n1_;
    }
    int
    get2() const
    {
        return n2_;
    }

  private:
    int n1_, n2_;
};

cl::sycl::cl_bool
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::cl_bool check = false;
    sycl::range<1> numOfItem{1};

    s::pair<type_one, type_zero> pp0(s::piecewise_construct, s::forward_as_tuple(-3), s::forward_as_tuple());
    s::pair<type_one, type_two> pp1(s::piecewise_construct, s::forward_as_tuple(6), s::forward_as_tuple(5, 4));
    s::pair<type_two, type_two> pp2(s::piecewise_construct, s::forward_as_tuple(2, 1), s::forward_as_tuple(-1, -3));
    {
        sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<decltype(pp0), 1> buffer2(&pp0, numOfItem);
        sycl::buffer<decltype(pp1), 1> buffer3(&pp1, numOfItem);
        sycl::buffer<decltype(pp2), 1> buffer4(&pp2, numOfItem);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc1 = buffer1.get_access<sycl_write>(cgh);
            auto acc1 = buffer2.get_access<sycl_write>(cgh);
            auto acc2 = buffer3.get_access<sycl_write>(cgh);
            auto acc3 = buffer4.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                ret_acc1[0] = (acc1[0].first.get() == -3);
                ret_acc1[0] &= (acc1[0].second.get() == 757);

                ret_acc1[0] &= (acc2[0].first.get() == 6);
                ret_acc1[0] &= (acc2[0].second.get1() == 5);
                ret_acc1[0] &= (acc2[0].second.get2() == 4);

                ret_acc1[0] &= (acc3[0].first.get1() == 2);
                ret_acc1[0] &= (acc3[0].first.get2() == 1);
                ret_acc1[0] &= (acc3[0].second.get1() == -1);
                ret_acc1[0] &= (acc3[0].second.get2() == -3);
            });
        });
    }
    // check data after executing kernel function
    check = (pp0.first.get() == -3);
    check &= (pp0.second.get() == 757);

    check &= (pp1.first.get() == 6);
    check &= (pp1.second.get1() == 5);
    check &= (pp1.second.get2() == 4);

    check &= (pp2.first.get1() == 2);
    check &= (pp2.first.get2() == 1);
    check &= (pp2.second.get1() == -1);
    check &= (pp2.second.get2() == -3);
    if (!check)
        return false;
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
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
