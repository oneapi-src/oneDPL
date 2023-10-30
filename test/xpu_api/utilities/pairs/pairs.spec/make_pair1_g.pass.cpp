#include "oneapi_std_test_config.h"
#include "test_macros.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
namespace s = std;
#endif

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

class test_obj
{
    int i;

  public:
    test_obj(int arg = 0) : i(arg) {}
    bool
    operator==(const test_obj& rhs) const
    {
        return i == rhs.i;
    }
    bool
    operator<(const test_obj& rhs) const
    {
        return i < rhs.i;
    }
};

template <typename T>
struct test_t
{
    bool b;

  public:
    test_t(bool arg = 0) : b(arg) {}
    bool
    operator==(const test_t& rhs) const
    {
        return b == rhs.b;
    }
    bool
    operator<(const test_t& rhs) const
    {
        return int(b) < int(rhs.b);
    }
};

// heterogeneous
sycl::cl_bool
kernel_test()
{
    sycl::queue deviceQueue;
    sycl::cl_bool ret = false;
    sycl::cl_bool check = false;
    sycl::range<1> numOfItem{1};
    typedef s::pair<bool, long> PBL;
    typedef s::pair<test_t<long>, test_obj> PST;
    PBL p_bl_1(true, 433);
    PBL p_bl_2 = s::make_pair(true, 433);
    PST p_st_1(test_t<long>(false), test_obj(5));
    PST p_st_2 = s::make_pair(test_t<long>(false), test_obj(5));
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<sycl::cl_bool, 1> buffer2(&check, numOfItem);
        sycl::buffer<PBL, 1> buffer3(&p_bl_1, numOfItem);
        sycl::buffer<PBL, 1> buffer4(&p_bl_2, numOfItem);
        sycl::buffer<PST, 1> buffer5(&p_st_1, numOfItem);
        sycl::buffer<PST, 1> buffer6(&p_st_2, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            auto check_acc = buffer2.get_access<sycl_write>(cgh);
            auto acc1 = buffer3.get_access<sycl_write>(cgh);
            auto acc2 = buffer4.get_access<sycl_write>(cgh);
            auto acc3 = buffer5.get_access<sycl_write>(cgh);
            auto acc4 = buffer6.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                // check if there is change from input after data transfer
                check_acc[0] = (acc1[0].first == true);
                check_acc[0] &= (acc1[0].second == 433);
                check_acc[0] &= (acc3[0].first == test_t<long>(false));
                check_acc[0] &= (acc3[0].second == test_obj(5));
                if (check_acc[0])
                {
                    ret_acc[0] = (acc1[0] == acc2[0]);
                    ret_acc[0] &= !(acc1[0] < acc2[0]);
                    ret_acc[0] &= (acc3[0] == acc4[0]);
                    ret_acc[0] &= !(acc3[0] < acc4[0]);
                }
            });
        });
    }
    // check data after executing kernel function
    check &= (p_bl_1.first == true);
    check &= (p_bl_1.second == 433);
    check &= (p_bl_1 == p_bl_2);
    check &= (p_st_1.first == test_t<long>(false));
    check &= (p_st_1.second == test_obj(5));
    check &= (p_st_1 == p_st_2);
    if (!check)
        return false;
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    if (ret)
    {
        std::cout << "pass" << std::endl;
    }
    else
    {
        std::cout << "fail" << std::endl;
    }
    return 0;
}
