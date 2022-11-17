#include "oneapi_std_test_config.h"

#include <complex>
#include <iostream>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <utility>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

struct CopyOnly
{
    CopyOnly() {}
    CopyOnly(CopyOnly const&) noexcept {}
    CopyOnly&
    operator=(CopyOnly const&)
    {
        return *this;
    }
};

struct MoveOnly
{
    MoveOnly() {}
    MoveOnly(MoveOnly&&) {}
    MoveOnly&
    operator=(MoveOnly&&) noexcept
    {
        return *this;
    }
};

struct NoexceptMoveOnly
{
    NoexceptMoveOnly() {}
    NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
    NoexceptMoveOnly&
    operator=(NoexceptMoveOnly&&) noexcept
    {
        return *this;
    }
};

struct NotMoveConstructible
{
    NotMoveConstructible&
    operator=(NotMoveConstructible&&)
    {
        return *this;
    }

  private:
    NotMoveConstructible(NotMoveConstructible&&);
};

struct NotMoveAssignable
{
    NotMoveAssignable(NotMoveAssignable&&);

  private:
    NotMoveAssignable&
    operator=(NotMoveAssignable&&);
};

template <class Tp>
auto
can_swap_test(int) -> decltype(s::swap(s::declval<Tp>(), s::declval<Tp>()));

template <class Tp>
auto
can_swap_test(...) -> s::false_type;

template <class Tp>
constexpr bool
can_swap()
{
    return s::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}

class KernelSwapTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::range<1> numOfItems_acc{2};
    int acc[2] = {1, 2};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
        cl::sycl::buffer<int, 1> acc_buffer(acc, numOfItems_acc);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto acc_dev = acc_buffer.get_access<sycl_read_write>(cgh);
            cgh.single_task<class KernelSwapTest>([=]() {
                {
                    int i = 1;
                    int j = 2;
                    s::swap(i, j);
                    ret_access[0] = (i == 2);
                    ret_access[0] &= (j == 1);
                }

                {
                    int a = 1;
                    int b = 2;
                    int* i = &a;
                    int* j = &b;
                    s::swap(i, j);
                    ret_access[0] &= (*i == 2);
                    ret_access[0] &= (*j == 1);
                }

                {
                    std::complex<float> c1(1.5f, 2.5f);
                    std::complex<float> c2(1.f, 5.5f);
                    ret_access[0] &= (c1.real() == 1.5f && c1.imag() == 2.5f);
                    ret_access[0] &= (c2.real() == 1.f && c2.imag() == 5.5f);
                    s::swap(c1, c2);
                    ret_access[0] &= (c2.real() == 1.5 && c2.imag() == 2.5);
                    ret_access[0] &= (c1.real() == 1 && c1.imag() == 5.5);
                }

                {
                    // test that the swap
                    static_assert(can_swap<CopyOnly&>(), "");
                    static_assert(can_swap<MoveOnly&>(), "");
                    static_assert(can_swap<NoexceptMoveOnly&>(), "");

                    static_assert(!can_swap<NotMoveConstructible&>(), "");
                    static_assert(!can_swap<NotMoveAssignable&>(), "");

                    CopyOnly c;
                    MoveOnly m;
                    NoexceptMoveOnly nm;
                    static_assert(!noexcept(s::swap(c, c)), "");
                    static_assert(!noexcept(s::swap(m, m)), "");
                    static_assert(noexcept(s::swap(nm, nm)), "");
                }

                {
                    ret_access[0] &= (acc_dev[0] == 1);
                    ret_access[0] &= (acc_dev[1] == 2);
                    s::swap(acc_dev[0], acc_dev[1]);
                }
            });
        });
    }

    if (ret && acc[0] == 2 && acc[1] == 1)
    {
        std::cout << "Pass" << std::endl;
    }
    else
    {
        std::cout << "Fail" << std::endl;
    }
}

int
main()
{

    kernel_test();
    return 0;
}
