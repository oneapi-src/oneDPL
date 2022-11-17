#include "oneapi_std_test_config.h"

#include <iostream>
#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(array)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <array>
#    include <utility>
#    include <type_traits>
namespace s = std;
#endif
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

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
can_swap_test(int) -> decltype(s::swap(std::declval<Tp>(), s::declval<Tp>()));

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
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);

    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelSwapTest>([=]() {
            {
                int i[3] = {1, 2, 3};
                int j[3] = {4, 5, 6};
                s::swap(i, j);
                ret_access[0] = (i[0] == 4);
                ret_access[0] &= (i[1] == 5);
                ret_access[0] &= (i[2] == 6);
                ret_access[0] &= (j[0] == 1);
                ret_access[0] &= (j[1] == 2);
                ret_access[0] &= (j[2] == 3);
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
                // test that the swap
                using CA = CopyOnly[42];
                using MA = NoexceptMoveOnly[42];
                using NA = NotMoveConstructible[42];
                static_assert(can_swap<CA&>(), "");
                static_assert(can_swap<MA&>(), "");
                static_assert(!can_swap<NA&>(), "");

                CA ca;
                MA ma;
                static_assert(!noexcept(s::swap(ca, ca)), "");
                static_assert(noexcept(s::swap(ma, ma)), "");
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    if (ret_access_host[0])
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
