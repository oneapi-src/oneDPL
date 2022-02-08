#include <CL/sycl.hpp>
#include <functional>
#include <type_traits>
#include <iostream>

// <functional>

// reference_wrapper

// template <class... ArgTypes>
//   requires Callable<T, ArgTypes&&...>
//   Callable<T, ArgTypes&&...>::result_type
//   operator()(ArgTypes&&... args) const;

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

struct A_void_1
{
    int* count_A_1;
    A_void_1(int* count) { count_A_1 = count; }
    void
    operator()(int i)
    {
        *count_A_1 += i;
    }

    void
    mem1()
    {
        ++(*count_A_1);
    }
};

struct A_int_1
{
    A_int_1() : data_(5) {}
    int
    operator()(int i)
    {
        return i - 1;
    }

    int
    mem1()
    {
        return 3;
    }
    int
    mem2() const
    {
        return 4;
    }
    int data_;
};

struct A_void_2
{
    A_void_2(int* c) { count = c; }
    void
    operator()(int i, int j)
    {
        *count += i + j;
    }

    int* count;
};

struct A_int_2
{
    int
    operator()(int i, int j)
    {
        return i + j;
    }

    int
    mem1(int i)
    {
        return i + 1;
    }
    int
    mem2(int i) const
    {
        return i + 2;
    }
};

class KernelInvokePassTest;

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelInvokePassTest>([=]() {
            int count = 0;
            int save_count = count;

            {
                A_void_1 a0(&count);
                std::reference_wrapper<A_void_1> r1(a0);
                int i = 4;
                r1(i);
                ret_access[0] = (count == save_count + 4);
                save_count = count;
                a0.mem1();
                ret_access[0] &= (count == save_count + 1);
                save_count = count;
            }

            {
                A_int_1 a0;
                std::reference_wrapper<A_int_1> r1(a0);
                int i = 4;
                ret_access[0] &= (r1(i) == 3);
            }

            // member data pointer
            {
                int A_int_1::*fp = &A_int_1::data_;
                std::reference_wrapper<int A_int_1::*> r1(fp);
                A_int_1 a;
                ret_access[0] &= (r1(a) == 5);
                r1(a) = 6;
                ret_access[0] &= (r1(a) == 6);
                A_int_1* ap = &a;
                ret_access[0] &= (r1(ap) == 6);
                r1(ap) = 7;
                ret_access[0] &= (r1(ap) == 7);
            }

            {
                A_void_2 a0(&count);
                std::reference_wrapper<A_void_2> r1(a0);
                int i = 4;
                int j = 5;
                r1(i, j);
                ret_access[0] &= (count == save_count + 9);
                save_count = count;
            }

            {
                A_int_2 a0;
                std::reference_wrapper<A_int_2> r1(a0);
                int i = 4;
                int j = 5;
                ret_access[0] &= (r1(i, j) == i + j);
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
