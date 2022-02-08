#include "oneapi_std_test_config.h"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(functional)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
namespace s = oneapi_cpp_ns;
#else
#    include <functional>
#    include <type_traits>
namespace s = std;
#endif

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelWeakResultTest;

template <class Arg, class Result>
struct my_unary_function
{ // std::unary_function was removed in C++17
    typedef Arg argument_type;
    typedef Result result_type;
};

template <class Arg1, class Arg2, class Result>
struct my_binary_function
{ // std::binary_function was removed in C++17
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};

class functor1 : public my_unary_function<int, char>
{
};

class functor2 : public my_binary_function<char, int, float>
{
};

class functor3 : public my_unary_function<char, int>, public my_binary_function<char, int, float>
{
  public:
    typedef float result_type;
};

class functor4 : public my_unary_function<char, int>, public my_binary_function<char, int, float>
{
  public:
};

class C
{
};

template <class T>
struct has_result_type
{
  private:
    struct two
    {
        char _;
        char __;
    };
    template <class U>
    static two
    test(...);
    template <class U>
    static char
    test(typename U::result_type* = 0);

  public:
    static const bool value = sizeof(test<T>(0)) == 1;
};

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelWeakResultTest>([=]() {
            // Static assert check...
            static_assert((s::is_same<s::reference_wrapper<functor1>::result_type, char>::value), "");
            static_assert((s::is_same<s::reference_wrapper<functor2>::result_type, float>::value), "");
            static_assert((s::is_same<s::reference_wrapper<functor3>::result_type, float>::value), "");
            static_assert((s::is_same<s::reference_wrapper<void()>::result_type, void>::value), "");
            static_assert((s::is_same<s::reference_wrapper<int*(float*)>::result_type, int*>::value), "");
            static_assert((s::is_same<s::reference_wrapper<void (*)()>::result_type, void>::value), "");
            static_assert((s::is_same<s::reference_wrapper<int* (*)(float*)>::result_type, int*>::value), "");
            static_assert((s::is_same<s::reference_wrapper<int* (C::*)(float*)>::result_type, int*>::value), "");
            static_assert(
                (s::is_same<s::reference_wrapper<int (C::*)(float*) const volatile>::result_type, int>::value), "");
            static_assert((s::is_same<s::reference_wrapper<C()>::result_type, C>::value), "");
            static_assert(has_result_type<s::reference_wrapper<functor3>>::value, "");
            static_assert(!has_result_type<s::reference_wrapper<functor4>>::value, "");
            static_assert(!has_result_type<s::reference_wrapper<C>>::value, "");

            // Runtime check...

            ret_access[0] = s::is_same<s::reference_wrapper<functor1>::result_type, char>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<functor2>::result_type, float>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<functor3>::result_type, float>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<void()>::result_type, void>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<int*(float*)>::result_type, int*>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<void (*)()>::result_type, void>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<int* (*)(float*)>::result_type, int*>::value;

            ret_access[0] &= s::is_same<s::reference_wrapper<int* (C::*)(float*)>::result_type, int*>::value;
            ret_access[0] &=
                s::is_same<s::reference_wrapper<int (C::*)(float*) const volatile>::result_type, int>::value;
            ret_access[0] &= s::is_same<s::reference_wrapper<C()>::result_type, C>::value;
            ret_access[0] &= has_result_type<s::reference_wrapper<functor3>>::value;
            ret_access[0] &= !has_result_type<s::reference_wrapper<functor4>>::value;
            ret_access[0] &= !has_result_type<s::reference_wrapper<C>>::value;
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
