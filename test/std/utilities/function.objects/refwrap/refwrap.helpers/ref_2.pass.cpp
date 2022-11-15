
#include <functional>
#include <iostream>

#include "counting_predicates.hpp"

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<T> ref(reference_wrapper<T>t);

#if TEST_DPCPP_BACKEND_PRESENT
constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

class KernelRef2PassTest;

// bool is5 ( int i ) { return i == 5; }
class is5
{
  public:
    bool
    operator()(int i) const
    {
        return i == 5;
    }
};

template <typename T>
bool
call_pred(T pred)
{
    return pred(5);
}

void
kernel_test()
{
    cl::sycl::queue deviceQueue;
    cl::sycl::cl_bool ret = false;
    cl::sycl::range<1> numOfItems{1};
    cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl_write>(cgh);
        cgh.single_task<class KernelRef2PassTest>([=]() {
            {
                int i = 0;
                std::reference_wrapper<int> r1 = std::ref(i);
                std::reference_wrapper<int> r2 = std::ref(r1);
                ret_access[0] = (&r2.get() == &i);
            }

            {
                is5 is5_functor;
                unary_counting_predicate<is5, int> cp(is5_functor);
                ret_access[0] &= (!cp(6));
                ret_access[0] &= (cp.count() == 1);
                ret_access[0] &= (call_pred(cp));
                ret_access[0] &= (cp.count() == 1);
                ret_access[0] &= (call_pred(std::ref(cp)));
                ret_access[0] &= (cp.count() == 2);
            }
        });
    });

    auto ret_access_host = buffer1.get_access<sycl_read>();
    TestUtils::exitOnError(ret_access_host[0]);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
