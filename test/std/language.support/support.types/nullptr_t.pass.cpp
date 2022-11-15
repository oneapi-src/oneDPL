//===----------------------------------------------------------------------===//
//
// typedef decltype(nullptr) nullptr_t;
//
//===----------------------------------------------------------------------===//

#include "oneapi_std_test_config.h"
#include "test_macros.h"

#include <iostream>

#ifdef USE_ONEAPI_STD
#    include _ONEAPI_STD_TEST_HEADER(cstddef)
#    include _ONEAPI_STD_TEST_HEADER(type_traits)
#    include _ONEAPI_STD_TEST_HEADER(utility)
namespace s = oneapi_cpp_ns;
#else
#    include <cstddef>
#    include <type_traits>
namespace s = std;
#endif

#if TEST_DPCPP_BACKEND_PRESENT
struct A
{
    A(std::nullptr_t) {}
};

template <class T>
void
test_conversions(cl_int& i)
{
    {
        T p = 0;
        i += (p == nullptr);
    }
    {
        T p = nullptr;
        i += (p == nullptr);
        i += (nullptr == p);
        i += (!(p != nullptr));
        i += (!(nullptr != p));
    }
}

template <class T>
struct Voider
{
    typedef void type;
};
template <class T, class = void>
struct has_less : s::false_type
{
};

template <class T>
struct has_less<T, typename Voider<decltype(s::declval<T>() < nullptr)>::type> : s::true_type
{
};

template <class T>
void
test_comparisons(cl_int& i)
{
    T p = nullptr;
    i += (p == nullptr);
    i += (!(p != nullptr));
    i += (nullptr == p);
    i += (!(nullptr != p));
}

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnull-conversion"
#endif
void
test_nullptr_conversions(cl_int& i)
{
    // GCC does not accept this due to CWG Defect #1423
    // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1423
    {
        bool b(nullptr);
        i += (!b);
    }
}
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(int, char**)
{
#if TEST_DPCPP_BACKEND_PRESENT
    const s::size_t N = 1;
    bool ret = true;

    {
        cl::sycl::buffer<bool, 1> buf(&ret, cl::sycl::range<1>{N});
        cl::sycl::queue q;
        q.submit([&](cl::sycl::handler& cgh) {
            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                static_assert(sizeof(std::nullptr_t) == sizeof(void*), "sizeof(s::nullptr_t) == sizeof(void*)");

                cl_int i = 0;
                {
                    test_conversions<std::nullptr_t>(i);
                    test_conversions<void*>(i);
                    test_conversions<A*>(i);
                    test_conversions<int A::*>(i);
                }
                {
#ifdef _LIBCPP_HAS_NO_NULLPTR
                    static_assert(!has_less<std::nullptr_t>::value, "");
#endif
                    test_comparisons<std::nullptr_t>(i);
                    test_comparisons<void*>(i);
                    test_comparisons<A*>(i);
                }
                test_nullptr_conversions(i);

                acc[0] &= (i == 33);
            });
        });
    }

    TestUtils::exitOnError(ret);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
