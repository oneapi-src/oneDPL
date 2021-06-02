#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"
#include <vector>


//This macro is required for the tests to work correctly in CI with tbb-backend.
#if TEST_DPCPP_BACKEND_PRESENT
struct test_buffer_wrapper
{
    template <typename Iterator1, typename Iterator2>
    void operator()(const Iterator1& begin1, const Iterator2& end1, std::size_t size)
    {
        std::fill(oneapi::dpl::execution::dpcpp_default,begin1,end1,1);
        auto begin_host1 = TestUtils::get_host_pointer(begin1);
        auto end_host1 = TestUtils::get_host_pointer(end1);

        for(auto it = begin_host1; it!= end_host1; it=it+1)
        {
            EXPECT_TRUE(*it == 1,"wrong effect of reduce with sycl_iterator");
        }
        
        EXPECT_TRUE(begin1 == begin1, "operator == returned false negative");
        EXPECT_TRUE(!(begin1 == begin1 + 1), "operator == returned false positive");

        EXPECT_TRUE(begin1 != begin1 + 1, "operator != returned false negative");
        EXPECT_TRUE(!(begin1 != begin1), "operator != returned false positive");

        auto it = begin1;

        EXPECT_TRUE(it == begin1, "wrong effect of iterator's copy constructor");

        it = end1;

        EXPECT_TRUE(it == end1, "wrong effect of iterator's copy assignment operator");
        EXPECT_TRUE(it - size == begin1, "wrong effect of iterator's operator - integer");
        EXPECT_TRUE(begin1 + size == end1, "wrong effect of iterator's operator + integer");
        EXPECT_TRUE(end1 - begin1 == size, "wrong effect of iterator's operator - iterator");

        auto buf = begin1.get_buffer();
        auto begin_host2 = TestUtils::get_host_pointer(oneapi::dpl::begin(buf));
        auto end_host2 = TestUtils::get_host_pointer(oneapi::dpl::end(buf));

        for(auto it = begin_host2; it!= end_host2; it=it+1)
        {
            EXPECT_TRUE(*it == 1,"wrong effect of iterator's get_buffer method");
        }

    }
};
#endif

int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    std::size_t size = 1000;
    sycl::buffer<uint32_t> buf{size};
    test_buffer_wrapper test{};

    test(oneapi::dpl::begin(buf), oneapi::dpl::end(buf), size);
    test(oneapi::dpl::begin(buf, sycl::write_only), oneapi::dpl::end(buf, sycl::write_only), size);
    test(oneapi::dpl::begin(buf, sycl::write_only, sycl::noinit), 
            oneapi::dpl::end(buf, sycl::write_only, sycl::noinit), size);
    test(oneapi::dpl::begin(buf, sycl::noinit), oneapi::dpl::end(buf, sycl::noinit), size);

#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
