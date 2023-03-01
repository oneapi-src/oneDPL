#define USE_ESIMD_SORT 1
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/ranges>

#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>
#include <random>

template<typename T>
void verify(const std::vector<T>& input, const std::vector<T>& ref)
{
    uint32_t err_count = 0;
    for(uint32_t i = 0; i < input.size(); ++i)
    {
        if(input[i] != ref[i])
        {
            ++err_count;
            if(err_count <= 5)
            {
                std::cout << "input[" << i << "] = " << input[i] << ", expected: " << ref[i] << std::endl;
            }
        }
    }
    if (err_count != 0)
    {
        std::cout << "error count: " << err_count << std::endl;
        std::cout << "n: " << input.size() << std::endl;
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void generate_data(std::vector<T>& in)
{
    std::default_random_engine gen{std::random_device{}()};
    if constexpr (std::is_integral_v<T>)
    {
        // std::uniform_int_distribution<T> dist(0, in.size());
        // std::generate(in.begin(), in.end(), [&]{ return dist(gen); });
        for(uint32_t i = 0; i < in.size(); ++i)
        {
            in[i] = i % 256;
        }
    }
    else
    {
        std::uniform_real_distribution<T> dist(0.0, 100.0);
        std::generate(in.begin(), in.end(), [&]{ return dist(gen); });
    }
}

template<typename T>
void test_all_view(uint32_t n)
{
    namespace dpl = oneapi::dpl;
    namespace dpl_ranges = dpl::experimental::ranges;

    std::vector<T> in(n);
    generate_data(in);
    std::vector<T> ref(in);
    std::sort(std::begin(ref), std::end(ref));
    {
        sycl::buffer<T> buf(in.data(), in.size());
        dpl_ranges::all_view<T, sycl::access::mode::read_write> view(buf);
        oneapi::dpl::experimental::esimd::radix_sort(dpl::execution::dpcpp_default, view);
    }
    verify(in, ref);
}

/*
template<typename T>
void test_subrange_view(uint32_t n, sycl::queue& q)
{
    namespace dpl = oneapi::dpl;
    namespace dpl_ranges = dpl::experimental::ranges;

    T* p_in = sycl::malloc_shared<T>(n, q);
    generate_data(p_in, n);
    std::vector<T> ref(p_in, p_in + p);
    std::sort(ref);
    {
        dpl_ranges::views::subrange<T, sycl::access::mode::read_write> view(p_in, p_in + n);
        oneapi::dpl::experimental::esimd::radix_sort(dpl::execution::dpcpp_default, view);
    }
    verify(in, ref);
}


template<typename T>
test_sycl_iterators()
{

}

template<typename T>
test_usm()
{

}
*/

int main()
{
    // test_all_view<uint32_t>(16);
    // test_all_view<uint32_t>(96);
    // test_all_view<uint32_t>(256);
    test_all_view<uint32_t>(512);
    // test_all_view<uint32_t>(2024);
    // test_all_view<uint32_t>(32768);
    // test_all_view<uint32_t>(524228);
    return 0;
}
