#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include <CL/sycl.hpp>

template <typename T>
void
dump_array(const char* message, int size, T data)
{
    std::cout << message;
    for (int i = 0; i < size; i++)
        std::cout << " " << data[i];
    std::cout << std::endl;
}

int
main(int argc, char* argv[])
{
    int size = 10;

    auto sycl_asynchandler = [](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const& e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& ex)
            {
                std::cout << "Caught asynchronous SYCL exception:" << std::endl << ex.what() << std::endl;
            }
        }
    };

    //sycl::device* sycl_device = new sycl::device(sycl::gpu_selector{});
    sycl::device* sycl_device = new sycl::device(sycl::cpu_selector{});
    sycl::context sycl_ctxt = sycl::context(*sycl_device, sycl_asynchandler);
    sycl::queue q(sycl_ctxt, *sycl_device, sycl::property_list{sycl::property::queue::in_order{}});

    int* keys_in  = (int*)sycl::malloc_shared(size * sizeof(int), q);
    int* vals_in  = (int*)sycl::malloc_shared(size * sizeof(int), q);
    int* vals_out = (int*)sycl::malloc_shared(size * sizeof(int), q);

    /* Set initial values */
    for (int i = 0; i < size; i++)
    {
        /* keys_in[i] = i; */
        keys_in [i] = 2 * i / size;
        vals_in [i] = 1;
        vals_out[i] = 0;
    }

    /* Print input values to screen */
    dump_array("keys_in:", size, keys_in);
    dump_array("vals_in:", size, vals_in);

    /* Do the exclusive scan by segment */
    std::cout << "Calling exclusive_scan_by_segment" << std::endl;
    oneapi::dpl::exclusive_scan_by_segment(
        oneapi::dpl::execution::make_device_policy(q),
        keys_in,            /* key begin */
        keys_in + size,     /* key end */
        vals_in,            /* input value begin */
        vals_out,           /* output value begin */
        0,                  /* init */
        std::equal_to<int>(), std::plus<int>());

    /* Print results to screen */
    dump_array("vals_out:", size, vals_out);

    /* Setup a trivial permuation */
    std::cout << "Calling exclusive_scan_by_segment with a permutation iterator" << std::endl;
    int* trivial_perm = (int*)sycl::malloc_shared(size * sizeof(int), q);
    for (int i = 0; i < size; i++)
    {
        trivial_perm[i] = i;
    }

    auto it_key_begin = oneapi::dpl::make_permutation_iterator(keys_in, trivial_perm);
    auto it_key_end = it_key_begin + size;
    auto it_val_input = oneapi::dpl::make_permutation_iterator(vals_in, trivial_perm);
    auto it_val_output = oneapi::dpl::make_permutation_iterator(vals_out, trivial_perm);

    dump_array("trivial_perm:", size, trivial_perm);
    dump_array("keys (tp):", size, it_key_begin);
    dump_array("vals (tp):", size, it_val_input);

    /* Do the exclusive scan by segment with permutation */
    oneapi::dpl::exclusive_scan_by_segment(
        oneapi::dpl::execution::make_device_policy(q),
        it_key_begin,       /* key begin */
        it_key_end,         /* key end */
        it_val_input,       /* input value begin */
        it_val_output,      /* output value begin */
        0,                  /* init */
        std::equal_to<int>(), std::plus<int>());

    /* Print results to screen */
    dump_array("vals_out:", size, vals_out);

    std::cout << "DONE" << std::endl;
    return 0;
}
