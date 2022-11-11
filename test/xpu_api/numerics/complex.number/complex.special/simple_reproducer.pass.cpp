#include <CL/sycl.hpp>

int
main()
{
    sycl::queue deviceQueue;

    // Example from https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:kernel.attributes
    if (deviceQueue.get_device().has(sycl::aspect::fp64))
    {
        deviceQueue.submit(
            [](sycl::handler & cgh) [[sycl::device_has(sycl::aspect::fp64)]]
            {
                cgh.single_task<class Test1>(
                    []()
                    {
                        double d = 1;
                        d = d + 1;
                        d = d;
                    });
            });
    }
    else
    {
        deviceQueue.submit(
            [](sycl::handler& cgh)
            {
                cgh.single_task<class Test2>(
                    []()
                    {
                        float f = 1;
                        f = f + 1;
                        f = f;
                    });
            });
    }

    return 0;
}
