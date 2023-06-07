#include "support/test_config.h"


#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include <type_traits>
#ifdef RUN_IPP
 #include "ipps.h"
 #include "ipps_l.h"
#endif

#ifdef USE_MPI
 #include "mpi.h"
 #define INIT MPI_Init(&argc, &argv); MPI_Comm_size(MPI_COMM_WORLD, &nproc); MPI_Comm_rank(MPI_COMM_WORLD, &myid);
 #define SYNC MPI_Barrier(MPI_COMM_WORLD);
 #define FINAL MPI_Finalize();
#else
 #define INIT ;
 #define SYNC ;
 #define FINAL ;
#endif

#define DUMP_CSV 1

using namespace oneapi;

constexpr double  nanosec = 1.0;
constexpr double microsec = 1.0E3;
constexpr double  milisec = 1.0E6;
constexpr int small = 16384;
constexpr int medium = 16777216;


#if 1
int main(int argc, char *argv[])
{
	int nproc {1}, myid {0};
	INIT

	int N = (argc > 1) ? std::atoi(argv[1]) : 16;
	if (N < 16)	N = 16;

	long niter = 1;
	if (N <= small)
		niter = 100;
	else if (N <= medium)
		niter = 10;

    //int Ns[8] = {320, 192, 64, 288, 96, 928, 128, 96};

    niter = 1;
    //N = 2;
    for(int p = 6; p <= 24; ++p)
    //for(int p = 2; p <= 2; ++p)
    {
        N = 1 << p;
        //N += N*0.13;

	using type = unsigned int;
	std::vector<type> in_host(N);
#if 0 //0..100
	std::default_random_engine gen{std::random_device{}()};
	std::uniform_real_distribution<float> dist(0, 100.0);
    std::generate(in_host.begin(), in_host.end(), [&]{ return dist(gen); });
#else
    srand(2009);
    std::generate(in_host.begin(), in_host.end(), 
    [&]
    { 
        type val = (type)rand() % 256;
        val |= ((type)rand() % 256) << 8;
        val |= ((type)rand() % 256) << 16;
        val |= ((type)rand() % 256) << 24; //set 0...255 value to the most significant byte
        return val;
    });
#endif
    in_host[0] = 9, in_host[1] = 4; /*, in_host[2] = 8, in_host[3] = 8; 
    in_host[4] = 0, in_host[5] = 1, in_host[6] = 1, in_host[7] = 11;
    in_host[8] = 0, in_host[9] = 4, in_host[10] = 4, in_host[11] = 11;
    in_host[12] = 0, in_host[13] = 4, in_host[14] = 4, in_host[15] = 11;*/

#if 0 // standard C++ container/algorithm
	std::vector<type> out_host(N);
	long h_nano = 0;
	for (int i = 0; i < niter; i++)
	{
		out_host = in_host;
		SYNC
		auto h1 = std::chrono::steady_clock::now();
//		std::sort(std::execution::par_unseq, out_host.begin(), out_host.end());
//		std::sort(std::execution::unseq, out_host.begin(), out_host.end());
		std::sort(out_host.begin(), out_host.end());
		SYNC
		auto h2 = std::chrono::steady_clock::now();
		h_nano += std::chrono::duration_cast<std::chrono::nanoseconds>(h2 - h1).count();
	}
	const double helap = static_cast<double>(h_nano / niter) / microsec;
	if (! std::is_sorted(out_host.cbegin(), out_host.cend()))
		std::cerr << "std::sort failed" << std::endl;
#endif // standard C++ container/algorithm

#ifdef RUN_IPP
// IPP RadixSort
	std::vector<float> out_ipp(N);
	IppSizeL pBufferSize;
	long i_nano = 0;
	for (int i = 0; i < niter; i++)
	{
		out_ipp = in_host;
		SYNC
		auto i1 = std::chrono::steady_clock::now();
		if (ippStsNoErr == ippsSortRadixGetBufferSize_L(N, ipp32f, &pBufferSize))
		{
			Ipp8u *pBuffer = ippsMalloc_8u_L(pBufferSize);
			IppStatus ret = ippsSortRadixAscend_32f_I_L(out_ipp.data(), N, pBuffer);
			ippsFree(pBuffer);
		}
		SYNC
		auto i2 = std::chrono::steady_clock::now();
		i_nano += std::chrono::duration_cast<std::chrono::nanoseconds>(i2 - i1).count();
	}
	const double ielap = static_cast<double>(i_nano / niter) / microsec;
	if (! std::is_sorted(out_ipp.cbegin(), out_ipp.cend()))
		std::cerr << "IPP RadixSort failed" << std::endl;
#endif

// DPC++
	auto Device {sycl::device(sycl::default_selector())};
	auto ndev = Device.get_info<sycl::info::device::partition_max_sub_devices>();
	std::vector<sycl::queue> Queues;
	if (ndev > 1 && Device.is_gpu())
	{
//		std::vector<size_t> counts(ndev, 1);
//		auto Subs = Device.create_sub_devices<sycl::info::partition_property::partition_by_counts>(counts);
		auto Subs = Device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
		for (auto &sub : Subs)
			Queues.emplace_back(sub);
	}
	else
	{
		ndev = 1;
		Queues.emplace_back(Device);
	}

	int qid = ((nproc+1)/ndev == 0) ? myid : myid / ((nproc+1) / ndev);
#if 0
   //auto &Q = Queues[qid];
   sycl::queue Q(sycl::gpu_selector{});
#else
    sycl::queue Q{sycl::gpu_selector{},
                sycl::property::queue::enable_profiling{}};
#endif
	if (0 == myid)
	{
		auto D = Q.get_device();
		auto P = D.get_platform();
#if !DUMP_CSV
		if (nproc > 1)
			std::cout << "\nRunning SYCL on " << ndev << "-tiles " << P.get_info<sycl::info::platform::name>() << " / " << D.get_info<sycl::info::device::name>() << "\n\n";
		else
			std::cout << "\nRunning SYCL on " << P.get_info<sycl::info::platform::name>() << " / " << D.get_info<sycl::info::device::name>() << "\n\n";
#endif
	}

#ifdef RUN_ONEDPL_CPU
// oneDPL on CPU OpenCL backend
	auto cpuQ {sycl::queue(sycl::cpu_selector())};
	float *in_cpu = sycl::malloc_device<float>(N, cpuQ);
	long c_nano = 0;
	for (int i = 0; i <= niter; i++)
	{
		cpuQ.copy(in_host.data(), in_cpu, N).wait();
		SYNC
		auto c1 = std::chrono::steady_clock::now();
		dpl::sort(dpl::execution::make_device_policy(cpuQ), in_cpu, in_cpu+N);
		SYNC
		auto c2 = std::chrono::steady_clock::now();
		if (i != 0) c_nano += std::chrono::duration_cast<std::chrono::nanoseconds>(c2 - c1).count();
	}
	const double celap = static_cast<double>(c_nano / niter) / microsec;
	type *out_cpu = sycl::malloc_host<type>(N, cpuQ);
	cpuQ.copy(in_cpu, out_cpu, N).wait();
	if (! std::is_sorted(out_cpu, out_cpu+N))
		std::cerr << "dpl::sort (CPU) failed" << std::endl;
	sycl::free(in_cpu, cpuQ); sycl::free(out_cpu, cpuQ);
#endif

// Preparation
	type *in_device = sycl::malloc_device<type>(N, Q);
	Q.copy(in_host.data(), in_device, N).wait();
	SYNC
	const auto j1 = std::chrono::steady_clock::now();
	dpl::sort(dpl::execution::make_device_policy(Q), in_device, in_device+N, std::less<type>{});	// Warmup for JIT
	const auto j2 = std::chrono::steady_clock::now();
	const auto jelap = (std::chrono::duration_cast<std::chrono::nanoseconds>(j2 - j1).count())/microsec;

// oneDPL on device
	long d_nano = 0, dt_nano = 0;
	for (int i = 0; i < niter; i++)
	{
		SYNC
		auto d0 = std::chrono::steady_clock::now();
		Q.copy(in_host.data(), in_device, N).wait();
		SYNC
		auto d1 = std::chrono::steady_clock::now();
		dpl::sort(dpl::execution::make_device_policy(Q), in_device, in_device+N, std::less<type>{});
        //dpl::exclusive_scan(dpl::execution::make_device_policy(Q), in_device, in_device+N, in_device, 5);
        //dpl::shift_right(dpl::execution::make_device_policy(Q), in_device, in_device+N, 1);
        //dpl::fill(dpl::execution::make_device_policy(Q), in_device, in_device+1, 1);
		SYNC
		auto d2 = std::chrono::steady_clock::now();
		dt_nano += std::chrono::duration_cast<std::chrono::nanoseconds>(d1 - d0).count();
		d_nano  += std::chrono::duration_cast<std::chrono::nanoseconds>(d2 - d1).count();
	}
	const double delap = static_cast<double>(d_nano / niter) / microsec;
	double dtrans = static_cast<double>(static_cast<long>(nproc)*static_cast<long>(N)*static_cast<long>(sizeof(float))) / static_cast<double>(dt_nano);
	dtrans *= 1.0E9 / 1073741824.0;
	type *out_device = sycl::malloc_host<type>(N, Q);
	Q.copy(in_device, out_device, N).wait();
#if 1 //check sorted
	if (! std::is_sorted(out_device, out_device+N, std::less<type>{}))
		std::cerr << "dpl::sort (GPU) failed" << std::endl;
#endif
	sycl::free(in_device, Q); sycl::free(out_device, Q);

#if !DUMP_CSV
    std::cout << "Iterations: "<< niter << std::endl;
	std::cout << " Subgroup sizes: ";
	for (const auto& x :
    	 Q.get_device().get_info<sycl::info::device::sub_group_sizes>()) {
  	std::cout << x << " ";
	}
	std::cout << std::endl;
#endif

	if (0 == myid)
	{
#if DUMP_CSV
        std::cout << N << "," << std::fixed << delap << std::endl;

#if 0
        std::cout << "src: ";
        for (int i = 0; i < N; i++)
            std::cout << in_host[i] << " ";
        
        std::cout << std::endl;
        std::cout << "res: ";
        for (int i = 0; i < N; i++)
            std::cout << out_device[i] << " ";
        std::cout << std::endl;
#endif
#else
		std::cout << N << " elements\n" << std::endl;
#if 0
		std::cout << "    std::sort (host): " << std::fixed << helap << " us\n";
#endif
#ifdef RUN_IPP
		std::cout << "IPP SortRadix (host): " << std::fixed << ielap << " us\n";
#endif
#ifdef RUN_ONEDPL_CPU
		std::cout << "    dpl::sort  (CPU): " << std::fixed << celap << " us\n";
#endif
		std::cout << "    dpl::sort  (GPU): " << std::fixed << delap << " us [ + " << std::max(0.0, jelap - delap) << " us for JIT ]\n";
		std::cout << "     transfer  (GPU): " << std::fixed << dtrans << " GiB/s\n\n";
#if 0		
		std::cout << " oneDPL/GPU vs. host: " << std::fixed << helap/delap << " times with " << N << " elements\n" << std::endl;	
#endif
#endif //DUMP_CSV
	}

	FINAL

    }
	return 0;
}
#endif
