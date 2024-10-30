# Host backends support for the histogram APIs

## Introduction
histogram added spec, gpu. Not supported with serial tbb, openmp backends. Not part of stl. 

Serial implementation is straightforward and is not worth discussing in much length here. We will add it but there is not much to discuss there.

## Motivations
Users don't always want to use device policies and accelerators to run their code. It may make more sense in many cases to use a serial implementation or a host side arallel imlementation or even a vector implementation of histogram. It's natural for a user to expect that one to deal with support these other back ends for all APIs.
Another motivation for adding the support is simply to be spec compliant with the oneAPI specification.

## Design considerations
For execution policies when calling oneDPL APIs. They also have a number of options for backends which they can select from when using one dpl. It is important that all of these options have some support for histogram.
We also care about how these perform and how they scale to the number of threads, and we also care about their memory footprint.

In general I believe we can safely assume that the normal case is for the number of elements to be far greater than the number of bins.Also this is a very low computation api which will likely be limited by memory bandwidth. This means  we should

### Key requirements
seq, unseq, par, par_unseq
serial, tbb, openmp

### Performance
As with all algorithms in oneDPL, our goal is to make them a performant as possible.

### Memory Footprint
There are no guidelines here from the standard library as this is an extension API. However, we should always try to minimize memory footprint whenever possible. Minimizing memory footprint may also help us improve performance here because as mentioned above this will be very likely to be a memory bandwidth bound api.

### Code Reuse
Our goal here is to make something maintainable and to reuse as much as we can which already exists and has been reviewed within oneDPL. With everything else this has to be balanced with performance considerations.

### unseq backend
As mentioned above histogram looks to be a memory bandwidth dependent algorithm. This means that we are unlikely to get much benefit from vector instructions as they provide assistance mostly in speeding up computation. Vector operations in this case also compound our issue of race conditions multiplying the number of virtual threads by the vector length. The advantage we get from vectorization of the increment operation or the lookup into the output histogram is unlikely to provide much benefit especially when we account for extra memory footprint required or synchronization required to overcome the race conditions which we add from the additional concurrant streams of execution. It may make sense to decline to add vectorized operations within histogram even when they are requested by the user via the execution policy.

## Existing patterns

### count_if

Histogram is similar to count if in that it is conditionally incrementing a number of counters based upon the data in a sequence. Count_if returns a different type which is a scalar, And doesn't provide any function To modify the variable being incremented.  Using count_if without modification would require us to loop through the entire sequence for each output bin in the histogram. From a memory bandwidth perspective this seems far from ideal.

### parallel_for

Parallel_for is an interesting pattern in that it is very generic and should allow us to do what we want to do for histogram ultimately. However we cannot simply use it without any added infrastructure. If we were to just run it through a parallel_for there would be a race condition between threads when incrementing the values in the output histogram.I believe parallel_for will be a part of our implementation but it requires some way to synchronize and accumulate between threads.

## Proposal
I propose to add a new pattern specific to histogram which goes as follows:
1) Determine the number of threads that we will use, perhaps add some method to do this generically based on back end.
2) Create temporary data for the number of threads minus one copies of the histogram output sequence.
3) Run a parallel_for pattern which performs a histogram on The input sequence where each thread accumulates into its own copy of the histogram using both the temporary storage and provided output sequence to remove any race conditions.
4) Run a second parallel for over the histogram sequence which accumulates all temporary copies of the histogram into the output histogram sequence.

New machinery that will be required here is the ability to query how many threads will be used and also the machinery to check what thread the current execution is using. Ideally these can be generic wrappers around the specific back ends which would allow a unified implementation for all host backends.

## Alternative Option
One alternative way to provide a parallel histogram which would minimize memory footprint would be to use atomic operations to remove the race conditions during accumulation. The user provides the output sequence It won't be a atomic variable. Open MP does provide wrappers around generic memory to provide atomic operations within an open MP parallel section however I do not know of a way to provide this within the tbb back end. We can Alternatively allocate a copy of the histogram as atomic variables and use them however this would require us to add a copy from the atomic copy of the histogram to the output sequence provided by the user. With large enough histogram bin counts relative to the number of threads, atomics may be an attractive solution because contention on the atomics will be relatively low. It also limits the requirement for extra temporary storage. Especially for open MP it may make sense to explore this option and compare performance.

## Open Questions
If we had access to std::atomic_ref from C++20, atomics may be a better option for many cases, without the need for extra allocation or copies / accumulation.
Would it be worthwhile to add our own implementation of atomic_ref for C++17? I believe this would require specializations for each of our supported compilers.

What is the overhead of atomics in general in this case and does the overhead there make them inherently worse than merely having extra copies of the histogram and accumulating?

Is it worthwhile to have separate implementations for tbb and openMP because they may differ in the best performing implementation?