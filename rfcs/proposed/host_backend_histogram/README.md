# Host Backends Support for the Histogram APIs

## Introduction
In version 2022.6.0, two `histogram` APIs were added to oneDPL, but implementations were only provided for device
policies with the dpcpp backend. `Histogram` was added to the oneAPI specification 1.4 provisional release and should
be present in the 1.4 specification. Please see the
[oneAPI Specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneDPL/source/parallel_api/algorithms.rst#parallel-algorithms)
for a full definition of the semantics of the histogram APIs. In short, they take elements from an input sequence and
classify them into either evenly distributed or user-defined bins via a list of separating values and count the number
of values in each bin, writing to a user-provided output histogram sequence. Currently, `histogram` is not supported
with serial, tbb, or openmp backends in our oneDPL implementation. This RFC aims to propose the implementation of
`histogram` for these host-side backends.

## Motivations
Users don't always want to use device policies and accelerators to run their code. It may make more sense in many cases
to use a serial implementation or a host-side parallel implementation of `histogram`. It's natural for a user to expect
that oneDPL supports these other backends for all APIs. Another motivation for adding the support is simply to be spec
compliant with the oneAPI specification.

## Design Considerations

### Key Requirements
Provide support for the `histogram` APIs with the following policies and backends:
- Policies: `seq`, `unseq`, `par`, `par_unseq`
- Backends: `serial`, `tbb`, `openmp`

Users have a choice of execution policies when calling oneDPL APIs. They also have a number of options of backends
which they can select from when using oneDPL. It is important that all combinations of these options have support for
the `histogram` APIs.

### Performance
As with all algorithms in oneDPL, our goal is to make them as performant as possible. By definition, `histogram` is a
low computation algorithm which will likely be limited by memory bandwidth, especially for the evenly-divided case.
Minimizing and optimizing memory accesses, as well as limiting unnecessary memory traffic of temporaries, will likely
have a high impact on overall performance.

### Memory Footprint
There are no guidelines here from the standard library as this is an extension API. However, we should always try to
minimize memory footprint whenever possible. Minimizing memory footprint may also help us improve performance here
because, as mentioned above, this will very likely be a memory bandwidth-bound API. In general, the normal case for
histogram is for the number of elements in the input sequence to be far greater than the number of output histogram
bins. We may be able to use that to our advantage.

### Code Reuse
Our goal here is to make something maintainable and to reuse as much as we can which already exists and has been
reviewed within oneDPL. With everything else, this must be balanced with performance considerations.

### unseq Backend
Currently oneDPL relies upon openMP SIMD to provide its vectorization, which is designed to provide vectorization across
loop iterations. OneDPL does not directly use any intrinsics which may offer more complex functionality than what is
provided by OpenMP.

As mentioned above, histogram looks to be a memory bandwidth-dependent algorithm. This may limit the benefit achievable
from vector instructions as they provide assistance mostly in speeding up computation.

For histogram, there are a few things to consider. First, lets consider the calculation to determine which bin to
increment. There are two APIs, even and custom range which have significantly different methods to determine the bin to
increment. For the even bin API, the calculations to determine selected bin have some opportunity for vectorization as
each input has the same mathematical operations applied to each. However, for the custom range API, each input element
uses a binary search through a list of bin boundaries to determine the selected bin. This operation will have a
different length and control flow based upon each input element and will be very difficult to vectorize.

Second, lets consider the increment operation itself. This operation increments a data dependant bin location, and may
result in conflicts between elements of the same vector. This increment operation therefore is unvectorizable without
more complex handling. Some hardware does implement SIMD conflict detection via specific intrinsics, but this is not
generally available, and certainly not available via OpenMP SIMD. Alternatively, we can multiply our number of temporary
histogram copies by a factor of the vector width, but we will need to determine if this is worth the overhead, memory
footprint, and extra accumulation at the end. OpenMP SIMD does provide an `ordered` structured block which we can use to
exempt the increment from SIMD operations as well. It must be determined if SIMD is beneficial in either API variety. It
seems only possible to be beneficial for the even bin API, but more investigation is required.

Finally, for our below proposed implementation, there is the task of combining temporary histogram data into the global
output histogram. This is directly vectorizable via our existing brick_walk implementation.

### Serial Backend
We plan to support a serial backend for histogram APIs in addition to openMP and TBB. This backend will handle all
policies types, but always provide a serial unvectorized implementation.

## Existing Patterns

### count_if
`histogram` is similar to `count_if` in that it conditionally increments a number of counters based upon the data in a
sequence. `count_if` returns a scalar-typed value and doesn't provide any function to modify the variable being
incremented. Using `count_if` without significant modification would require us to loop through the entire sequence for
each output bin in the histogram. From a memory bandwidth perspective, this is untenable. Similarly, using a
`histogram` pattern to implement `count_if` is unlikely to provide a well-performing result in the end, as contention
should be far higher, and `reduce` is a very well-matched pattern performance-wise.

### parallel_for
`parallel_for` is an interesting pattern in that it is very generic and embarrassingly parallel. This is close to what
we need for `histogram`. However, we cannot simply use it without any added infrastructure. If we were to just use
`parallel_for` alone, there would be a race condition between threads when incrementing the values in the output
histogram. We should be able to use `parallel_for` as a building block for our implementation, but it requires some way
to synchronize and accumulate between threads.


## Alternative Approaches

### Atomics
This method uses atomic operations to remove the race conditions during accumulation. With atomic increments of the
output histogram data, we can merely run a `parallel_for` pattern.

To deal with atomics appropriately, we have some limitations. We must either use standard library atomics, atomics
specific to a backend, or custom atomics specific to a compiler. `C++17` provides `std::atomic<T>`, however, this can
only provide atomicity for data which is created with atomics in mind. This means allocating temporary data and then
copying it to the output data. `C++20` provides `std::atomic_ref<T>` which would allow us to wrap user-provided output
data in an atomic wrapper, but we cannot assume `C++20` for all users. OpenMP provides atomic
operations, but that is only available for the OpenMP backend.  The working plan was to implement a macro like
`_ONEDPL_ATOMIC_INCREMENT(var)` which uses an `std::atomic_ref` if available, and alternatively uses compiler builtins
like `InterlockedAdd` or `__atomic_fetch_add_n`. In a proof of concept implementation,this seemed to work, but does
reach more into details than compiler / OS specifics than is desired for implementations prior to `C++20`.

After experimenting with a proof of concept implementation of this implementation, it seems that the atomic
implementation has very limited applicability to real cases. We explored a spectrum of number of elements combined with
number of bins with both OpenMP and TBB. There was some subset of cases for which the atomics implementation
outperformed the proposed implementation (below). However, this was generally limited to some specific cases where
the number of bins was very large (~1 Million), and even for this subset significant benefit was only found for cases
with a small number for input elements relative to number of bins. This makes sense because the atomic implementation
is able to avoid the overhead of allocating and initializing temporary histogram copies, which is largest when
the number of bins is large compared to the number of input elements. With many bins, contention on atomics is also
limited as compared to the embarassingly parallel proposal which does experience this contention.

When we examine the real world utility of these cases, we find that they are uncommon and unlikely to be the important
use cases. Histograms generally are used to categorize large images or arrays into a smaller number of bins to
characterize the result. Cases for which there are similar or more bins than input elements are not very practical in
practice. The maintenance and complexity cost associated with supporting and maintaining a second implementation to
serve this subset of cases does not seem to be justified. Therefore, this implementation has been discarded at this
time.

### Other Unexplored Approaches
* One could consider some sort of locking approach which locks mutexes for subsections of the output histogram prior to
  modifying them. It's possible such an approach could provide a similar approach to atomics, but with different
  overhead trade-offs. It seems quite likely that this would result in more overhead, but it could be worth exploring.

* Another possible approach could be to do something like the proposed implementation one, but with some sparse
  representation of output data. However, I think the general assumptions we can make about the normal case make this
  less likely to be beneficial. It is quite likely that `n` is much larger than the output histograms, and that a large
  percentage of the output histogram may be occupied, even when considering dividing the input amongst multiple
  threads. This could be explored if we find temporary storage is too large for some cases and the atomic approach
  does not provide a good fallback.

## Proposal
After exploring the above implementation for `histogram`, it seems the following proposal better represents the use
cases which are important, and provides reasonable performance for most cases.

### Embarrassingly Parallel Via Temporary Histograms
This method uses temporary storage and a pair of embarrassingly parallel `parallel_for` loops to accomplish the
`histogram`.

#### OpenMP:
1) Determine the number of threads that we will use locally
2) In parallel, create and initialize temporary data for the number of threads copies of the histogram output sequence.
3) Run a `parallel_for` pattern which performs a `histogram` on the input sequence where each thread accumulates into
   its own copy of the output sequence using the temporary storage to remove any race conditions.
4) Run a second `parallel_for` over the `histogram` output sequence which accumulates all temporary copies of the
   histogram into the output histogram sequence. This step is also embarrassingly parallel.
5) Deallocate temporary storage.

#### TBB
For TBB, we can do something similar, but we can use `enumerable_thread_specific` and its member function, `local()` to
provide a lazy allocation of thread local management, which does not require querying the number of threads or getting
the index. This allows us to operate in a composable manner while keeping the same conceptual implementation.
1) Embarrassingly parallel accumulation to thread local storage
2) Embarrassingly parallel aggregate to output data

