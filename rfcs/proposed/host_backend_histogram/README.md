# Host Backends Support for the Histogram APIs

## Introduction
The oneDPL library added histogram APIs, currently implemented only for device policies with the DPC++ backend. These APIs are defined in the oneAPI Specification 1.4. Please see the
[oneAPI Specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneDPL/source/parallel_api/algorithms.rst#parallel-algorithms)
for the details. The host-side backends (serial, TBB, OpenMP) are not yet supported. This RFC proposes extending histogram support to these backends.

## Motivations
There are many cases to use a host-side serial or a host-side implementation of histogram. Another motivation for adding the support is simply to be spec compliant with the oneAPI specification.

## Design Considerations

### Key Requirements
Provide support for the `histogram` APIs with the following policies and backends:
- Policies: `seq`, `unseq`, `par`, `par_unseq`
- Backends: `serial`, `tbb`, `openmp`

Users have a choice of execution policies when calling oneDPL APIs. They also have a number of options of backends
which they can select from when using oneDPL. It is important that all combinations of these options have support for
the `histogram` APIs.

### Performance
With little computation, a histogram algorithm is likely a memory-bound algorithm. So, the implementation prioritize
reducing memory accesses and minimizing temporary memory traffic.

### Memory Footprint
There are no guidelines here from the standard library as this is an extension API. Still, we will minimize memory
footprint where possible.

### Code Reuse
It is a priority to reuse as much as we can which already exists and has been reviewed within oneDPL. We want to
minimize adding requirements for parallel backends to implement, and lift as much as possible to the algorithm
implementation level. We should be able to avoid adding a `__parallel_histogram` call in the individual backends, and
instead rely upon `__parallel_for`.

### unseq Backend
Currently oneDPL relies upon openMP SIMD to provide its vectorization, which is designed to provide vectorization across
loop iterations. OneDPL does not directly use any intrinsics which may offer more complex functionality than what is
provided by OpenMP.

There are a few parts of the histogram algorithm to consider. For the calculation to determine which bin to increment
there are two APIs, even and custom range which have significantly different methods to determine the bin to
increment. For the even bin API, the calculations to determine selected bin have some opportunity for vectorization as
each input has the same mathematical operations applied to each. However, for the custom range API, each input element
uses a binary search through a list of bin boundaries to determine the selected bin. This operation will have a
different length and control flow based upon each input element and will be very difficult to vectorize.

Next, lets consider the increment operation itself. This operation increments a data dependant bin location, and may
result in conflicts between elements of the same vector. This increment operation therefore is unvectorizable without
more complex handling. Some hardware does implement SIMD conflict detection via specific intrinsics, but this is not
available via OpenMP SIMD. Alternatively, we can multiply our number of temporary histogram copies by a factor of the
vector width, but it is unclear if it is worth the overhead. OpenMP SIMD provides an `ordered` structured block which
we can use to exempt the increment from SIMD operations as well.  However, this often results in vectorization being
refused by the compiler. Initial implementation will avoid vectorization of this main histogram loop.

Last, for our below proposed implementation there is the task of combining temporary histogram data into the global
output histogram. This is directly vectorizable via our existing brick_walk implementation, and will be vectorized when
a vector policy is used.

### Serial Backend
We plan to support a serial backend for histogram APIs in addition to openMP and TBB. This backend will handle all
policies types, but always provide a serial unvectorized implementation. To make this backend compatible with the other
approaches, we will use a single temporary histogram copy, which then is copied to the final global histogram.

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
After exploring the above implementation for `histogram`, the following proposal better represents the use
cases which are important, and provides reasonable performance for most cases.

### Embarrassingly Parallel Via Temporary Histograms
This method uses temporary storage and a pair of embarrassingly parallel `parallel_for` loops to accomplish the
`histogram`.

Create a generic `__thread_enumerable_storage` struct which will be defined by all parallel backends, which provides
the following:
* constructor which specifies the storage to be held per thread and a method to initialize it
* `get()` returns an iterator to the beginning of the current thread's temporary vector
* `get_with_id(int i)` returns an iterator to the beginning of temporary vector with index provided
* `size()` returns number of temporary arrays

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

