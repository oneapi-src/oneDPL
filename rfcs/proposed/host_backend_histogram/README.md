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
`histogram` for these host-side backends. The serial implementation is straightforward and is not worth discussing in
much length here. We will add it, but there is not much to discuss within the RFC, as its implementation will be
straightforward.

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
As mentioned above, histogram looks to be a memory bandwidth-dependent algorithm. This may limit the benefit achievable
from vector instructions as they provide assistance mostly in speeding up computation. Vector operations in this case
also compound our issue of race conditions, multiplying the number of concurrent lines of execution by the vector
length. The advantage we get from vectorization of the increment operation or the lookup into the output histogram may
not provide much benefit, especially when we account for the extra memory footprint required or synchronization
required to overcome the race conditions which we add from the additional concurrent streams of execution. It may make
sense to decline to add vectorized operations within histogram depending on the implementation used, and based on
performance results.

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

## Proposal
I believe there are two competing options for `histogram`, which may both have utility in the final implementation
depending on the use case.

### Implementation One (Embarrassingly Parallel)
This method uses temporary storage and a pair of embarrassingly parallel `parallel_for` loops to accomplish the
`histogram`.

#### OpenMP:
1) Determine the number of threads that we will use locally
2) Create temporary data for the number of threads minus one copy of the histogram output sequence. Thread zero can
   use the user-provided output data.
3) Run a `parallel_for` pattern which performs a `histogram` on the input sequence where each thread accumulates into
   its own copy of the output sequence using the temporary storage to remove any race conditions.
4) Run a second `parallel_for` over the `histogram` output sequence which accumulates all temporary copies of the
   histogram into the output histogram sequence. This step is also embarrassingly parallel.
5) Deallocate temporary storage.

#### TBB
For TBB, we can do something similar, but we can use `enumerable_thread_specific` and its member function, `local()` to
provide a lazy allocation of thread local management, which does not require querying the number of threads or getting
the index. This allows us to operate in a compose-able manner while keeping the same conceptual implementation.
1) Embarassingly parallel accumulation to thread local storage
2) Embarassingly parallel aggregate to output data

I believe the challenge here may be to properly provide the heuristics to choose between this implementation and the
other implementation.  However, we should be able to have some reasonable division.

### Implementation Two (Atomics)
This method uses atomic operations to remove the race conditions during accumulation. With atomic increments of the
output histogram data, we can merely run a `parallel_for` pattern.

To deal with atomics appropriately, we have some limitations. We must either use standard library atomics, atomics
specific to a backend, or custom atomics specific to a compiler. `C++17` provides `std::atomic<T>`, however, this can
only provide atomicity for data which is created with atomics in mind. This means allocating temporary data and then
copying it to the output data. `C++20` provides `std::atomic_ref<T>` which would allow us to wrap user-provided output
data in an atomic wrapper, but we cannot assume `C++17` for all users. OpenMP provides atomic
operations, but that is only available for the OpenMP backend.  The working plan is to implement a macro like
`_ONEDPL_ATOMIC_INCREMENT(var)` which uses an `std::atomic_ref` if available , and alternatively uses compiler builtins
like `InterlockedAdd` or `__atomic_fetch_add_n`.  It needs to be investigated if we need to have any version which
needs to turn off the atomic implementation, due to lack of support by the compiler (I think this is unlikely).

It remains to be seen if atomics are worth their overhead and contention from a performance perspective and may depend
on the different approaches available.

### Selecting Between Algorithms
It may be the case that multiple aspects may provide an advantage to either algorithm one or two. Which `histogram` API
has been called, `n`, the number of output bins, and backend/atomic provider may all impact the performance trade-offs
between these two approaches. My intention is to experiment with these and be open to a heuristic to choose one or the
other based upon the circumstances if that is what the data suggests is best. The larger the number of output bins, the
better atomics should do vs redundant copies of the output.

## Alternative Approaches
* One could consider some sort of locking approach which locks mutexes for subsections of the output histogram prior to
  modifying them. It's possible such an approach could provide a similar approach to atomics, but with different
  overhead tradeoffs. It seems quite likely that this would result in more overhead, but it could be worth exploring.

* Another possible approach could be to do something like the proposed implementation one, but with some sparse
  representation of output data. However, I think the general assumptions we can make about the normal case make this
  less likely to be beneficial. It is quite likely that `n` is much larger than the output histograms, and that a large
  percentage of the output histogram may be occupied, even when considering dividing the input amongst multiple
  threads. This could be explored if we find temporary storage is too large for some cases and the atomic approach
  does not provide a good fallback.

## Open Questions
* Would it be worthwhile to add our own implementation of `atomic_ref` for C++17? I believe this would require
  specializations for each of our supported compilers.

* What is the overhead of atomics in general in this case and does the overhead there make them inherently worse than
  merely having extra copies of the histogram and accumulating?

* Is it worthwhile to have separate implementations for TBB and OpenMP because they may differ in the best-performing
  implementation? What is the best heuristic for selecting between algorithms (if one is not the clear winner)?

* How will vectorized bricks perform, and in what situations will it be advantageous to use or not use vector
  instructions?
