# General support for asynchronous API

## Introduction

oneDPL algorithms with device execution policies are developed on top of SYCL, and since the early
days there was some demand for the algorithms to preserve the SYCL ability of computing
asynchronously to the main program running on the host CPU. However the C++ standard semantics for
parallel algorithms, which oneDPL follows, does not assume asynchronous execution, as the calling
thread can only return when the algorithm finishes (for details, see [algorithms.parallel.exec]
section of the C++ standard).

To address this demand, experimental [asynchronous algorithms](https://oneapi-src.github.io/oneDPL/parallel_api/async_api.html)
have been added. These functions do not block the calling thread but instead return a *future* that
can be used to synchronize and obtain the computed value at a later time. These algorithms can also
accept a list of `sycl::event` objects as *input dependencies* (though the implementation has not
advanced beyond immediate wait on these events). The `wait_for_all` function waits for completion
of a given list of events or futures.

Later the experimental functionality for [dynamic selection](https://oneapi-src.github.io/oneDPL/dynamic_selection_api_main.html)
of an execution device has been added. The `submit` function there executes a user-specified
function object, which can start asynchronous work and return a *waitable* object
to synchronize later with. There is a `wait` function to wait for such an object as well.

For these experimental APIs to get solid and go into production, there is a clear need for a single
consistent approach to asynchronous execution. Defining that is the goal of this RFC proposal.

## The use cases

In the practical use of the oneDPL asynchronous APIs as well as similar APIs of other libraries
(such as Thrust) we observed several typical patterns, pseudocode examples of which follow.
In these examples, `foo-async` represents a call such as oneDPL `for_each_async` and `submit`
functions that start some asynchronous work, and `sync-with` indicates a synchronization
point with previously started asynchronous work.

### 1. Synchronize with a single call

This is the basic use case where the main program invokes a function asynchronously and later waits
for its completion via the returned object.

```
/* start asynchronous work */
sync-object s = foo-async(/*arguments*/);
/* do some other work */
...
/* synchronize */
sync-with(s);
```

 Variations of the pattern are supported by many APIs including `std::thread` and `std::async`.
 Some of the APIs do not guarantee that the execution of `foo-async` is actually done simultaneously
 with the main program; it might get *deferred* till the synchronization point. However the usage
 of oneDPL asynchronous APIs, and specifically those that work with or on top of SYCL, likely
 assumes that the execution is *eager*, not deferred.

### 2. Synchronize with a work queue

This pattern is specifically common for *heterogeneous* APIs that offload the execution to
another compute device such as a GPU. Usually devices are represented by *queues* or *streams*
where the work can be submitted to. The main program then waits for completion of all work
previoulsy submitted to a queue.

```
work-queue q{/*initialization arguments*/};
/* submit work to the queue* /
foo-async(q, /*arguments*/);
bar-async(q, /*arguments*/);
...
/* synchronize */
sync-with(q);
```

The work queue is shown explicitly in this pseudocode example, but in real APIs it might be
implicit as well if a default device is assumed. Note also that, unlike the first example,
asynchronous calls are not expected to return a synchronization object; it is excessive because
the synchronization is done once with all the work in the queue.

It is important to mention that work queues are usually provided by the "core" heterogeneous
programming model such as CUDA or SYCL, and it is common for a program to use one queue
with several libraries as well as custom-written kernels.

### 3. Fork and join

We also observed programs using asynchronous algorithms in a very common *fork-join* parallel
pattern for processing some big work in parallel on several execution devices.

```
work-queue qs[] = {/*initialize all the queues*/};
sync-object jobs[];
/* split work to multiple queues */
for (work-queue q in qs) {
    sync-object s = foo-async(q, /*arguments*/);
    append s to jobs;
}
/* synchronize with all parts */
sync-with(jobs);
/* combine the results, if needed */
```

For example, [Distributed Ranges](https://github.com/oneapi-src/distributed-ranges) use oneDPL
asynchronous algorithms in such a way.

## Existing approaches

### Thrust

### C++ async & future

### C++26 execution control library

## Proposal

TODO: Replace the text in this section with a full and detailed description of the proposal.
It is expected to have:

- The proposed API such as class definitions and function declarations.
- Coverage of the described use cases.
- Alternatives that were considered, along with their pros and cons.

## Open Questions

TODO: List any questions that are not sufficiently elaborated in the proposal,
need more discussion or prototyping experience, etc.
