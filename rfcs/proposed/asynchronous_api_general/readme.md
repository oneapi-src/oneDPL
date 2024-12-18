# General architecture for asynchronous API

## Introduction

oneDPL algorithms with device execution policies are developed on top of SYCL, and since the early
days there was some demand for the algorithms to preserve the SYCL ability of computing
asynchronously to the main program running on the host CPU. However the C++ standard semantics for
parallel algorithms, which oneDPL follows, does not assume asynchronous execution, as the calling
thread can only return when the algorithm finishes (for details, see [algorithms.parallel.exec]
section of the C++ standard).

To address this demand, [experimental asynchronous algorithms](#onedpl-experimental-asynchronous-algorithms)
have been added that not block the calling thread. Then oneDPL added the experimental functionality for
[dynamic selection](https://oneapi-src.github.io/oneDPL/dynamic_selection_api_main.html) that also
allows starting asynchronous work and wait for its completion later.

For these experimental APIs to get solid and go into production, we wanted to design a single
consistent approach to asynchronous execution. That was the original goal of this RFC proposal.
However, due to new information we found while studying the topic, eventually we concluded that
such common asynchronous API is not needed.

## Context

### Thrust & CUB

The Thrust library from Nvidia uses two approaches for its asynchronous algorithms.
Both approaches are implemented for the CUDA backend only.

First, it has a small set of explicitly asynchronous algorithms in `namespace thrust::async`
that return an event or a *future* to later synchronize with. However, recently we have learned
that, according to https://github.com/NVIDIA/cccl/issues/100, this API is considered deprecated
(though yet unofficially).

Second, Thrust has a special `par_nosync` execution policy that indicates that the implementation
can skip non-essential synchronization as the caller will explicitly synchronize with the device
or stream before accessing the results.

More information can be found in the [Thrust changelog](https://nvidia.github.io/cccl/thrust/releases/changelog.html).

The device algorithms of CUB are (implicitly) asynchronous but, unlike Thrust, these do not return
anything waitable and require explicit synchronization with the device. There are notably more
`cub::Device*` algorithms than those in `thrust::async`.

### oneDPL experimental asynchronous algorithms

As mentioned before, [the asynchronous algorithms](https://oneapi-src.github.io/oneDPL/parallel_api/async_api.html)
in oneDPL are intended to allow the underlying SYCL implementation proceed without blocking
the calling thread. These functions return a future that can be used to synchronize and obtain
the computed value at a later time. The functions can also accept a list of `sycl::event` objects
as *input dependencies* (though the implementation has not advanced beyond immediate wait on these events).
The `wait_for_all` function waits for completion of a given list of events or futures.

The second goal of this API was to allow functional mapping for `thrust::async` algorithms,
facilitating support for SYCL in applications that use Thrust.

### oneDPL experimental API for kernel templates

TBD

## Reasons for archival

TBD

## Backup: The use case study

In the practical use of the oneDPL asynchronous APIs as well as similar APIs of other libraries
(such as Thrust) we observed several typical patterns, pseudocode examples of which follow.
**The proposal is aimed primarily at supporting these very patterns**. The list can be extended
if there is enough evidence of demand for other patterns of asynchronous compute.

In the examples, `foo-async` represents a call such as oneDPL `for_each_async` and `submit`
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
previously submitted to a queue.

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
pattern for processing some big work in parallel.

```
work-queue qs[] = {/*initialize all the queues*/};
sync-object jobs[];
/* split work to multiple queues */
for (work-queue q in qs) {
    sync-object s = foo-async(q, /*arguments*/);
    append s to jobs;
}
/* synchronize with all parts */
sync-with(jobs); /* possibly in a loop */

/* combine the results, if needed */
```

For example, [Distributed Ranges](https://github.com/oneapi-src/distributed-ranges) use oneDPL
asynchronous algorithms in this way to distribute work across available devices.

The fork-join pattern can also use work queue synchronization:
```
work-queue qs[] = {/*initialize all the queues*/};
/* split work to multiple queues */
for (work-queue q in qs) {
    foo-async(q, /*arguments*/);
}
/* synchronize with all queues */
for (work-queue q in qs) {
    sync-with(q);
}
/* combine the results, if needed */
```

### Out of the scope (non-goals)

There is no intention to support asynchronous computations in general nor use cases beyond the functionality
of oneDPL, such as dependencies between any asynchronously executed functions. There exist other libraries
as well as the C++ standard capacities for that purpose, some mentioned later in the document.

SYCL supports out-of-order queues and provides APIs to set dependencies between kernels,
including explicit dependencies via events. The experimental async algorithms in oneDPL were designed
to preserve this capability; however, we have no evidence of it being used in practice. Also our study of
the usage of Thrust asynchronous algorithms have not found examples of dependency chains. Therefore,
support for this use case is not a requirement.

## Backup: Asynchrony support in the C++ standard

### C++ async & future

The C++ standard provides several ways for a program to use asynchrony, but [`std::future` and
related APIs](https://en.cppreference.com/w/cpp/header/future), especially `std::async`, are of the
most interest for this discussion. The `std::async` routine runs a given function asynchronously,
returning `std::future` to obtain the result later. Essentially, this is the model that both
oneDPL and Thrust (as well as other libraries) use for their asynchronous APIs.

There is a fair amount of criticism for this model, which primarily points to the lack of support
for advanced usage scenarios, including setting graphs of dependent asynchronous tasks.
Alternative implementations of futures, for example in [stlab](https://stlab.cc/includes/stlab/concurrency/)
as well as in Thrust, try addressing some of the shortcomings.

### C++26 execution control library

The new [execution control library](https://eel.is/c++draft/exec) in C++ 26, also known as
[*schedulers/senders/receivers*](https://wg21.link/p2300), is the proposed way to improve
asynchronous programming with C++, dealing with the limitations of `std::future`. In the essence,
this library provides a language to build program flow graphs and then run those on chosen execution
resources.

The stages of creating and executing a graph of computations are separate; the execution can only
be started explicitly by one of a few dedicated calls. Therefore the approach appears more
suitable for deferred execution, while eager execution would be at least more verbose to code.

Some companion proposals, notably for [async_scope](https://wg21.link/p3149) and [system execution
context](https://wg21.link/p2079), are yet to be accepted to the working draft. The proposal for
adding [asynchronous parallel algorithms](https://wg21.link/p3300) is at a very early stage and
is not planned for C++ 26.
