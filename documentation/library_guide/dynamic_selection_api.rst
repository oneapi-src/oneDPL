Dynamic Device Selection API
############################

What is Dynamic Device Selection?
---------------------------------

Dynamic device selection ia an experimental feature in oneDPL that selects an 
*execution resource* based on a chosen *selection policy*. When used with
the experimental ``invoke`` or ``invoke_async`` functions, the selected execution 
resource is passed to the developer-provieded callable object (for example a lambda).
By default, the execution resources used in oneDPL are SYCL queues.  

A basic example
---------------

In this section, an example is provided that uses using dynamic device
selection to selects SYCL queues. The example code creates a 
selection policy, ``round_robin_policy``, object ``p``. It then queries ``p``
to discover how many devices are available on the platform. The  
``round_robin_policy`` simply rotates through the devices on the platform. 
In the example's for-loop, ``p`` is passed as the first argument to the 
``experimental::invoke`` function. The second argument is a lambda expression 
that receives the selected ``sycl::queue``. The lamba body prints out whether 
the selected queue is a cpu, a gpu or something else. The number of iterations 
of the for-loop is twice the number of devices on the platform, so we expect
that it will received a SYCL queue for each available device twice. 

.. code:: cpp

  #include <iostream>
  #include <sycl/sycl.hpp>
  #include <oneapi/dpl/dynamic_selection>
  using namespace oneapi::dpl;

  int main() {
    // choose a policy
    experimental::round_robin_policy p;

    auto N = experimental::property::query(p, 
                                          experimental::property::universe_size);
    std::cout << "By default there are " << N 
              << " queues in the universe" << std::endl;

    for (int i = 0; i < 2*N; ++i) {
      // call experimental::invoke with policy p.
      // It invokes the lambda with a selected q.
      experimental::invoke(p, [=](sycl::queue q) {
        if (q.get_device().is_cpu()) {
          std::cout << "selection #" << i << " is a cpu" << std::endl;
        } else if (q.get_device().is_gpu()) {
          std::cout << "selection #" << i << " is a gpu" << std::endl;
        } else {
          std::cout << "selection #" << i << " is something else" << std::endl;
        }   

        // you must return a sycl event
        return sycl::event{};
      }); 
    }
    return 0;
  }

This code was compiled and then run on a test platform that contains four devices: an OpenCL FPGA emulator
device, an OpenCL CPU device, an OpenCL integrated GPU device, and a level-zero GPU device.  The output
for the run is shown below and shows that each device was rotated through twice:

.. code:: cpp
  
  By default there are 4 queues in the universe
  selection #0 is something else
  selection #1 is a cpu
  selection #2 is a gpu
  selection #3 is a gpu
  selection #4 is something else
  selection #5 is a cpu
  selection #6 is a gpu
  selection #7 is a gpu

The Value of Dynamic Device Selection
-------------------------------------

The dynamic device selection API in |onedpl_short| is intended to support two general use cases. The first case is to provide 
load balancing across similar devices, such as across identical GPUs. The second case is to select an appropriate resource 
when given disparate devices, such as CPUs and GPUs. In these cases, performance portability is a concern but there are many 
cases, such as embarrassingly parallel problems, where overall application throughput is improved even if some compute 
is sent to devices that do not offer the best possible performance for the compute.

The power of dynamic device selection comes from its selection policies. At the time of this early access study, there
are only two simple policies available a static policy and a round-robin policy.

Without dynamic device selection, |onedpl_short| developers must directly choose which device to target 
by creating a dpcpp policy that is tied to a specific SYCL queue. Likewise, for most of the oneAPI performance 
libraries, developers also provides a specific queue through the library interface to select the device to use. 
When using Dynamic Device Selection, the decision of what queue to use is left to the selection policy.

Dynamic Device Selection Terminology
------------------------------------

.. csv-table::
    :header: "Term", "Description"

    "Task", "A unit of scheduled work."
    "Execution Resource", "A set of compute resources to use for executing a task."
    "Universe", "The set of choices for Execution Resources."
    "Scoring Policy", "A heuristic for selecting an appropriate Execution Resource for given a Task."
    "Scheduler", "An object that provides supporting types and functions needed to support the usae of the Execution Resources."
    "Properties", "Facilities for querying and reporting static and dynamic properties of the Execution Resources, Scoring Policies, Schedulers and Task executions."

User-Facing APIs
----------------

The main header for dynamic device selection features is ``dynamic_selection``. Including this header file 
brings in all of the algorithms, scoring policies, schedulers and property support.

.. code:: cpp

    #include <oneapi/dpl/dynamic_selection>

+++++++++
Functions
+++++++++

The following functions are part of the dynamic device selection API and are in the ``oneapi::dpl::experimental``
namespace:

* ``invoke``
* ``invoke_async``
* ``select``
* ``wait``
* ``property::query``
* ``property::report``

.. code:: cpp
  
    namespace dpl {
      namespace experimental {
        template<typename DSPolicy, typename Function, typename... Args>
        SyncType auto invoke_async(DSPolicy&& dp, Function&&f, Args&&... args);

        template<typename DSPolicy, typename Function, typename... Args>
        SyncType auto invoke(DSPolicy&& dp, Function&&f, Args&&... args);
      }
    }

``invoke_async``: Selects an Execution Resource using the provided DSPolicy and then invokes ``f(selected_resource, args...)``. 
The ``invoke_async`` function executes the user's function ``f`` synchronously, but the user's function is expected to submit work 
for asynchronous execution using the selected resource and to return a ``DSPolicy::native_sync_t`` object. The SyncType 
object returned by ``invoke_async`` wraps the native synchronization object. There is no argument that represents an event-list 
passed to invoke_async. The invocation of ``f`` happens in-line at the time of the invoke_async call. Any synchronization that 
is required can be done in the user's function ``f`` by using SyncType objects captured by ``f`` or passed as a regular 
argument in ``args``.  

``invoke``: Selects an Execution Resource using the provided DSPolicy, invokes ``f(selected_resource, args...)`` and then 
calls ``wait`` on the SyncType object returned by invoking ``f``. The SyncType object returned by ``invoke`` is guaranteed 
to be complete at the time it is returned. It is returned so that, if supported by native type, it can be converted to the 
native type and used to get a value.

.. code:: cpp
  
    namespace dpl {
      namespace experimental {
        template<typename DSPolicy, typename... Args>
        DSPolicy::selection_handle_t select(DSPolicy&& dp, Args&&... args);

        template<typename DSPolicy, typename Function, typename... Args>
        SyncType auto invoke_async(DSPolicy&& dp, typename DSPolicy::selection_handle_t e, 
                                  Function&&f, Args&&... args);

        template<typename DSPolicy, typename Function, typename... Args>
        SyncType auto invoke(DSPolicy&& dp, typename DSPolicy::selection_handle_t e, 
                            Function&&f, Args&&... args);
      }
    }

``select``: Returns an object that models SelectionHandle given a Policy and a set of arguments. The arguments may or may not 
include the function that will later be submitted.  For example, a round-robin policy does not need to know the function 
that will be executed in order to select the next Execution Resource in the round-robin order.  

``invoke_async``: In addition to the usual arguments for ``invoke_async`` this overload also receives a SelectionHandle. 
Dynamic device selection is skipped and instead the Execution Resource and PropertyHandle in the provided SelectionHandle is used. 
This API is useful for developers that do not want to do manual property reporting -- the implementation takes care of 
reporting necessary events back through the PropertyHandle.

``invoke``: In addition to the usual arguments for ``invoke`` this overload also receives a SelectionHandle. Dynamic device selection 
is skipped and instead the Execution Resource and PropertyHandle in the provided SelectionHandle is used. This API is 
useful for developers that do not want to do manual telemetry -- the implementation takes care of reporting necessary events 
back through the PropertyHandle. The SyncType object returned by ``invoke`` is guaranteed to be complete at the time it 
is returned.  It is returned so that, if supported by native type, it can be converted to the native type and used 
to get a value.

.. code:: cpp
  
    namespace dpl {
      namespace experimental {
        template<typename Handle>
        void wait(Handle&& h);

        template<typename HandleList>
        void wait(HandleList&& l);

        template<typename Policy>
        SyncTypeList get_wait_list(Policy p);
      }
    }

``wait``: Waits on the Handle.  The Handle models SyncType and could be an object returned by ``invoke_async`` or could 
be a list of handles returned by ``get_wait_list(p)``.  If ``wait`` is called on an object returned by ``invoke_async``, 
it waits on the corresponding task to complete.  If ``wait`` is called on a list, it waits for all tasks represented 
in the list to complete.

``get_wait_list``: Returns a list of Handles.  Each Handle models SyncType..  

.. code:: cpp
  
    namespace dpl {
      namespace experimental {
        namespace property {
          template<typename T, typename Property>
          auto query(T& t, const Property& prop);

          template<typename T, typename Property, typename Argument>
          auto query(T& t, const Property& prop, const Argument& arg);

          template<typename Handle, typename Property>
          auto report(Handle&& h, const Property& prop);

          template<typename Handle, typename Property, typename ValueType>
          auto report(Handle&& h, const Property& prop, const ValueType& v);
        } 
      }
    }

``query``: Receives an object on which to query a property, the property id and optionally an additional argument.
For example, ``dpl::experimental::property::universe_size`` can be queried on a Policy with no additional arguments.  

``report``: Reports the value of a property to the Handle, which models PropertyHandle. Some properties represent 
events without a value such as ``dpl::experimental::property::task_completion``, while others may require a value.

++++++++
Policies
++++++++

.. csv-table::
    :header: "Policy", "Description", "Motivation"

    "static_policy", "Always selects default resource. No dynamic decision.", "Least surprise. Equivalent to default_device_selector in SYCL."
    "round_robin_policy", "Rotates through resources in universe. Decision is independent of task and current platform state.", "Good for load balancing similar tasks across similar devices."

.. code:: cpp

    namespace dpl {
      namespace experimental {
        // policies that use the default scheduler (SYCL)
        using static_policy = policy<static_policy_impl<sycl_scheduler_t>>;
        using round_robin_policy = policy<round_robin_policy_impl<sycl_scheduler_t>>;

        // policies that require a user-specified scheduler
        template<typename S> using static_policy_t = policy<static_policy_impl<S>>;
        template<typename S> using round_robin_policy_t = policy<round_robin_policy_impl<S>>;

        // the default policy
        inline static_policy default_policy;
      }
    }

++++++++++
Schedulers
++++++++++

.. csv-table::
    :header: "Scheduler", "Native Resource", "Native Sync Object"

    "sycl_scheduler", "sycl::queue", "sycl::event"

++++++++++
Properties
++++++++++

.. csv-table::
    :header: "Property", "Type", "Target", "Reportable", "Description"

    "universe", "universe_t", "Policy", "No", "The devices in a policy's universe."
    "universe_size", "universe_size_t", "Policy", "No", "The number of devices in a policy's universe."
    "task_completion", "task_completion_t", "Policy", "Yes", "Used to communicate to Policy that a task is complete." 


