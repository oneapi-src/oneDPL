# Passed Directly Customiation Point for User Defined Types

## Introduction

OneDPL handles some types of input data automatically as input to its dpcpp (sycl-based) backend as described
[here](https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html). Unified Shared Memory (USM)
pointers refer to data which is device accessible inherently, so no processing is required to pass this type of input
data to SYCL kernels, we refer to this trait as "passed directly". OneDPL also defines some rules for its provided
[iterator types](https://uxlfoundation.github.io/oneDPL/parallel_api/iterators.html) to be passed directly to SYCL
under some circumstances (based usually on their base types).

Internally, these rules are defined with a trait `oneapi::dpl::__ranges::is_passed_directly<T>` which evaluates to
`std::true_type` or `std::false_type` to indicate whether the type `T` should be passed directly to sycl kernels.
There exists a unofficial legacy `is_passed_directly` trait which types can define like this:
`using is_passed_directly = std::true_type;` which is supported within oneDPL. This method is currently used for a
number of helper types within the SYCLomatic compatability headers, (`device_pointer`, `device_iterator`,
`tagged_pointer`, `constant_iterator`, `iterator_adaptor`). There is no official public API for users who want to
create their own types which could be passed directly to SYCL kernels, this is a gap we should fill in with an official
public API.

Without something like this users are forced to only rely upon our provided types, or reach into implementation details
which are not part of oneDPL's specified interface.

## Proposal

Create a customization point `oneapi::dpl::is_passed_directly_to_sycl_kernels` free function which allows users to
define to mark their types as passed directly:

```
template <typename T>
constexpr bool is_passed_directly_to_sycl_kernels(const T&);
```

oneDPL will provide a default implementation which will defer to the existing trait:

```
template <typename T>
constexpr
bool
is_passed_directly_to_sycl_kernels(const T&)
{
	return oneapi::dpl::__ranges::is_passed_directly_v<T>;
}
```

Below is a simple example of a type and customization point definition which is always passed directly.

```
namespace user
{

    struct my_passed_directly_type
    {
        /* unspecified user definition */
    };

    template <typename It1, typename It2>
    constexpr
    bool
    is_passed_directly_to_sycl_kernels(const my_passed_directly_type&)
    {
        return true;
    }
} //namespace user
```

Users can use any constexpr logic based on their type to determine if the type can be passed directly into a SYCL kernel
without any processing. Below is an example of a type which contains a pair of iterators, and should be treated as
passed directly if and only if both base iterators are also passed directly. OneDPL will use this customization point
internally when determining how to handle incoming data, picking up any user customizations in the process.

```
namespace user
{
    template <typename It1, typename It2>
    struct iterator_pair 
    {
        It1 first;
        It2 second;
    };

    template <typename It1, typename It2>
    constexpr
    bool
    is_passed_directly_to_sycl_kernels(const iterator_pair<It1, It2>& pair)
    {
        return oneapi::dpl::is_passed_directly_to_sycl_kernels(pair.first) &&
               oneapi::dpl::is_passed_directly_to_sycl_kernels(pair.second);
    }
} //namespace user
```

This allows the user to provide rules for their types next to their implementation, without cluttering the
implementation of the type itself with extra typedefs, etc.

This option can exist in concert with existing methods, the legacy `is_passed_directly` typedef in types, the internal
`oneapi::dpl::__ranges::is_passed_directly` trait specializations. It would be possible to simplify the internal
implementation away from explicit specializations of the trait to the customization point, but that is not required
at first implementation.

### Implementation details
To make this robust, we will follow an C++17 updated version of what is discussed in
[Eric Niebler's Post](https://ericniebler.com/2014/10/21/customization-point-design-in-c11-and-beyond/), using a
callable, and using an `inline constexpr` to avoid issues with ODR and to avoid issues with resolving customization
points when not separating the call to two steps with a `using` statement first.

## Alternatives considered
### Public trait struct explicit specialization
We could simply make public our internal structure `oneapi::dpl::__ranges::is_passed_directly` as
`oneapi::dpl::is_passed_directly` for users to specialize to define rules for their types. This would be a similar
mechanism to `sycl::is_device_copyable`. The implementation details of this option should avoid some complexity required
to properly implement the customization point.

However, as we have learned from experience within oneDPL, explicit specialization of a structure in another library's
namespace makes for maintenance problems. It either requires lots of closing of nested namespaces, opening of the
external library's namespace for the specialization or it requires separating these specializations to a separate
location removed from the types they are specializing for. OneDPL has chosen to use the later, which can be seen in
`include/oneapi/dpl/pstl/hetero/dpcpp/sycl_traits.h`. This has made for several errors where changes to structures
should have included changes to sycl_traits, but did not, and needed to be fixed later.

In an effort to avoid this same issue for our users, we propose a similar method but instead with a constexpr
customization point, allowing the user to override that customization point within their own namespace as a free
function.

### Require specifically named typedef / using in user's type
We could simply make official our requirements for user's types to include a typedef or using statement to define if the
type is passed directly like `using is_passed_directly = std::true_type;`, where the absence of this would be equivalent
to a `std::false_type`. 

However, this clutters the user type definitions with specifics of oneDPL. It also may not be as clear what this
signifies for maintenance of user code without appropriate comments describing the details of oneDPL and SYCL. Users
have expressed that this is undesirable.

### Testing
We will need a detailed test checking both positive and negative responses to `is_passed_directly_to_sycl_kernels` come
as expected, with custom types and combinations of iterators, usm pointers etc.

## Open Questions

Is there a better / more concise name than `is_passed_directly_to_sycl_kernels` we can use which properly conveys the
meaning to the users?

Should we be targeting Experimental or fully supported with this proposal?
 (Do we think user feedback is required to solidify an interface / experience?)