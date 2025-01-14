# Passed Directly Customization Point for User Defined Types

## Introduction

oneDPL handles some types of input data automatically as input to its dpcpp (sycl-based) backend as described
[here](https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html). Unified Shared Memory (USM)
pointers refer to data which is device accessible inherently, so no processing is required to pass this type of input
data to SYCL kernels, we refer to this trait as "passed directly". oneDPL also defines some rules for its provided
[iterator types](https://uxlfoundation.github.io/oneDPL/parallel_api/iterators.html) to be passed directly to SYCL
under some circumstances (based usually on their base types).

Internally, these rules are defined with a trait `oneapi::dpl::__ranges::is_passed_directly<T>` which evaluates to
`std::true_type` or `std::false_type` to indicate whether the iterator type `T` should be passed directly to sycl
kernels. There exists an unofficial legacy `is_passed_directly` trait which types can define like this:
`using is_passed_directly = std::true_type;` which is supported within oneDPL. This method is currently used for a
number of helper types within the SYCLomatic compatibility headers, (`device_pointer`, `device_iterator`,
`tagged_pointer`, `constant_iterator`, `iterator_adaptor`). There is no official public API for users who want to
create their own iterator types which could be passed directly to SYCL kernels, this is a gap that should be filled 
with an official public API.

Without something like this users are forced to only rely upon our provided iterator types, or reach into implementation
details which are not part of oneDPL's specified interface.

## Proposal

Create a customization point `oneapi::dpl::is_passed_directly_in_onedpl_device_policies` free function which allows
users to mark their types as passed directly:

```
template <typename T>
constexpr bool is_passed_directly_in_onedpl_device_policies(const T&);
```

oneDPL will provide a default implementation which will defer to the existing trait:

```
template <typename T>
constexpr
bool
is_passed_directly_in_onedpl_device_policies(const T&)
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

    constexpr
    bool
    is_passed_directly_in_onedpl_device_policies(const my_passed_directly_type&)
    {
        return true;
    }
} //namespace user
```

Users can use any constexpr logic based on their type to determine if the type can be passed directly into a SYCL kernel
without any processing. Below is an example of a type which contains a pair of iterators, and should be treated as
passed directly if and only if both base iterators are also passed directly. oneDPL will use this customization point
internally when determining how to handle incoming data, picking up any user defined customizations.

When using device policies, oneDPL will run compile time checks on argument iterator types by calling
`is_passed_directly_in_onedpl_device_policies` as a `constexpr`. If `true` is returned, oneDPL will pass the iterator
directly to sycl kernels rather than copying the data into sycl buffers and using accessors to those buffers in the
kernel. Users may also call `oneapi::dpl::is_passed_directly_in_onedpl_device_policies` themselves to check how the
oneDPL internals will treat any iterator types. This may be useful to ensure that no extra overhead occurs in device
policy calls.

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
    is_passed_directly_in_onedpl_device_policies(const iterator_pair<It1, It2>& pair)
    {
        return oneapi::dpl::is_passed_directly_in_onedpl_device_policies(pair.first) &&
               oneapi::dpl::is_passed_directly_in_onedpl_device_policies(pair.second);
    }
} //namespace user
```

This allows the user to provide rules for their types next to their implementation, without cluttering the
implementation of the type itself with extra typedefs, etc.

This option can exist in concert with existing methods, the legacy `is_passed_directly` typedef in types, the internal
`oneapi::dpl::__ranges::is_passed_directly` trait specializations. It would be possible to simplify the internal
implementation away from explicit specializations of the trait to the customization point, but that is not required
at first implementation.

`oneapi::dpl::is_passed_directly_in_onedpl_device_policies()` will be defined in `oneapi/dpl/execution`. It must be
included prior to calling or overriding `oneapi::dpl::is_passed_directly_in_onedpl_device_policies()` with their own
customizations.

### Implementation Details
We will follow a C++17 updated version of what is discussed in
[Eric Niebler's Post](https://ericniebler.com/2014/10/21/customization-point-design-in-c11-and-beyond/). Using his
proposed method will allow unqualified calls to `is_passed_directly_in_onedpl_device_policies()` after a
`using oneapi::dpl::is_passed_directly_in_onedpl_device_policies;` statement, as well as qualified calls to
`oneapi::dpl::is_passed_directly_in_onedpl_device_policies()` to find the default implementation provided by oneDPL.
Both options will also have access to any user defined customizations defined in the same namespace of the type.
With access to c++17, we will use `inline constexpr` to avoid issues with ODR, rather than his described method.

### Drawbacks
#### Unavailable For SFINAE
While `is_passed_directly_in_onedpl_device_policies` is defined to be `constexpr`, and all user customizations must also
be `constexpr`, they will be unavailable for `std::enable_if` or other SFINAE checks. These checks only have access to
the type of the template parameter, and do not have access to any named instance of that type. Therefore, without
imposing a requirement like default constructibility on types, we cannot use
`is_passed_directly_in_onedpl_device_policies` in this context, as we have no instance to use as the argument to our
Argument Dependant Lookup (ADL) function. This is an inconvenience, and it will require some refactoring of the code
which processes input sequences, but it should only impact internal usage of
`is_passed_directly_in_onedpl_device_policies`.  I don't anticipate users wanting to incorporate this function into
their own SFINAE checks. Alternatives below do not have such drawbacks, but I still believe this to be the superior
option for users.

## Alternatives Considered
### Public Trait Struct Explicit Specialization
oneDPL could make public our internal structure `oneapi::dpl::__ranges::is_passed_directly` as
`oneapi::dpl::is_passed_directly` for users to specialize to define rules for their types. This would be a similar
mechanism to `sycl::is_device_copyable`. The implementation details of this option should avoid some complexity required
to properly implement the customization point.

However, as we have learned from experience within oneDPL, explicit specialization of a structure in another library's
namespace makes for maintenance problems. It either requires lots of closing of nested namespaces, opening of the
external library's namespace for the specialization or it requires separating these specializations to a separate
location removed from the types they are specializing for. oneDPL has chosen to use the later, which can be seen in
`include/oneapi/dpl/pstl/hetero/dpcpp/sycl_traits.h`. This has made for several errors where changes to structures
should have included changes to sycl_traits, but did not, and needed to be fixed later.

In an effort to avoid this same issue for our users, we propose a similar method but instead with a constexpr
customization point, allowing the user to override that customization point within their own namespace as a free
function.

### Require Specifically Named Typedef / Using in User's Type
oneDPL could make official our requirements for user's types to include a typedef or using statement to define if the
type is passed directly like `using is_passed_directly = std::true_type;`, where the absence of this would be equivalent
to a `std::false_type`. 

However, this clutters the user type definitions with specifics of oneDPL. It also may not be as clear what this
signifies for maintenance of user code without appropriate comments describing the details of oneDPL and SYCL. Users
have expressed that this is undesirable.

### Wrapper Class
oneDPL could provide some wrapper iterator `direct_iterator` which wraps an arbitrary base iterator and marks it as
passed directly. `direct_iterator` could utilize either of the above alternatives to accomplish this, and signal
that the iterator should be passed directly. It would need to pass through all operations to the wrapped base iterator,
and make sure no overhead is added in its usage.
There is some complexity in adding such a wrapper iterator, and it would need to be considered carefully to make sure no
problems would be introduced. This wrapper class may obfuscate users types, and make them more unwieldy to use. It is
also less expressive than the other options in that it only has the ability to unilaterally mark a type as passed
directly.  There is no logic that can be used to express some iterator type which may be conditionally passed directly,
other than to have logic to conditionally apply the wrapper in the first place. This option seems less clear and has
more opportunity to cause problems.

## Testing
We will need a detailed test checking both positive and negative responses to
`is_passed_directly_in_onedpl_device_policies` have the expected result, with custom types and combinations of
iterators, usm pointers etc.

## Open Questions

* Is there a better, more concise name than `is_passed_directly_in_onedpl_device_policies` that properly conveys the
meaning to the users?
* Should we be targeting Experimental or fully Supported with this proposal?
(Do we think user feedback is required to solidify an interface / experience?)
