# Implementation design proposal

## Introduction
According to proposed RFC document about `std::ranges::zip_view` a following implementation design is proposed here.

## Implementation details
- `oneapi::dpl::ranges::zip_view` is designed as C++ class, which represents a range adaptor (see C++ Range Library).
This class encapsulate a tuple-like type to keep a combination of two or more ranges.
- The implementation provides all necessary operators to satisfy 'random accessible view' requirement.
- To provide a device copyability requirement `oneapi::dpl::__internal::tuple` is proposed as tuple-like type underhood.
- To provide a value-swappable requirement `oneapi::dpl::__internal::tuple` is proposed as a dereferenced value for
`oneapi::dpl::ranges::zip_view::iterator`, due to the standard `std::tuple` C++20 is not swappable type.
- Usage of C++ concepts is desirable to write type requirements for types, methods and members of the class.
- C++20 is minimum supported version for the class. It allows to use modern C++ things like concepts and others.

### Test coverage

- `oneapi::dpl::ranges::zip_view` is tested itself, base functionality.
- should be tested with at least one oneDPL range based algorithm.
- should be tested with at least one oneDPL iterator based algorithm.

