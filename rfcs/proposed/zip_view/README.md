# zip_view Support for the oneDPL Range APIs with C++20

## Introduction
`std::ranges::zip_view` is powerful utility enables developers to combine two or more ranges into a single view,
where each element is represented as a tuple containing corresponding elements from each input range.

## Motivations
`std::ranges::zip_view` is a convenient way to combine multiple ranges into a single view, where each element of
the resulting range is a tuple containing one element from each of the input ranges. This can be particularly
useful for iterating over multiple collections in parallel. `std::ranges::zip_view` is introduced starting with C++23,
but there are many users who work with C++20 standard yet. So, oneDPL introduces `oneapi::dpl::ranges::zip_view`,
which the same API and functionality as `std::ranges::zip_view`.

### Key Requirements
`oneapi::dpl::ranges::zip_view` should be:
- compilable with C++20 version (minimum)
- API compliant with `std::ranges::zip_view`
- random accessable view; the "underlying" views also should be random accessable
- in case of a device usage: a device copyable view itself and the "underlying" views also should be device copyable
  
`oneapi::dpl::ranges::zip_view::iterator` should be:
- value-swappable (https://en.cppreference.com/w/cpp/named_req/ValueSwappable)
- convertible `oneapi::dpl::zip_iterator`

### Performance
TBD
