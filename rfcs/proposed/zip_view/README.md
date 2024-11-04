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
`oneapi::dpl::ranges::zip_view` should have:
- API compliance with `std::ranges::zip_view`
- `oneapi::dpl::__internal::tuple` underhood, due to `std::tuple` has issues with swapability and device copyability.


### Performance
Combining `std::ranges::zip_view` with data pipelines and kernel fusion enables developers to write expressive,
efficient code for processing multiple related datasets. This approach not only simplifies the logic but also
optimizes performance, making it an essential technique in modern C++ development. Whether you're working with
simple transformations or complex data processing workflows, zip_view can be a valuable utility.
