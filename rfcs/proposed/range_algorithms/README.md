# Support the second portion of the oneDPL Range APIs

## Introduction
Based on statistics (observing C++ code within github.com) for the usage of popular algorithms, the following
range-based APIs are suggested to be implemented next in oneDPL.
`fill`, `generate`, `move`, `replace`, `replace_if`, `remove`, `remove_if`, `mismatch`, `minmax_element`, `minmax`,
`min`, `max`, `find_first_of`, `find_end`, `is_sorted_until`

## Motivations
The feature is proposed as the next step of range-based API support for oneDPL.

### Key Requirements
- The range-based signatures for the mentioned API should correspond to the proposal for C++ parallel range algorithms, P3179.
(https://wg21.link/p3179)
- The proposed implementation should support all oneDPL execution policies: `seq`, `unseq`, `par`, `par_unseq`, and `device_policy`.

### Implementation proposal
- The implementation is supposed to rely on existing range-based or iterator-based algorithm patterns, which are already
implemented in oneDPL.
- Several algorithms described in P3179 have slightly different semantics. To implement these, some existing algorithm patterns
might require modifications or new versions.

### Test coverage

- It should be called with both small and large data sizes and with all the policies mentioned above.
- Output data, return type, and value should be checked/compared with the reference result
computed by the corresponding serial std::ranges algorithm or by a custom implemented serial version
in case of different semantics.
