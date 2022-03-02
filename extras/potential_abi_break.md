Potential ABI breaking changes
------------------------------

ABI breaking changes that might happen in the future are listed in the table below

| API                            | Change                           | Comment                                                             |
| :---:                          | :----:                           | :---:                                                               |
| discard_block_engine           | `int` field to `std::size_t`     | No demand so far. `static_assert` to check overflow at compile-time |