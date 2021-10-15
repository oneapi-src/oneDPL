Additional Algorithms
######################

The definitions of the algorithms listed below are available through the ``oneapi/dpl/algorithm``
header.  All algorithms are implemented in the ``oneapi::dpl`` namespace.

* ``reduce_by_segment``: performs partial reductions on a sequence's values and keys. Each
  reduction is computed with a given reduction operation for a contiguous subsequence of values, which are
  determined by keys being equal according to a predicate. A return value is a pair of iterators holding
  the end of the output sequences for keys and values.

  For correct computation, the reduction operation should be associative. If no operation is specified,
  the default operation for the reduction is ``std::plus``, and the default predicate is ``std::equal_to``.
  The algorithm requires that the type of the elements used for values be default constructible. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    output_keys:   [0,1]
    output_values: [1+2+3=6,4+5+6=15]

* ``inclusive_scan_by_segment``: performs partial prefix scans on a sequence's values. Each
  scan applies to a contiguous subsequence of values, which are determined by the keys associated with the
  values being equal. The return value is an iterator targeting the end of the result sequence.

  For correct computation, the prefix scan operation should be associative. If no operation is specified,
  the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. The algorithm
  requires that the type of the elements used for values be default constructible. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    result: [1,1+2=3,1+2+3=6,4,4+5=9,4+5+6=15]

* ``exclusive_scan_by_segment``: performs partial prefix scans on a sequence's values. Each
  scan applies to a contiguous subsequence of values that are determined by the keys associated with the values
  being equal, and sets the first element to the initial value provided. The return value is an iterator
  targeting the end of the result sequence.

  For correct computation, the prefix scan operation should be associative. If no operation is specified,
  the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    initial value: [0]
    result: [0,0+1=1,0+1+2=3,0,0+4=4,0+4+5=9]

* ``binary_search``: performs a binary search of the input sequence for each of the values in
  the search sequence provided.  For each element of the search sequence the algorithm writes a boolean value
  to the result sequence that indicates whether the search value was found in the input sequence. An iterator
  to one past the last value in the result sequence is returned. The algorithm assumes the input sequence has
  been sorted by the comparator provided. If no comparator is provided, then a function object that uses
  ``operator<`` to compare the elements is used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [true, true, false, false, true]

* ``lower_bound``: performs a binary search of the input sequence for each of the values in
  the search sequence provided to identify the lowest index in the input sequence where the search value could
  be inserted without violating the sorted ordering of the input sequence.  The lowest index for each search
  value is written to the result sequence, and the algorithm returns an iterator to one past the last value
  written to the result sequence. If no comparator is provided, then a function object that uses ``operator<``
  to compare the elements is used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [0, 1, 8, 10, 8]

* ``upper_bound``: performs a binary search of the input sequence for each of the values in
  the search sequence provided to identify the highest index in the input sequence where the search value could
  be inserted without violating the sorted ordering of the input sequence.  The highest index for each search
  value is written to the result sequence, and the algorithm returns an iterator to one past the last value
  written to the result sequence. If no comparator is provided, then a function object that uses ``operator<``
  to compare the elements is used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [1, 4, 8, 10, 10]