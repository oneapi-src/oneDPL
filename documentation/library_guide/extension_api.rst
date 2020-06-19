Extension API
################################
A list of the Extension API algorithms and functional utility classes.

The Extension API currently includes six algorithms and three functional utility classes. The algorithms include segmented reduce, segmented scan and vectorized search algorithms. Detailed descriptions of each algorithm are provided below.


- reduce_by_segment

    The ``reduce_by_segment`` algorithm performs partial reductions on a sequence's values and keys. Each reduction is computed with a given reduction operation for a contiguous subsequence of values, which are determined by keys being equal according to a predicate. A return value is a pair of iterators holding the end of the output sequences for keys and values.

    For correct computation, the reduction operation should be associative. If no operation is specified, the default operation for the reduction is ``std::plus``, and the default predicate is ``std::equal_to``. The algorithm requires that the type of the elements used for values be default constructible.

    Example::

        keys:   [0,0,0,1,1,1]
        values: [1,2,3,4,5,6]
        output_keys:   [0,1]
        output_values: [1+2+3=6,4+5+6=15]

- inclusive_scan_by_segment

    The ``inclusive_scan_by_segment`` algorithm performs partial prefix scans on a sequence's values. Each scan applies to a contiguous subsequence of values, which are determined by the keys associated with the values being equal. The return value is an iterator targeting the end of the result sequence.

    For correct computation, the prefix scan operation should be associative. If no operation is specified, the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. The algorithm requires that the type of the elements used for values be default constructible.

    Example::

        keys:   [0,0,0,1,1,1]
        values: [1,2,3,4,5,6]
        result: [1,1+2=3,1+2+3=6,4,4+5=9,4+5+6=15]

- exclusive_scan_by_segment

    The ``exclusive_scan_by_segment`` algorithm performs partial prefix scans on a sequence's values. Each scan applies to a contiguous subsequence of values that are determined by the keys associated with the values being equal, and sets the first element to the initial value provided. The return value is an iterator targeting the end of the result sequence.

    For correct computation, the prefix scan operation should be associative. If no operation is specified, the default operation is ``std::plus``, and the default predicate is ``std::equal_to``.

    Example::

        keys:   [0,0,0,1,1,1]
        values: [1,2,3,4,5,6]
        initial value: [0]
        result: [0,0+1=1,0+1+2=3,0,0+4=4,0+4+5=9]

- binary_search

    The ``binary_search`` algorithm performs a binary search of the input sequence for each of the values in the search sequence provided.  The result of a search for the *i-th* element of the search sequence, a boolean value indicating whether the search value was found in the input sequence, is assigned to the *i-th* element of the result sequence. The algorithm returns an iterator that points to one past the last element of the result sequence that was assigned a result. The algorithm assumes the input sequence has been sorted by the comparator provided. If no comparator is provided then a function object that uses ``operator<`` to compare the elements will be used.

    Example::

        input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
        search sequence: [0, 2, 4, 7, 6]
        result sequence: [true, true, false, false, true]

- lower_bound

    The ``lower_bound`` algorithm performs a binary search of the input sequence for each of the values in the search sequence provided to identify the lowest index in the input sequence where the search value could be inserted without violating the ordering provided by the comparator used to sort the input sequence.  The result of a search for the *i-th* element of the search sequence, the first index in the input sequence where the search value could be inserted without violating the ordering of the input sequence, is assigned to the *i-th* element of the result sequence. The algorithm returns an iterator that points to one past the last element of the result sequence that was assigned a result. If no comparator is provided then a function object that uses ``operator<`` to compare the elements will be used.

    Example::

        input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
        search sequence: [0, 2, 4, 7, 6]
        result sequence: [0, 1, 8, 10, 8]


- upper_bound

    The ``upper_bound`` algorithm performs a binary search of the input sequence for each of the values in the search sequence provided to identify the highest index in the input sequence where the search value could be inserted without violating the ordering provided by the comparator used to sort the input sequence.  The result of a search for the *i-th* element of the search sequence, the last index in the input sequence where the search value could be inserted without violating the ordering of the input sequence, is assigned to the *i-th* element of the result sequence. The algorithm returns an iterator that points to one past the last element of the result sequence that was assigned a result. If no comparator is provided then a function object that uses ``operator<`` to compare the elements will be used.

    Example::

        input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
        search sequence: [0, 2, 4, 7, 6]
        result sequence: [1, 4, 8, 10, 10]

Here are the details for the functional utility classes:

- identity: A C++11 implementation of the C++20 ``std::identity`` function object type, where the operator() returns the argument unchanged.
- minimum: A function object type where the operator() applies ``std::less`` to its arguments, then returns the lesser argument unchanged.
- maximum: A function object type where the operator() applies ``std::greater`` to its arguments, then returns the greater argument unchanged.
