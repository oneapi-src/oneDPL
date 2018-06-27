set(PASS_SOURCES
  sort.pass.cpp;
  partial_sort.pass.cpp;
  is_sorted.pass.cpp
)

# TODO add more tests

foreach(test_source ${PASS_SOURCES})
  add_pass_test(${test_source})
endforeach(test_source ${PASS_SOURCES})
