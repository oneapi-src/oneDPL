set(PASS_SOURCES
  all_of.pass.cpp;
  any_of.pass.cpp;
  none_of.pass.cpp;
  for_each.pass.cpp;
  find.pass.cpp;
  find_end.pass.cpp;
  find_first_of.pass.cpp;
  find_if.pass.cpp;
  adjacent_find.pass.cpp;
  count.pass.cpp;
  mismatch.pass.cpp;
  equal.pass.cpp;
  algorithm.pass.cpp;
  search.pass.cpp;
)

# TODO add more tests

foreach(test_source ${PASS_SOURCES})
  add_pass_test(${test_source})
endforeach(test_source ${PASS_SOURCES})
