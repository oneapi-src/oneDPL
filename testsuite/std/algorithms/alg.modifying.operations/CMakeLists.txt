set(PASS_SOURCES
  copy.pass.cpp;
  move.pass.cpp;
  replace.pass.cpp;
  replace_copy.pass.cpp;
  fill.pass.cpp;
  generate.pass.cpp;
  remove.pass.cpp;
  remove_copy.pass.cpp;
  unique_copy_equal.pass.cpp;
  transform_unary.pass.cpp;
  transform_binary.pass.cpp
)

# TODO add more tests

foreach(test_source ${PASS_SOURCES})
  add_pass_test(${test_source})
endforeach(test_source ${PASS_SOURCES})
