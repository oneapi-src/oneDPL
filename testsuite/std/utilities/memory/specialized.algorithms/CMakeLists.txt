set(PASS_SOURCES
  uninitialized_copy_move.pass.cpp
  uninitialized_fill_destroy.pass.cpp
  uninitialized_construct.pass.cpp
)

# TODO add more tests

foreach(test_source ${PASS_SOURCES})
  add_pass_test(${test_source})
endforeach(test_source ${PASS_SOURCES})
