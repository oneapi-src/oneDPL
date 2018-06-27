add_subdirectory(numeric.ops)

set(PASS_SOURCES
  numeric.pass.cpp
)

# TODO add more tests

foreach(test_source ${PASS_SOURCES})
  add_pass_test(${test_source})
endforeach(test_source ${PASS_SOURCES})
