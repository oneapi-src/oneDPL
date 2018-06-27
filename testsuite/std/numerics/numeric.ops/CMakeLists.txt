set(PASS_SOURCES
  reduce.pass.cpp
  transform_reduce.pass.cpp
  scan.pass.cpp
  transform_scan.pass.cpp
)

# TODO add more tests

foreach(test_source ${PASS_SOURCES})
  add_pass_test(${test_source})
endforeach(test_source ${PASS_SOURCES})
