ADD_EXECUTABLE(gamma_correction gamma_correction.cpp utils.cpp)

IF(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(gamma_correction rt ${PROJECT_NAME})
ELSE(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(gamma_correction ${PROJECT_NAME})
ENDIF(UNIX AND NOT APPLE)
