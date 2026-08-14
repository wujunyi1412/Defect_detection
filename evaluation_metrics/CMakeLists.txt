cmake_minimum_required(VERSION 3.16)

add_library(evaluation_metrics_core STATIC src/evaluation_metrics.cpp)
target_include_directories(evaluation_metrics_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(evaluation_metrics_core PUBLIC cxx_std_17)
set_target_properties(evaluation_metrics_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(evaluation_metrics SHARED src/evaluation_metrics_c_api.cpp)

target_include_directories(evaluation_metrics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(evaluation_metrics PUBLIC cxx_std_17)
target_compile_definitions(evaluation_metrics PRIVATE EVALUATION_METRICS_EXPORTS)
target_link_libraries(evaluation_metrics PRIVATE evaluation_metrics_core)

if(MSVC)
    target_compile_options(evaluation_metrics_core PRIVATE /utf-8 /W4)
    target_compile_options(evaluation_metrics PRIVATE /utf-8 /W4)
else()
    target_compile_options(evaluation_metrics_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(evaluation_metrics PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(evaluation_metrics PROPERTIES
    OUTPUT_NAME "evaluation_metrics"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

option(BUILD_EVALUATION_METRICS_TESTS "Build evaluation metrics unit tests" ON)
if(BUILD_EVALUATION_METRICS_TESTS)
    add_executable(evaluation_metrics_tests tests/evaluation_metrics_tests.cpp)
    target_link_libraries(evaluation_metrics_tests PRIVATE evaluation_metrics_core evaluation_metrics)
    target_compile_features(evaluation_metrics_tests PRIVATE cxx_std_17)
    set_target_properties(evaluation_metrics_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME evaluation_metrics_tests COMMAND evaluation_metrics_tests)
endif()

install(TARGETS evaluation_metrics evaluation_metrics_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
