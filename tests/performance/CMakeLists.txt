enable_testing()

set (perf_tests 
    fft
)

foreach(perf_test ${perf_tests})
    message("adding perf test: " ${perf_test})

    add_executable(${perf_test} ${perf_test}.cpp)

    target_link_libraries(${perf_test} sve)

    add_test(NAME ${perf_test} COMMAND ${perf_test})
endforeach()