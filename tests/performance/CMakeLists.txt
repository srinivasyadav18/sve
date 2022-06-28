enable_testing()

set (perf_tests 
    fft
)

foreach(perf_test ${perf_tests})
    message("adding perf test: " ${perf_test})
    set(target ${perf_test}_perf_test)
    add_executable(${target} ${perf_test}.cpp)

    target_link_libraries(${target} sve)

    add_test(NAME ${target} COMMAND ${target})
endforeach()