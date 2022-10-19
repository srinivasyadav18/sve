enable_testing()

set (unit_tests 
    iota
    dummy
    load_store
    copy_from_to
    addition
    fft
    reduce
    scan
    index_series
    random
    permute
)

foreach(unit_test ${unit_tests})
    message("adding unit test: " ${unit_test})
    set(target ${unit_test}_unit_test)
    add_executable(${target} ${unit_test}.cpp)

    target_link_libraries(${target} sve)

    add_test(NAME ${target} COMMAND ${CMAKE_CROSSCOMPILING_CMD} ${target})
endforeach()