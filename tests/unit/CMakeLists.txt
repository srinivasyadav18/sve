enable_testing()

set (unit_tests 
    dummy
    load_store
    copy_from_to
    addition    
)

foreach(unit_test ${unit_tests})
    message("adding unit test: " ${unit_test})

    add_executable(${unit_test} ${unit_test}.cpp)

    target_link_libraries(${unit_test} sve)

    add_test(NAME ${unit_test} COMMAND ${unit_test})
endforeach()