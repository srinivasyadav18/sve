cmake_minimum_required(VERSION 3.19)

project(SVE VERSION 1.0)

include(GNUInstallDirs)
include(CTest)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(IS_TOPLEVEL_PROJECT TRUE)
else()
  set(IS_TOPLEVEL_PROJECT FALSE)
endif()

option(SVE_BUILD_TESTING "Build and run SVE tests " ${IS_TOPLEVEL_PROJECT})
option(SVE_INSTALL_LIBRARY
       "Enable installing of SVE library into default locations"
       ${IS_TOPLEVEL_PROJECT})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(sve INTERFACE)
add_library(SVE::sve ALIAS sve)

target_include_directories(
  sve INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(sve INTERFACE cxx_std_20)
target_compile_options(sve INTERFACE "-mcpu=a64fx" "-msve-vector-bits=512")

if(BUILD_TESTING AND SVE_BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(SVE_INSTALL_LIBRARY)
  install(
    TARGETS sve
    EXPORT SVE_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "SVEConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/SVEConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/SVEConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/sve/cmake)

  install(
    EXPORT SVE_Targets
    FILE SVETargets.cmake
    NAMESPACE SVE::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/sve/cmake)

  install(FILES "${PROJECT_BINARY_DIR}/SVEConfig.cmake"
                "${PROJECT_BINARY_DIR}/SVEConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/sve/cmake)

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/sve
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
