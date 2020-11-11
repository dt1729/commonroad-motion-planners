project(crcc)
cmake_minimum_required(VERSION 3.0)

option(ADD_TESTS "unit tests" OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

message(STATUS " ${EIGEN3_INCLUDE_DIR}")

find_package(Eigen3 3.0.5 QUIET CONFIG)

# If Eigen3Config.cmake is not found, use the FindEigen3.cmake module
if(NOT Eigen3_FOUND)
  find_package(Eigen3 3.0.5 QUIET MODULE)
  set(Eigen3_FOUND ON)
endif()

if(Eigen3_FOUND)
  set(COM_HAVE_EIGEN TRUE)
else()
  message(SEND_ERROR "EIGEN3 (>= 3.0.5) is required by the CollisionChecker")
  set(COM_HAVE_EIGEN FALSE)
endif()

if(ADD_TESTS)
  add_subdirectory(cpp/tests)
endif()

if(ADD_PYTHON_BINDINGS)
    set(PYTHON_INCLUDE_DIR ${PATH_TO_PYTHON_ENVIRONMENT}/include/python${PYTHON_VERSION}m)
    set(PYTHON_LIBRARY ${PATH_TO_PYTHON_ENVIRONMENT}/lib/libpython${PYTHON_VERSION}m.so)
    set(PYTHON_EXECUTABLE ${PATH_TO_PYTHON_ENVIRONMENT}/bin/python${PYTHON_VERSION}m)
    add_subdirectory(third_party/pybind11)
    add_subdirectory(python_binding)
endif()

add_subdirectory(cpp)


