set(PROJECT "TestDerivated")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE(MPI REQUIRED)

message("\nTest on cmake v" ${CMAKE_VERSION} "  System: " ${CMAKE_SYSTEM} "\n")

# ------------------------------------------------------------------

Set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# CHECK C++ 11

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# ------------------------------------------------------------------

include_directories(${MPI_INCLUDE_PATH})

add_executable(testderivated ${PROJECT_SOURCE_DIR}/TestDerivated.cpp)

set(EXECUTABLE_OUTPUT_PATH .)
