cmake_minimum_required(VERSION 3.22)
project(main LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
file(GLOB_RECURSE lib_files
	lib/*.cpp
	lib/*.cu
	)
add_library(func SHARED ${lib_files})
target_include_directories(func PUBLIC include)

include(CTest)
add_subdirectory(third_party/googletest)
include_directories(third_party/googletest/googletest/include)
add_executable(reduce_test test/reduce_test.cu)
target_link_libraries(reduce_test gtest gtest_main func)
add_test(NAME reduce_test COMMAND reduce_test)
