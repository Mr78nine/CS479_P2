cmake_minimum_required(VERSION 3.13)
set(CMAKE_MINIMUM_REQUIRED_VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(Exp3_a)

add_subdirectory(Common)

add_executable(main Experiment3/main.cpp)
target_link_libraries(main common)
