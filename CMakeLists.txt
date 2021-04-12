cmake_minimum_required(VERSION 3.10)
set(CMAKE_MINIMUM_REQUIRED_VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")

project(Exp3_a)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(Common)

add_executable(main Experiment3/main.cpp)
target_link_libraries(main common ${OpenCV_LIBS})
