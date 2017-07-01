cmake_minimum_required(VERSION 2.6)
project(simljp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

find_package(MKL REQUIRED)

link_libraries(${MKL_LIBRARIES})
include_directories(${MKL_INCLUDE_DIR} $ENV{EIGEN_INCLUDE_DIR})

add_executable(simljp main.cpp)

install(TARGETS simljp RUNTIME DESTINATION bin)
