cmake_minimum_required(VERSION 3.2)

project(CNET
		LANGUAGES CXX)
		
SET(CMAKE_CONFIGURATION_TYPES Debug;Release;RelWithDebInfo)

configure_file(train-images.idx3-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-images.idx3-ubyte COPYONLY)
configure_file(train-labels.idx1-ubyte ${CMAKE_CURRENT_BINARY_DIR}/train-labels.idx1-ubyte COPYONLY)

add_executable(CNET main.cpp layers.h helperFuncs.h)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CNET PUBLIC OpenMP::OpenMP_CXX)
endif()