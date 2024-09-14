cmake_minimum_required(VERSION 3.18)
project(torch_niggli LANGUAGES CXX CUDA)

option(USE_PYBIND "Use pybind11 to create Python bindings" ON)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_POSITION_INDEPENDENT_CODE ON)

include_directories(include ${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

add_library(niggli STATIC src/niggli/niggli.cu)
set_target_properties(niggli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(USE_PYBIND)
    pybind11_add_module(torch_niggli src/torch_niggli/torch_niggli.cu)
    target_link_libraries(torch_niggli PRIVATE niggli ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

    install(TARGETS torch_niggli
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
