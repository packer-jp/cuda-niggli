cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(NiggliLibrary CUDA CXX)

find_package(CUDA REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_POSITION_INDEPENDENT_CODE ON)

include_directories(include ${CUDA_INCLUDE_DIRS})

add_library(niggli STATIC src/niggli/niggli.cu)
set_target_properties(niggli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET niggli PROPERTY CXX_STANDARD 17)
set_target_properties(niggli PROPERTIES CUDA_ARCHITECTURES "80")

add_executable(test_niggli src/niggli/main.cpp)
target_link_libraries(test_niggli niggli)
set_property(TARGET test_niggli PROPERTY CXX_STANDARD 17)
set_target_properties(test_niggli PROPERTIES CUDA_ARCHITECTURES "80")

option(USE_PYBIND "Use pybind11 to create Python bindings" ON)

if(USE_PYBIND)
    find_package(Torch REQUIRED)
    include_directories(${TORCH_INCLUDE_DIRS})
    add_subdirectory(pybind11)

    pybind11_add_module(torch_niggli src/torch_niggli/torch_niggli.cu)
    set_property(TARGET torch_niggli PROPERTY CXX_STANDARD 17)
    target_link_libraries(torch_niggli PRIVATE niggli ${TORCH_LIBRARIES})

    find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(torch_niggli PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()
