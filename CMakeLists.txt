cmake_minimum_required(VERSION 3.10)

set(target_name myLayers)
project(${target_name})

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND
)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${target_name} MODULE myLayers.cpp)

target_include_directories(${target_name} PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(${target_name} ${PYTHON_LIBRARIES})
set_target_properties(${target_name} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
endif()
