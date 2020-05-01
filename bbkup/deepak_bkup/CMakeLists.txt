cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_ENABLE_EXPORTS 1)
set(PYBIND11_PYTHON_VERSION 3.7)

#set(CMAKE_PREFIX_PATH third-party/libtorch/)

# set the project name
set(this_project resident-smith)

# third party software
set(thirdparty third-party)

# RUST, Erlang, what else
project(${this_project} C CXX)

enable_testing()

add_subdirectory(${thirdparty}/googletest)
add_subdirectory(${thirdparty}/googlebenchmark/benchmark)
add_subdirectory(${thirdparty}/drogon)
add_subdirectory(${thirdparty}/neo-ai-dlr)
add_subdirectory(${thirdparty}/pybind11)


find_package(cpprestsdk REQUIRED)

file(GLOB MMGR_SRC src/mmgr/*.cc)
file(GLOB MMGR_INC src/mmgr/*.h)
file(GLOB DROGON_CTRL_INC src/ctrl/drogon/*.h)
file(GLOB DROGON_CTRL_SRC src/ctrl/drogon/*.cc)

set(dlr_project
    ${thirdparty}/neo-ai-dlr
  )


# header files
set(headers
  include/resident-smith.h
  src/runtime/dlr_wrapper.hpp
  ${MMGR_INC}
  ${DROGON_CTRL_INC}
)

# src files
set(sources
  src/resident-smith.cc
  ${MMGR_SRC}
  ${DROGON_CTRL_SRC}
)

# add the executable
add_executable(${this_project} ${sources} ${headers})

# temporarily adding lib instead of executable / for running test
#add_library(${this_project} ${sources} ${headers})

target_include_directories(${this_project} PUBLIC ${dlr_project}/include)
target_include_directories(${this_project} PUBLIC include)
target_include_directories(${this_project} PUBLIC src/runtime)

target_link_libraries(${this_project} PUBLIC
  drogon
  dlr
)

target_link_libraries(${this_project} PRIVATE cpprestsdk::cpprest)

target_link_libraries(${this_project} PRIVATE pybind11::embed)

#find_package(Torch REQUIRED)
#target_link_libraries(${this_project} PRIVATE "${TORCH_LIBRARIES}")

#find_package(OpenCV REQUIRED)
#target_link_libraries(${this_project} PRIVATE ${OpenCV_LIBS})

add_custom_command(
        TARGET ${this_project} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/src/ctrl/drogon/config.json
                ${CMAKE_CURRENT_BINARY_DIR})

#For test
include_directories(src include)
#add_subdirectory(test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
