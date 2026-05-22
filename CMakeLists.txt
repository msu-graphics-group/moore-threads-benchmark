cmake_minimum_required(VERSION 3.26)
project(moore-threads-benchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


##################
### Properties ###
##################

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(BIN_DIR  "${ROOT_DIR}/Bin")

unset(CMAKE_INSTALL_PREFIX CACHE)
if(NOT MSVC AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  unset(CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)
endif()

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
  set(CMAKE_SUPPRESS_REGENERATION ON)
endif()

set(API CUDA CACHE STRING "")
set_property(CACHE API PROPERTY STRINGS "CUDA" "HIP" "MUSA" "VULKAN")
if(NOT "${API}" MATCHES "^(CUDA|HIP|MUSA|VULKAN)$")
  message(FATAL_ERROR "Unknown API '${API}'")
endif()


#####################
### Configuration ###
#####################

if(NOT EXISTS ${BIN_DIR})
  file(MAKE_DIRECTORY "${BIN_DIR}")
endif()

if("${API}" STREQUAL "CUDA")
  set(CMAKE_CUDA_ARCHITECTURES native)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(API_CUDA ON)
  add_compile_definitions(API_CUDA)

elseif("${API}" STREQUAL "HIP")
  set(CMAKE_HIP_PLATFORM amd)
  enable_language(HIP)
  find_package(hip REQUIRED)
  set(API_HIP ON)
  add_compile_definitions(API_HIP)

elseif("${API}" STREQUAL "MUSA")
  list(APPEND CMAKE_MODULE_PATH "/usr/local/musa/cmake")
  find_package(MUSAToolkit REQUIRED)
  set(API_MUSA ON)
  add_compile_definitions(API_MUSA)
endif()


#################
### Utilities ###
#################

macro(discover_sources directory headers sources)
  file(GLOB_RECURSE new_headers ${directory}/*.h
                                ${directory}/*.hpp)
  list(APPEND ${headers} ${new_headers})
  file(GLOB_RECURSE new_sources ${directory}/*.cpp)
  list(APPEND ${sources} ${new_sources})

  if(API_CUDA)
    file(GLOB_RECURSE new_sources ${directory}/*.cu)
    list(APPEND ${sources} ${new_sources})
  endif()

  if(API_MUSA)
    file(GLOB_RECURSE new_sources ${directory}/*.mu)
    list(APPEND ${sources} ${new_sources})
  endif()

  # Configure filters for VS project
  if(MSVC)
    foreach(_source IN ITEMS ${${headers}})
      get_filename_component(_source_path "${_source}" PATH)
      file(RELATIVE_PATH _source_path_rel "${directory}" "${_source_path}")
      string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
      source_group("Header Files\\${_group_path}" FILES "${_source}")
    endforeach()

    foreach(_source IN ITEMS ${${sources}})
      get_filename_component(_source_path "${_source}" PATH)
      file(RELATIVE_PATH _source_path_rel "${directory}" "${_source_path}")
      string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
      source_group("Source Files\\${_group_path}" FILES "${_source}")
    endforeach()
  endif()
endmacro()


############
### Core ###
############

discover_sources("${ROOT_DIR}/Core" CORE_HEADERS CORE_SOURCES)
add_executable(Core ${CORE_HEADERS} ${CORE_SOURCES})
target_include_directories(Core PRIVATE "${ROOT_DIR}/Core")
if(API_CUDA)
  target_link_libraries(Core PRIVATE CUDA::cudart)
elseif(API_HIP)
  target_link_libraries(Core PRIVATE hip::host)
elseif(API_MUSA)
  target_link_libraries(Core PRIVATE MUSA::musart)
endif()

add_custom_command(TARGET Core POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Core> "${BIN_DIR}")
