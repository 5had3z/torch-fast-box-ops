cmake_minimum_required(VERSION 3.26)

project(profile_box_ops LANGUAGES CXX CUDA)

include(CPM.cmake)

cpmaddpackage(
  NAME
  cxxopts
  GITHUB_REPOSITORY
  jarro2783/cxxopts
  VERSION
  3.3.1
  OPTIONS
  "CXXOPTS_BUILD_EXAMPLES NO"
  "CXXOPTS_BUILD_TESTS NO")

find_package(Python3 COMPONENTS Development)

if(NOT TORCH_PATH)
  # Query python as to where PyTorch is located if not already defined
  execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0])" OUTPUT_VARIABLE TORCH_PATH)
  string(STRIP ${TORCH_PATH} TORCH_PATH)
endif()

find_package(
  Torch
  REQUIRED
  HINTS
  "${TORCH_PATH}/share/cmake/Torch")

find_package(TBB REQUIRED)

file(
  GLOB
  SOURCES
  "*.cpp"
  "*.cu")

add_executable(box-ops-profile ${SOURCES})

target_compile_options(box-ops-profile PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G;>>
                                              $<$<CONFIG:RelWithDebInfo>:$<$<COMPILE_LANGUAGE:CUDA>:-G;>>)

set_target_properties(box-ops-profile PROPERTIES CUDA_ARCHITECTURES "native")

target_link_libraries(
  box-ops-profile
  PUBLIC ${TORCH_LIBRARIES}
         TBB::tbb
         cxxopts::cxxopts
         Python3::Python)
