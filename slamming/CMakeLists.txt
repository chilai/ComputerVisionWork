cmake_minimum_required(VERSION 2.8.11)
project(motionEstimation)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#message("src dir: " ${CMAKE_SOURCE_DIR}})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIR})

#CERES
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Gflags REQUIRED)
add_definitions(-DCERES_GFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable( motionEstimation src/motionEstimation.cpp src/backend.cpp)
target_link_libraries( motionEstimation ${PCL_LIBRARIES} ${EIGEN3_LIBS} ${CERES_LIBRARIES})
