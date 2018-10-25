cmake_minimum_required(VERSION 3.12)
project(mpilab2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mpilab2 main.cpp)
target_link_libraries(mpilab2 ${OpenCV_LIBS})

add_executable(mpilab2p parallel.cpp)
target_link_libraries(mpilab2p PUBLIC ${MPI_CXX_LIBRARIES} ${OpenCV_LIBS})