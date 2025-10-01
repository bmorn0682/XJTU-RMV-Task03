cmake_minimum_required(VERSION 3.10.0)
project(ltest VERSION 0.1.0 LANGUAGES C CXX)

add_executable(ltest main.cpp)
find_package(OpenCV REQUIRED)

# 找到 Ceres
list(APPEND CMAKE_PREFIX_PATH "~/abseil-cpp/install")
find_package(absl REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ltest
    ${OpenCV_LIBS}
    Ceres::ceres
)


include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
