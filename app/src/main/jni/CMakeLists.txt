project(nanodetncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR D:/Android_projects/libs/opencv-mobile-2.4.13.7-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR D:/Android_projects/libs/ncnn-20240102-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(nanodetncnn SHARED nanodetncnn.cpp nanodetplus.cpp ndkcamera.cpp)

target_link_libraries(nanodetncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
