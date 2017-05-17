cmake_minimum_required(VERSION 2.6)
project(objectDetect)

#添加依赖头文件
include_directories( "/usr/include/opencv2" )
include_directories( "/usr/include/opencv" )

add_library(lib svm.cpp)

link_libraries(lib)

add_executable(train-classifier train-classifier.cpp)
add_executable(test-classifier test-classifier.cpp)
add_executable(test test.cpp)

target_link_libraries( train-classifier libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_ml.so
 libopencv_objdetect.so)

target_link_libraries( test-classifier libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_ml.so
 libopencv_objdetect.so)

target_link_libraries( test libopencv_core.so libopencv_ml.so 
libopencv_highgui.so libopencv_imgproc.so libopencv_objdetect.so)

install(TARGETS train-classifier RUNTIME DESTINATION bin)
