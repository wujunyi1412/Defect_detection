cmake_minimum_required(VERSION 3.16)

project(UseInspectionDLLDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(OPENCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/install_opencv")
set(OPENCV_INCLUDE_DIR "${OPENCV_ROOT}/include")
set(OPENCV_LIB_DIR "${OPENCV_ROOT}/lib")
set(OPENCV_BIN_DIR "${OPENCV_ROOT}/bin")

if(NOT EXISTS "${OPENCV_INCLUDE_DIR}")
    message(FATAL_ERROR "OpenCV 头文件目录不存在: ${OPENCV_INCLUDE_DIR}")
endif()

file(GLOB OPENCV_LIBS "${OPENCV_LIB_DIR}/opencv_*.lib")
if(NOT OPENCV_LIBS)
    message(FATAL_ERROR "在 ${OPENCV_LIB_DIR} 中未找到 OpenCV 库文件")
endif()

add_executable(use_examples main.cpp)

target_include_directories(use_examples PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../Corona_defect_detection"
    "${OPENCV_INCLUDE_DIR}"
)

set(CORONA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/$<CONFIG>" CACHE PATH "Corona_defect_detection.lib 所在目录")

target_link_libraries(use_examples PRIVATE
    ${OPENCV_LIBS}
    "${CORONA_LIB_DIR}/Corona_defect_detection.lib"
)

set_target_properties(use_examples PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_custom_command(TARGET use_examples POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/config.ini"
    $<TARGET_FILE_DIR:use_examples>
)

if(WIN32)
    file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/opencv_*.dll")
    foreach(OPENCV_DLL ${OPENCV_DLLS})
        add_custom_command(TARGET use_examples POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL}"
            $<TARGET_FILE_DIR:use_examples>
        )
    endforeach()

    set(ONNX_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/../onnx_model")

    add_custom_command(TARGET use_examples POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ONNX_MODEL}"
        "$<TARGET_FILE_DIR:use_examples>/onnx_model"
    )

    set(INSPECTION_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/bin/Release" CACHE PATH "InspectionDLL.dll 所在目录(可选)")
    if(INSPECTION_DLL_DIR AND EXISTS "${INSPECTION_DLL_DIR}")
        file(GLOB INSPECTION_RUNTIME_DLLS "${INSPECTION_DLL_DIR}/*.dll")
        foreach(RUNTIME_DLL ${INSPECTION_RUNTIME_DLLS})
            add_custom_command(TARGET use_examples POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${RUNTIME_DLL}"
                $<TARGET_FILE_DIR:use_examples>
            )
        endforeach()
    endif()
endif()

