cmake_minimum_required(VERSION 3.16)

project(Corona_defect_detection VERSION 1.0.1 LANGUAGES CXX)

include(CTest)
add_subdirectory(evaluation_metrics)

# set(CORONA_COMPANY_NAME "Corona Defect Detection" CACHE STRING "DLL company name")
set(CORONA_FILE_DESCRIPTION "Corona Defect Detection Inference Library" CACHE STRING "DLL file description")
set(CORONA_PRODUCT_NAME "Corona Defect Detection" CACHE STRING "DLL product name")
# set(CORONA_LEGAL_COPYRIGHT "Copyright (C) 2026" CACHE STRING "DLL copyright")
# set(CORONA_COMMENTS "ONNX-based defect detection inference DLL" CACHE STRING "DLL comments")

if(WIN32)
    enable_language(RC)

    # Windows VERSIONINFO requires four numeric version components.
    set(CORONA_VERSION_TWEAK "${PROJECT_VERSION_TWEAK}")
    if(CORONA_VERSION_TWEAK STREQUAL "")
        set(CORONA_VERSION_TWEAK 0)
    endif()
    set(CORONA_VERSION_STRING
        "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.${CORONA_VERSION_TWEAK}")

    set(CORONA_VERSION_RESOURCE
        "${CMAKE_CURRENT_BINARY_DIR}/generated/Corona_defect_detection_version.rc")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows_version_info.rc.in"
        "${CORONA_VERSION_RESOURCE}"
        @ONLY
    )
endif()

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS ON)

set(OPENCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/install_opencv")
set(OPENCV_INCLUDE_DIR "${OPENCV_ROOT}/include")
set(OPENCV_LIB_DIR "${OPENCV_ROOT}/lib")
set(OPENCV_BIN_DIR "${OPENCV_ROOT}/bin")


if(NOT EXISTS "${OPENCV_INCLUDE_DIR}")
    message(FATAL_ERROR "OpenCV 头文件目录不存在: ${OPENCV_INCLUDE_DIR}")
endif()

file(GLOB OPENCV_LIBS "${OPENCV_LIB_DIR}/opencv_*.lib")
if(OPENCV_LIBS)
    list(GET OPENCV_LIBS 0 FIRST_OPENCV_LIB)
    message(STATUS "找到 OpenCV 库: ${FIRST_OPENCV_LIB}")
else()
    message(FATAL_ERROR "在 ${OPENCV_LIB_DIR} 中未找到 OpenCV 库文件")
endif()

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-win-x64-1.24.4")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "ONNX Runtime 未找到: ${ONNXRUNTIME_ROOT}")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)
message(STATUS "使用 ONNX Runtime: ${ONNXRUNTIME_ROOT}")

set(HALCON_ROOT "D:/halcon_12" CACHE PATH "HALCON 根目录")
set(HALCON_ARCH "x64-win64" CACHE STRING "HALCON 架构目录名")
set(HALCON_INCLUDE_DIR "${HALCON_ROOT}/include")
set(HALCONCPP_INCLUDE_DIR "${HALCON_ROOT}/include/halconcpp")
set(HALCON_LIB_DIR "${HALCON_ROOT}/lib/${HALCON_ARCH}")
set(HALCON_BIN_DIR "${HALCON_ROOT}/bin/${HALCON_ARCH}")

if(NOT EXISTS "${HALCON_INCLUDE_DIR}")
    message(FATAL_ERROR "HALCON 头文件目录不存在: ${HALCON_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${HALCON_LIB_DIR}")
    message(FATAL_ERROR "HALCON 库目录不存在: ${HALCON_LIB_DIR}")
endif()
if(NOT EXISTS "${HALCON_BIN_DIR}")
    message(FATAL_ERROR "HALCON bin 目录不存在: ${HALCON_BIN_DIR}")
endif()

set(FAISS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/install_faiss")
if(EXISTS "${FAISS_ROOT}")
    include_directories("${FAISS_ROOT}/include")
    link_directories("${FAISS_ROOT}/lib")
    message(STATUS "使用 Faiss: ${FAISS_ROOT}")
else()
    message(WARNING "Faiss 未找到，PatchCore 功能将受限")
endif()

set(OPENBLAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/install_openblas")


set(DLL_SOURCES
    config/config.cpp
    config/ini_parser.cpp
    logging/logger.cpp
    profiling/performance_timer.cpp
    Corona_defect_detection/inference_c_api.cpp
    Corona_defect_detection/inference_dll.cpp                    
    Corona_defect_detection/internal/overlay_renderer.cpp
    Corona_defect_detection/internal/image_utils.cpp
    Corona_defect_detection/internal/image_utils_adaptive.cpp
    Corona_defect_detection/internal/aa_yolo_filter.cpp
    Corona_defect_detection/internal/iqt_yolo_filter.cpp
    Corona_defect_detection/internal/dark_defect_refiner.cpp
    Corona_defect_detection/internal/dark_defect_refiner_adaptive.cpp
    Corona_defect_detection/internal/patchcore_yolo_iou_filter.cpp
    Corona_defect_detection/internal/patchcore_postprocess.cpp
    Corona_defect_detection/internal/yolo_postprocess.cpp
    Corona_defect_detection/internal/result_builder.cpp
    Corona_defect_detection/yolo_inference.cpp                   
    Corona_defect_detection/patchcore_inference.cpp             
    image_process/image_process.cpp      
)

if(WIN32)
    list(APPEND DLL_SOURCES "${CORONA_VERSION_RESOURCE}")
endif()

set(DLL_HEADERS
    config/config.h
    config/ini_parser.h
    logging/logger.h
    profiling/performance_timer.h
    Corona_defect_detection/inference_dll.h
    Corona_defect_detection/inference_c_api.h
    Corona_defect_detection/internal/overlay_renderer.h
    Corona_defect_detection/internal/image_utils.h
    Corona_defect_detection/internal/aa_yolo_filter.h
    Corona_defect_detection/internal/iqt_yolo_filter.h
    Corona_defect_detection/internal/dark_defect_refiner.h
    Corona_defect_detection/internal/patchcore_yolo_iou_filter.h
    Corona_defect_detection/internal/patchcore_postprocess.h
    Corona_defect_detection/internal/yolo_postprocess.h
    Corona_defect_detection/internal/yolo_categories.h
    Corona_defect_detection/internal/result_builder.h
    Corona_defect_detection/yolo_inference.h
    Corona_defect_detection/patchcore_inference.h
    image_process/image_process.h
)

add_library(Corona_defect_detection SHARED ${DLL_SOURCES} ${DLL_HEADERS})

set_target_properties(Corona_defect_detection PROPERTIES
    VERSION "${PROJECT_VERSION}"
)

target_compile_definitions(Corona_defect_detection PRIVATE INSPECTION_CAPI_EXPORTS)

target_include_directories(Corona_defect_detection PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_SOURCE_DIR}/logging
    ${CMAKE_CURRENT_SOURCE_DIR}/profiling
    ${CMAKE_CURRENT_SOURCE_DIR}/Corona_defect_detection 
    ${CMAKE_CURRENT_SOURCE_DIR}/Corona_defect_detection/internal
    ${CMAKE_CURRENT_SOURCE_DIR}/image_process         
    ${OPENCV_INCLUDE_DIR}                
    ${ONNXRUNTIME_ROOT}/include         
    ${HALCON_INCLUDE_DIR}
    ${HALCONCPP_INCLUDE_DIR}
)

target_link_libraries(Corona_defect_detection ${OPENCV_LIBS})
target_link_libraries(Corona_defect_detection "${HALCON_LIB_DIR}/halconcpp.lib" "${HALCON_LIB_DIR}/halcon.lib")

if(WIN32)
    target_link_libraries(Corona_defect_detection ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
else()
    target_link_libraries(Corona_defect_detection ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

if(faiss_FOUND OR FAISS_ROOT)
    if(WIN32)
        if(EXISTS "${FAISS_ROOT}/lib/faiss.lib")
            target_link_libraries(Corona_defect_detection ${FAISS_ROOT}/lib/faiss.lib)
        elseif(EXISTS "${FAISS_ROOT}/lib/faiss_d.lib")
            target_link_libraries(Corona_defect_detection ${FAISS_ROOT}/lib/faiss_d.lib)
        endif()
    else()
        target_link_libraries(Corona_defect_detection faiss)
    endif()
endif()


set_target_properties(Corona_defect_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)


if(WIN32)
    add_custom_command(TARGET Corona_defect_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:Corona_defect_detection>    
    )

    add_custom_command(TARGET Corona_defect_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALCON_BIN_DIR}/halcon.dll"
        $<TARGET_FILE_DIR:Corona_defect_detection>
    )

    add_custom_command(TARGET Corona_defect_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALCON_BIN_DIR}/halconcpp.dll"
        $<TARGET_FILE_DIR:Corona_defect_detection>
    )

#    add_custom_command(TARGET Corona_defect_detection POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        "${HALCON_BIN_DIR}/halconc.dll"
#        $<TARGET_FILE_DIR:Corona_defect_detection>
#    )

    if(FAISS_ROOT AND EXISTS "${FAISS_ROOT}/bin/faiss.dll")
        add_custom_command(TARGET Corona_defect_detection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FAISS_ROOT}/bin/faiss.dll"
            $<TARGET_FILE_DIR:Corona_defect_detection>
        )
    endif()

    if(FAISS_ROOT AND EXISTS "${FAISS_ROOT}/bin/openblas.dll")
        add_custom_command(TARGET Corona_defect_detection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FAISS_ROOT}/bin/openblas.dll"
            $<TARGET_FILE_DIR:Corona_defect_detection>
        )
    elseif(OPENBLAS_ROOT AND EXISTS "${OPENBLAS_ROOT}/bin/openblas.dll")
        add_custom_command(TARGET Corona_defect_detection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENBLAS_ROOT}/bin/openblas.dll"
            $<TARGET_FILE_DIR:Corona_defect_detection>
        )
    endif()

#    file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/opencv_*.dll")
    set(OPENCV_DLLS
    "${OPENCV_BIN_DIR}/opencv_core453.dll"
    "${OPENCV_BIN_DIR}/opencv_imgproc453.dll"
    "${OPENCV_BIN_DIR}/opencv_imgcodecs453.dll"
    )
    foreach(OPENCV_DLL ${OPENCV_DLLS})
        add_custom_command(TARGET Corona_defect_detection POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_DLL}"
            $<TARGET_FILE_DIR:Corona_defect_detection>
        )
    endforeach()

endif()

option(BUILD_SMOKE_TESTS "Build lightweight smoke tests" ON)

if(BUILD_SMOKE_TESTS)
    add_executable(corona_smoke_tests
        tests/smoke_tests.cpp
        config/config.cpp
        config/ini_parser.cpp
        logging/logger.cpp
        profiling/performance_timer.cpp
        Corona_defect_detection/internal/overlay_renderer.cpp
        Corona_defect_detection/internal/image_utils.cpp
        Corona_defect_detection/internal/image_utils_adaptive.cpp
        Corona_defect_detection/internal/aa_yolo_filter.cpp
        Corona_defect_detection/internal/iqt_yolo_filter.cpp
        Corona_defect_detection/internal/dark_defect_refiner.cpp
        Corona_defect_detection/internal/dark_defect_refiner_adaptive.cpp
        Corona_defect_detection/internal/patchcore_yolo_iou_filter.cpp
        Corona_defect_detection/internal/result_builder.cpp
        Corona_defect_detection/internal/yolo_postprocess.cpp
        image_process/image_process.cpp
    )

    target_include_directories(corona_smoke_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/config
        ${CMAKE_CURRENT_SOURCE_DIR}/logging
        ${CMAKE_CURRENT_SOURCE_DIR}/profiling
        ${CMAKE_CURRENT_SOURCE_DIR}/Corona_defect_detection
        ${CMAKE_CURRENT_SOURCE_DIR}/image_process
        ${OPENCV_INCLUDE_DIR}
        ${HALCON_INCLUDE_DIR}
        ${HALCONCPP_INCLUDE_DIR}
    )

    target_link_libraries(corona_smoke_tests PRIVATE ${OPENCV_LIBS})
    target_link_libraries(corona_smoke_tests PRIVATE "${HALCON_LIB_DIR}/halconcpp.lib" "${HALCON_LIB_DIR}/halcon.lib")

    set_target_properties(corona_smoke_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()


if(BUILD_EXAMPLES)
    add_executable(dll_test_main dll_main.cpp)

    target_link_libraries(dll_test_main Corona_defect_detection)
    
    set_target_properties(dll_test_main PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

install(TARGETS Corona_defect_detection
    RUNTIME DESTINATION bin      
    LIBRARY DESTINATION lib      
    ARCHIVE DESTINATION lib      
)

install(FILES ${DLL_HEADERS} DESTINATION include)

message(STATUS "")
message(STATUS "=== Corona_defect_detection 构建配置 ===")
message(STATUS "OpenCV 路径:      ${OPENCV_ROOT}")
message(STATUS "ONNX Runtime:     ${ONNXRUNTIME_ROOT}")
message(STATUS "Faiss 可用:       ${faiss_FOUND}")
message(STATUS "编译示例程序:     ${BUILD_EXAMPLES}")
message(STATUS "输出目录:         ${CMAKE_BINARY_DIR}/bin")
message(STATUS "==============================")
