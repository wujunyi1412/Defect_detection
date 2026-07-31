cmake_minimum_required(VERSION 3.24)

project(CoronaDetectionWorkbench LANGUAGES NONE)

find_program(DOTNET_EXECUTABLE dotnet REQUIRED)

option(CORONA_SELF_CONTAINED
    "Publish with the .NET Windows Desktop runtime included" OFF)

set(APP_PROJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/CoronaDetection.App/CoronaDetection.App.csproj")

if(CORONA_SELF_CONTAINED)
    set(SELF_CONTAINED_VALUE true)
else()
    set(SELF_CONTAINED_VALUE false)
endif()

if(CMAKE_CONFIGURATION_TYPES)
    set(APP_CONFIGURATION "$<CONFIG>")
    set(APP_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    set(APP_CONFIGURATION "${CMAKE_BUILD_TYPE}")
    set(APP_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")
endif()

add_custom_target(CoronaDetection ALL
    COMMAND "${DOTNET_EXECUTABLE}" publish
            "${APP_PROJECT}"
            --configuration "${APP_CONFIGURATION}"
            --runtime win-x64
            --self-contained "${SELF_CONTAINED_VALUE}"
            --output "${APP_OUTPUT_DIR}"
            --nologo
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Publishing CoronaDetection WPF application"
    VERBATIM
    USES_TERMINAL)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT CoronaDetection)

add_custom_target(run
    COMMAND "${APP_OUTPUT_DIR}/CoronaDetection.exe"
    DEPENDS CoronaDetection
    WORKING_DIRECTORY "${APP_OUTPUT_DIR}"
    COMMENT "Starting CoronaDetection"
    VERBATIM
    USES_TERMINAL)
