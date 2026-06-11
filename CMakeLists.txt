# CMake minimum version (FetchContent requires 3.14+)
cmake_minimum_required(VERSION 3.14)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(ocr_service LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# ─── Optional ncnn dependency ────────────────────────────────
# Expected layout:
#   3rd_party/ncnn/x64/lib/cmake/ncnn/ncnnConfig.cmake
#   3rd_party/ncnn/x86/lib/cmake/ncnn/ncnnConfig.cmake
#   3rd_party/ncnn/arm64/lib/cmake/ncnn/ncnnConfig.cmake
set(NCNN_ROOT "${CMAKE_SOURCE_DIR}/3rd_party/ncnn" CACHE PATH "Root directory containing prebuilt ncnn x64/x86/arm64 packages")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        set(NCNN_ARCH_DIR "${NCNN_ROOT}/arm64")
    else()
        set(NCNN_ARCH_DIR "${NCNN_ROOT}/x64")
    endif()
else()
    set(NCNN_ARCH_DIR "${NCNN_ROOT}/x86")
endif()

set(NCNN_CMAKE_DIR "${NCNN_ARCH_DIR}/lib/cmake/ncnn")
set(HAVE_NCNN OFF)
if(EXISTS "${NCNN_CMAKE_DIR}/ncnnConfig.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${NCNN_CMAKE_DIR}")
    find_package(ncnn CONFIG QUIET)
    if(ncnn_FOUND)
        set(HAVE_NCNN ON)
        message(STATUS "Using ncnn from: ${NCNN_ARCH_DIR}")
    endif()
endif()

# ─── Dependencies ─────────────────────────────────────────────
include(FetchContent)

# cpp-httplib (header-only HTTP server)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.3
)

# nlohmann/json (header-only JSON library)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(httplib json)

# Tesseract OCR (optional for ncnn-only development)
option(BUILD_TESSERACT_SERVER "Build the Tesseract-based OCR HTTP service" ON)
set(HAVE_TESSERACT OFF)
if(BUILD_TESSERACT_SERVER)
    find_package(Tesseract QUIET)
    if(Tesseract_FOUND)
        set(HAVE_TESSERACT ON)
        message(STATUS "Using Tesseract from package configuration")
    else()
        message(WARNING "Tesseract development package not found. Tesseract server targets will be skipped.")
    endif()
endif()

# ─── OCR Service Executable ──────────────────────────────────
if(HAVE_TESSERACT)
    add_executable(ocr_server
        src/main.cpp
        src/ocr_engine.cpp
        src/ocr_server.cpp
    )

    target_include_directories(ocr_server PRIVATE src)

    target_link_libraries(ocr_server PRIVATE
        Tesseract::libtesseract
        httplib::httplib
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(ocr_server PRIVATE OCR_HAS_TESSERACT=1)
endif()

if(BUILD_TESTING AND HAVE_TESSERACT)
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)

    add_executable(ocr_unit_tests
        tests/ocr_engine_test.cpp
        src/ocr_engine.cpp
    )

    target_include_directories(ocr_unit_tests PRIVATE src)

    target_link_libraries(ocr_unit_tests PRIVATE
        Tesseract::libtesseract
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(ocr_integration_tests
        tests/ocr_integration_test.cpp
        src/ocr_engine.cpp
        src/ocr_server.cpp
    )

    target_include_directories(ocr_integration_tests PRIVATE src)

    target_link_libraries(ocr_integration_tests PRIVATE
        Tesseract::libtesseract
        httplib::httplib
        nlohmann_json::nlohmann_json
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(ocr_unit_tests)
    gtest_discover_tests(ocr_integration_tests)
endif()

if(HAVE_NCNN)
    add_library(ocr_ncnn_config INTERFACE)
    target_link_libraries(ocr_ncnn_config INTERFACE ncnn)
    target_compile_definitions(ocr_ncnn_config INTERFACE OCR_HAS_NCNN=1)

    add_library(ocr_ncnn_engine STATIC
        src/ncnn_ocr_engine.cpp
    )
    target_include_directories(ocr_ncnn_engine PUBLIC src)
    target_link_libraries(ocr_ncnn_engine PRIVATE ocr_ncnn_config)

    if(NOT TARGET ocr_server)
        add_executable(ocr_server
            src/main.cpp
            src/ocr_server.cpp
        )
        target_include_directories(ocr_server PRIVATE src)
        target_link_libraries(ocr_server PRIVATE
            httplib::httplib
            nlohmann_json::nlohmann_json
        )
    endif()

    target_link_libraries(ocr_server PRIVATE ocr_ncnn_engine ocr_ncnn_config)
endif()

# ─── Legacy DLL target (optional, disabled by default) ───────
option(BUILD_LEGACY_DLL "Build the legacy tess_engine DLL" OFF)
if(BUILD_LEGACY_DLL)
    if(NOT HAVE_TESSERACT)
        message(FATAL_ERROR "BUILD_LEGACY_DLL=ON requires Tesseract development package")
    endif()
    add_library(tess_engine SHARED src/tess_ocr.cpp src/tess_engine.def)
    target_include_directories(tess_engine PRIVATE include)
    target_link_libraries(tess_engine PRIVATE Tesseract::libtesseract)
    target_compile_definitions(tess_engine PRIVATE OCR_LIB_EXPORT)

    add_executable(test_legacy src/test.cpp)
    target_include_directories(test_legacy PRIVATE include)
    target_link_libraries(test_legacy PRIVATE tess_engine)
endif()
