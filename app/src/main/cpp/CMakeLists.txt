# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("snpechainingdemo" LANGUAGES CXX)

# Use modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally allow the user to request debug/release from the Gradle side
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.

find_library(log-lib log)
find_library(android-lib android)

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        inference.cpp inference_helper.cpp snpedemo_jni.cpp
        TensorWorkspace.cpp ModelSession.cpp GraphRunner.cpp
        ParseConfig.cpp newInferenceHelper.cpp typical_usage_jni.cpp
        initTensorsHelper.cpp
        SDXLPipeline.cpp EulerDiscreteScheduler.cpp
        SDXLPipelineGranular.cpp
        Spar3DPipeline.cpp)
# --- Optimization Flags for Main Library ---
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Ofast -ffast-math)
endif()

#add_library(${CMAKE_PROJECT_NAME} SHARED
#        # List C/C++ source files with relative paths to this CMakeLists.txt.
#        native-lib.cpp)

set(EIGEN5_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
add_library(eigen5 INTERFACE)
target_include_directories(eigen5 INTERFACE ${EIGEN5_INCLUDE_DIR})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(eigen5 INTERFACE EIGEN_NO_DEBUG EIGEN_NO_STATIC_ASSERT)
endif()

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/zdl
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/eigen
)

set(SNPE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

add_library(SNPE            SHARED IMPORTED)
set_target_properties(SNPE  PROPERTIES IMPORTED_LOCATION ${SNPE_LIB_DIR}/libSNPE.so)

# --- OpenCV: import prebuilt libopencv_java4.so from jniLibs ----------
# Assumes you copied libopencv_java4.so into app/src/main/jniLibs/<ABI>/
set(OPENCV_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
add_library(opencv_java4 SHARED IMPORTED)
set_target_properties(opencv_java4 PROPERTIES
    IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_java4.so
)


# --- UVUnwrapper native library
set(UV_SOURCES
        # core algorithm sources - ensure these files exist in this directory
        bvh.cpp
        intersect.cpp
        unwrapper.cpp            # or unwrapper_native.cpp if you renamed it
        uv_unwrapper_api.cpp     # the C API wrapper you created
        UnwrapperClass.cpp
        MeshCPP.cpp
        uv_unwrapper_api_bridge.cpp
        baker_cpu.cpp
        tiny_gltf.cc
)
# Build as a static lib first, then link into your main shared lib
add_library(uvunwrapper STATIC ${UV_SOURCES})
# Allow the unwrapper to find headers in the current directory (and any subfolders)
target_include_directories(uvunwrapper PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/zdl
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/gltf
)

# Optional: compilation flags for uvunwrapper
# e.g. high optimization in release builds
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
#    target_compile_options(uvunwrapper PRIVATE -O3)
    # -Ofast implies -O3 and -ffast-math.
    # We add -ffast-math explicitly just to be sure.
    target_compile_options(uvunwrapper PRIVATE -Ofast -ffast-math)
endif()

# --- OpenMP support (optional) -----------------------------------------------
# We try to find OpenMP; if found we add compile/link flags. On many Android NDKs
# OpenMP requires special flags and libraries. Leave commented/optional until checked.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling OpenMP for uvunwrapper")
    target_compile_options(uvunwrapper PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(uvunwrapper PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found (this is OK). Building uvunwrapper single-threaded.")
endif()

# NOTE: if you want to explicitly enable OpenMP on Android, you can uncomment:
# target_compile_options(uvunwrapper PRIVATE -fopenmp)
# target_link_libraries(uvunwrapper PRIVATE -fopenmp)
# but ensure your NDK provides libomp/libgomp (varies by NDK version).

# --- Link uvunwrapper into main shared lib -------------------------------
# Link PRIVATE so the symbols are available to your main target, but no additional
# transitive include/link issues appear.
#target_link_libraries(${CMAKE_PROJECT_NAME}
#        PRIVATE
#        uvunwrapper
#)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
#target_link_libraries(${CMAKE_PROJECT_NAME}
#        # List libraries link to the target library
#        android
#        log)
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        PUBLIC
        android
        log
        ${log-lib}
        ${android-lib}
        PRIVATE
        SNPE
        uvunwrapper
        opencv_java4
        eigen5
)

#target_link_libraries(${CMAKE_PROJECT_NAME}
#    PRIVATE
#        opencv_java4
#        eigen5
#)
target_link_libraries(uvunwrapper
        PUBLIC
        android
        log
        PRIVATE
        opencv_java4
        eigen5
)