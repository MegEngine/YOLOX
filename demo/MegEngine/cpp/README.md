# YOLOX-CPP-MegEngine

Compilation of CPP files for YOLOX object detection based on [MegEngine](https://github.com/MegEngine/MegEngine).

## Tutorial

### Step1: Install Toolchain

* Native building
    
    sudo apt install gcc/g++ (gcc/g++, which version >= 6) build-essential git git-lfs gfortran libgfortran-6-dev autoconf gnupg flex bison gperf curl zlib1g-dev gcc-multilib g++-multilib cmake

* Cross building with Android
    1. download [NDK](https://developer.android.com/ndk/downloads)
    1. unzip NDK
    1. export NDK_ROOT="path of NDK" 

        ***Replace `path of NDK` according to your situation.***

### Step2: Build MegEngine

1. Download MegEngine
    ```sh
    git clone https://github.com/MegEngine/MegEngine.git
    ```
1. Init third_party
    ```sh
    # Replace `path of MegEngine` according to your situation.
    export megengine_root="path of MegEngine"

    cd $megengine_root 

    ./third_party/prepare.sh

    ./third_party/install-mkl.sh
    ```

1. Build example:
    * Native building **without** CUDA 
        ```sh
        ./scripts/cmake-build/host_build.sh
        ```
    * Native building **with** CUDA
        ```sh
        ./scripts/cmake-build/host_build.sh -c
        ```
    * Cross building for Android AArch64
        ```sh
        ./scripts/cmake-build/cross_build_android_arm_inference.sh
        ```
    * Cross building for Android AArch64 **(V8.2+fp16)**
        ```sh
        ./scripts/cmake-build/cross_build_android_arm_inference.sh -f
        ```
1. Export `MGE_INSTALL_PATH`
    * Native building **without** CUDA
      ```sh
      export MGE_INSTALL_PATH=${megengine_root}/build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/install
      ```
    * Native building **with** CUDA
      ```sh
      export MGE_INSTALL_PATH=${megengine_root}/build_dir/host/MGE_WITH_CUDA_ON/MGE_INFERENCE_ONLY_ON/Release/install
      ```
    * Cross building for Android AArch64
      ```sh
      export MGE_INSTALL_PATH=${megengine_root}/build_dir/android/arm64-v8a/Release/install
      ```
1. Refer to [Build Tutorial of MegEngine](https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md) to build for other platforms (windows, macos, etc.).

### Step3: Build OpenCV

1. Download OpenCV
    ```sh
    git clone https://github.com/opencv/opencv.git

    # Replace `path of opencv` according to your situation.
    export opencv_root="path of opencv"
    ```

1. Choose Version
    ```sh
    # Our test is based on version 3.4.15.
    # If test other versions, following building steps may not work and some changes need to be made.
    git checkout 3.4.15 
    ```

1. Build
    * Native building

        ```sh
        cd root_dir_of_opencv

        mkdir -p build/install

        cd build

        cmake -DBUILD_JAVA=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install 

        make install -j32
        ```

    * Cross building for Android-AArch64

        * Apply Patch
        ```sh
        # Patch for Android
        diff --git a/CMakeLists.txt b/CMakeLists.txt
        index f6a2da5310..10354312c9 100644
        --- a/CMakeLists.txt
        +++ b/CMakeLists.txt
        @@ -643,7 +643,7 @@ if(UNIX)
            if(NOT APPLE)
              CHECK_INCLUDE_FILE(pthread.h HAVE_PTHREAD)
              if(ANDROID)
        -      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log)
        +      set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} dl m log z)
              elseif(CMAKE_SYSTEM_NAME MATCHES "FreeBSD|NetBSD|DragonFly|OpenBSD|Haiku")
                set(OPENCV_LINKER_LIBS ${OPENCV_LINKER_LIBS} m pthread)
              elseif(EMSCRIPTEN)
        ```
        * Build
        ```sh
        cd root_dir_of_opencv

        mkdir -p build_android/install

        cd build_android
        
        cmake -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_NDK="$NDK_ROOT"  -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 -DBUILD_JAVA=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$PWD/install ..
        
        make install -j32
        ```

1. Export `OPENCV_INSTALL_INCLUDE_PATH ` and `OPENCV_INSTALL_LIB_PATH`

    * Native building
        ```sh
        export OPENCV_INSTALL_INCLUDE_PATH=${opencv_root}/build/install/include

        export OPENCV_INSTALL_LIB_PATH=${opencv_root}/build/install/lib
        ```
    * Cross building for Android AArch64
        ```sh
        export OPENCV_INSTALL_INCLUDE_PATH=${opencv_root}/build_android/install/sdk/native/jni/include

        export OPENCV_INSTALL_LIB_PATH=${opencv_root}/build_android/install/sdk/native/libs/arm64-v8a
        ```

### Step4: Build Test Demo
* Native building
    ```sh
    export CXX=g++

    ./build.sh
    ```
* Cross building for Android AArch64
    ```sh
    export PATH=${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/:$PATH

    export CXX=aarch64-linux-android21-clang++

    ./build.sh
    ```

### Step5: Run Demo

> **Note**: Two ways to get `yolox_s.mge` model file
>
> 1. Reference python demo's `dump.py` script.
> 1. wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.mge

* Native building
    ```sh
    LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib/:$OPENCV_INSTALL_LIB_PATH ./yolox yolox_s.mge ../../../assets/dog.jpg cuda/cpu <warmup_count> <thread_number> <run_with_fp16>
    ```
* Cross building for Android
    ```sh
    adb push/scp $MGE_INSTALL_PATH/lib/libmegengine.so android_phone

    adb push/scp $OPENCV_INSTALL_LIB_PATH/*.so android_phone

    adb push/scp ./yolox yolox_s.mge ../../../assets/dog.jpg android_phone
    
    # Execute the following cmd after logging in android_phone by adb or ssh
    
    # <warmup_count> means warmup count, valid number >=0
    # <thread_number> means thread number, valid number >=1, only take effect on `cpu` device
    # <run_with_fp16> if >=1, will run with fp16 mode
    
    LD_LIBRARY_PATH=. ./yolox yolox_s.mge dog.jpg cpu <warmup_count> <thread_number>  <run_with_fp16>
    ```

## Benchmark

* Model Info: yolox-s @ input(1,3,640,640)

* Testing Devices

  * x86_64 -- Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
  * AArch64 -- xiamo phone mi9
  * CUDA -- 1080TI @ cuda-10.1-cudnn-v7.6.3-TensorRT-6.0.1.5.sh @ Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz

| megengine @ tag1.5(fastrun + weight\_preprocess)/sec | 1 thread | 2 thread | 4 thread | 8 thread |
| ---------------------------------------------------- | -------- | -------- | -------- | -------- |
| x86\_64(fp32)                                        | 0.516245 | 0.31829  | 0.253273 | 0.222534 |
| x86\_64(fp32+chw88)                                  | 0.362020 |   NONE   |   NONE   |   NONE   |
| aarch64(fp32+chw44)                                  | 0.555877 | 0.351371 | 0.242044 |   NONE   |
| aarch64(fp16+chw)                                    | 0.439606 | 0.327356 | 0.255531 |   NONE   |

| CUDA @ 1080TI/sec   | 1 batch    | 2 batch   | 4 batch   | 8 batch   | 16 batch  | 32 batch | 64 batch |
| ------------------- | ---------- | --------- | --------- | --------- | --------- | -------- | -------- |
| megengine(fp32+chw) | 0.00813703 | 0.0132893 | 0.0236633 | 0.0444699 | 0.0864917 | 0.16895  | 0.334248 |

## Acknowledgement

* [MegEngine](https://github.com/MegEngine/MegEngine)
* [OpenCV](https://github.com/opencv/opencv)
* [NDK](https://developer.android.com/ndk)
* [CMAKE](https://cmake.org/)
