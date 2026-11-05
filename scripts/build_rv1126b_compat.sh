#!/bin/bash
# Build isolated aarch64 compatibility tests using the existing Rockchip SDK.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${OPENCV_SOURCE:?Set OPENCV_SOURCE to an extracted official OpenCV source directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to an isolated writable build directory}"
RK_TOOLCHAIN=${RK_TOOLCHAIN:-$HOME/Dev/RK1126B/rockchip_rv1126b_toolchain}
IMAGE=${IMAGE:-gcc:13-bookworm}
[ -f "$OPENCV_SOURCE/CMakeLists.txt" ]
[ -x "$RK_TOOLCHAIN/bin/aarch64-buildroot-linux-gnu-g++" ]
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
docker run --rm -i --platform linux/amd64 \
    -v "$RK_TOOLCHAIN:/toolchain:ro" -v "$OPENCV_SOURCE:/opencv:ro" \
    -v "$ROOT:/project:ro" -v "$OUTPUT_DIR:/work" \
    -e "JOBS=${JOBS:-4}" "$IMAGE" bash -s <<'BUILD'
set -euo pipefail
CMAKE=/toolchain/bin/cmake
TOOLCHAIN=/toolchain/share/buildroot/toolchainfile.cmake
"$CMAKE" -S /opencv -B /work/opencv-build \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_INSTALL_PREFIX=/work/opencv-install -DCMAKE_BUILD_TYPE=Release \
    -DOPENCV_DOWNLOAD_PATH=/work/downloads \
    -DPYTHON3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DOPENCV_PYTHON3_VERSION:STRING=3.11 \
    -DBUILD_LIST=calib3d,imgcodecs -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_gapi=OFF -DWITH_ADE=OFF \
    -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_TBB=OFF \
    -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DWITH_PROTOBUF=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
    -DWITH_CUDA=OFF -DWITH_V4L=OFF -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=OFF -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_OPENEXR=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF -DOPENCV_GENERATE_PKGCONFIG=OFF
"$CMAKE" --build /work/opencv-build --parallel "$JOBS"
"$CMAKE" --install /work/opencv-build
mkdir -p /work/runtime
# Link only built modules: OpenCV's minimal install can export unused codec targets.
/toolchain/bin/aarch64-buildroot-linux-gnu-g++ -std=c++17 -O2 \
    -I/work/opencv-install/include/opencv4 /project/cpp/rv1126b/compatibility_test.cpp \
    -o /work/runtime/compatibility_test -static-libstdc++ -static-libgcc \
    -Wl,--start-group /work/opencv-install/lib/libopencv_calib3d.a \
    /work/opencv-install/lib/libopencv_features2d.a \
    /work/opencv-install/lib/libopencv_flann.a /work/opencv-install/lib/libopencv_imgproc.a \
    /work/opencv-install/lib/libopencv_core.a \
    /work/opencv-install/lib/opencv4/3rdparty/libtegra_hal.a \
    /work/opencv-install/lib/opencv4/3rdparty/libzlib.a -Wl,--end-group -ldl -lm -pthread -lrt
/toolchain/bin/aarch64-buildroot-linux-gnu-strip /work/runtime/compatibility_test
/toolchain/bin/aarch64-buildroot-linux-gnu-g++ -std=c++17 -O2 \
    -I/work/opencv-install/include/opencv4 /project/cpp/rv1126b/calibration_service.cpp \
    -o /work/runtime/calibration_service -static-libstdc++ -static-libgcc \
    -Wl,--start-group /work/opencv-install/lib/libopencv_calib3d.a \
    /work/opencv-install/lib/libopencv_features2d.a /work/opencv-install/lib/libopencv_flann.a \
    /work/opencv-install/lib/libopencv_imgcodecs.a /work/opencv-install/lib/libopencv_imgproc.a \
    /work/opencv-install/lib/libopencv_core.a \
    /work/opencv-install/lib/opencv4/3rdparty/libtegra_hal.a \
    /work/opencv-install/lib/opencv4/3rdparty/libzlib.a \
    /work/opencv-install/lib/opencv4/3rdparty/liblibjpeg-turbo.a \
    /work/opencv-install/lib/opencv4/3rdparty/liblibjasper.a \
    -Wl,--end-group -ldl -lm -pthread -lrt
/toolchain/bin/aarch64-buildroot-linux-gnu-strip /work/runtime/calibration_service
/toolchain/bin/aarch64-buildroot-linux-gnu-readelf -h -d /work/runtime/compatibility_test \
    > /work/runtime/elf_report.txt
sha256sum /work/runtime/compatibility_test > /work/runtime/SHA256SUMS.txt
if [ ! -s /work/GCC_COPYING.RUNTIME ]; then
    curl -fL --connect-timeout 10 --max-time 60 \
        https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-13.3.0/COPYING.RUNTIME \
        -o /work/GCC_COPYING.RUNTIME
fi
{
    cat /opencv/LICENSE
    cat /project/cpp/rv1126b/vendor/LICENSE.httplib
    find /work/opencv-install/share/licenses/opencv4 -type f -exec cat {} \;
    sed -n '1,80p' /opencv/modules/flann/include/opencv2/flann/general.h
    sed -n '1,60p' /opencv/3rdparty/carotene/src/common.hpp
    cat /work/GCC_COPYING.RUNTIME /usr/share/common-licenses/GPL-3
} > /work/runtime/THIRD_PARTY_NOTICES.txt
BUILD
