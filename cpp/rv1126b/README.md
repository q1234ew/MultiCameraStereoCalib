# RV1126B isolated compatibility adapter

This is a candidate-file adapter and offline test, not a streaming service or an
activation tool. It does not modify the RV1126B project or its running processes.

## Export

Run `python scripts/export_rv1126b.py <rig.json> <new-directory> --pair <name>
--aux <id> --device-id <stable-label> --endpoint <address>` from the MultiCamera
checkout. Existing output directories are refused. The source file is unchanged.

The bundle contains `calibration_mm.yml`, `test_vectors.yml`, an exact source
copy and a SHA256 manifest. The selected auxiliary result must explicitly bind
to the RGB pair. Device identity remains unverified; an IP or an offline camera
name does not establish physical lens identity.

## Contract

- `camera_intrinsics.left/right/ir` each contains K, D, model, width and height.
- `extrinsics.T_RL`: X_right = R * X_left + t.
- `extrinsics.T_IR_L`: X_left = R * X_ir + t (not inverted).
- Translations and optical-axis depth are millimetres. R1/R2/P1/P2/Q are
  recomputed together in millimetres, not copied from metre-based input.
- Pinhole distortion retains all coefficients; fisheye retains its four angular
  coefficients and uses OpenCV fisheye projection/unprojection.
- `irToRgb` requires IR optical-axis Z; `rgbToIr` requires RGB optical-axis Z.
  Depth must match the unrectified input camera, not Euclidean ray length.
  Rectified stereo depth must first be transformed by R1 transpose into RGB-left
  coordinates. Temperature/palette values are not depth measurements.
- Out-of-image projected pixels need masking by the caller; occlusion handling,
  depth acquisition, thermal measurement and a live image overlay are not provided.

The legacy IrLeftProjector can read the nested K/D/R/t fields, but silently treats
IR as pinhole and projects at fixed Z=1. DO NOT use its legacy projection with
these files. Integrate the model-aware adapter explicitly first. This adapter
does not make historical candidate calibrations physically verified.

## Build and validate

```sh
cmake -S cpp/rv1126b -B build/rv1126b-compat
cmake --build build/rv1126b-compat
build/rv1126b-compat/compatibility_test <candidate>/calibration_mm.yml <candidate>/test_vectors.yml
```

Alternatively compile `compatibility_test.cpp` using C++17 and
`pkg-config --cflags --libs opencv4`. Tests compare 27 Python-generated rays at
500/1000/2000 mm, both transform directions, model-specific projection and Q depth
scale. They establish numerical compatibility only, not empirical calibration
accuracy. A macOS binary cannot run on the aarch64 board; build on the board or
with a matching aarch64 OpenCV toolchain. Do not upload a host binary as a device
executable.

## Candidate-only device layout (proposed)

Use `/opt/fusion-sensor/calibration-candidates/<device-label>/<unique-version>/`
for a new, isolated directory. Verify identity, writable persistent storage,
available OpenCV and existing configuration read-only before uploading. Keep
source hashes and verification results alongside the candidate. Never change
current symlinks, startup environment, services, USB controls or active files as
part of this test. No device activation is authorized by a successful self-test.

## Offline installer archives

Export with `--device-mac <expected-MAC>`, then run
`python scripts/package_rv1126b_candidate.py <bundle> <output-directory> --version <unique-version>`.
The default archive contains sources, not an executable or OpenCV runtime. Add
`--runtime-binary <aarch64-compatibility_test>` to the packager to include the
cross-compiled numeric test and its adjacent `THIRD_PARTY_NOTICES.txt`. Non-aarch64
ELF or Mac binaries are rejected. The runtime package does not activate a service.
Extract
to a temporary directory on the intended device, enter the extracted directory
and run `sh install.sh`. It verifies the expected network MAC, checks SHA256,
creates a new candidate version and checks the installed copy. A repeated version
or mismatched MAC is refused. No current symlink, service or active configuration
is changed. An interrupted installation remains incomplete, never automatically
deleted or reused. Hashes protect transfer integrity, not publisher authenticity.

An optional absolute candidate-root argument supports staging; the default is
`/opt/fusion-sensor/calibration-candidates`. Network MAC matching does not verify
the attached lenses or historical left/right binding. Do not activate based on
MAC matching alone.

## Existing Rockchip cross compiler

`scripts/build_rv1126b_compat.sh` runs the existing Buildroot GCC 13.3 SDK in an
amd64 Linux Docker image. Set `OPENCV_SOURCE` to extracted OpenCV 4.11.0 sources,
`OUTPUT_DIR` to an isolated build directory, and optionally `RK_TOOLCHAIN` and
`JOBS`. It mounts project/SDK/source read-only, builds static OpenCV dependencies
and the aarch64 numeric test, and records ELF dependencies. No board compiler or
OpenCV installation is needed for that test binary; base glibc/libm compatibility
must still be tested on each board. Original runtime services are not modified.

## Optional isolated calibration service

The same build also produces `calibration_service`, linking imgcodecs and bundled
JPEG dependencies. HTTP uses the pinned MIT-licensed header in `vendor/`.
`scripts/package_rv1126b_service.py` prepares a separate device-bound payload,
optionally integrating the exact current device dashboard with an original-byte
backup. This is separate from the immutable candidate installer above.

Service installation creates its own version directory under
`/opt/fusion-sensor/calibration-services` and `S92multicamera-calibration` init entry.
It never starts or rewrites the raw camera services. The default listener is
port 8090 on a trusted camera LAN; there is no authentication/TLS. Do not expose
it to the Internet. `/health` means parameters loaded, not camera readiness.
`/api/sources` checks original JPEG availability and exact calibrated dimensions.

`/dashboard` exposes the device dashboard with the new calibration entry;
`/panel` is the standalone view. `/api/calibration` returns the exact loaded mm
YAML, `/api/rectified.jpg?camera=left|right` returns independent corrected
snapshots. These requests are not synchronized stereo pairs for moving scenes.
`/api/project?from=rgb|ir&u=...&v=...&depth_mm=...` requires raw source pixels and
optical-axis depth in mm. `/api/fusion.jpg?depth_mm=...` is an explicitly marked
constant RGB-depth plane validation, not real scene depth fusion, temperature
measurement or an occlusion-aware reconstruction. Physical accuracy remains
unverified even after the service loads candidate parameters.

Run `sh <installed-service-directory>/rollback.sh` to disable only this service
and restore the original dashboard, provided neither has been changed since
installation. Payloads and candidate files remain. Existing startup entries or
versions are not overwritten; review/rollback before installing a replacement.

Native HTTP/image regression tests are opt-in:

```sh
MULTICAMERA_SERVICE_TEST_BINARY=/absolute/path/calibration_service \
  python -m pytest -q tests/test_rv1126b_service.py
```

Without the environment variable, portable packaging tests still run and only
native executable tests skip. This is not a hardware accuracy or Windows EXE test.
