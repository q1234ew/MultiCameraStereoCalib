# MultiCamera 技术文档

## 1. 项目概览

MultiCamera 是一个 PySide6 桌面应用，用于多组双目相机的 ChArUco 采集、内参标定、双目标定、联合外参构建、标定结果导出和点云查看。

当前入口：

```bash
source .venv/bin/activate
python -m multicamera
```

如果不激活虚拟环境，也可以直接运行：

```bash
.venv/bin/python -m multicamera
```

主要运行链路：

1. 配置 ChArUco 标定板、相机模型和双目相机组。
2. 连接 MJPEG 相机流或导入离线左右图。
3. 按采集引导保存双目共视图像。
4. 对每组双目相机计算左右内参和双目外参。
5. 构建 `MultiCameraRig`。
6. 导出 JSON、OpenCV YAML 或 Kalibr camchain YAML。
7. 可选生成融合点云并在 3D 点云页查看。

## 2. 模块结构

- `streaming/`
  - `mjpeg_grabber.py`：每路 MJPEG 一个 QThread，负责 HTTP 读取、JPEG 解码和 FPS 信号。
  - `stream_manager.py`：管理相机组、软同步帧、虚拟离线相机组和 grabber 生命周期。
  - `discovery.py`：mDNS/Zeroconf 和局域网 HTTP 探测。

- `board/`
  - `charuco_board.py`：ChArUco 标定板参数、生成和序列化。
  - `detector.py`：ChArUco 角点检测和 overlay 绘制。

- `calibration/`
  - `intrinsic.py`：针孔/鱼眼单目标定。
  - `stereo.py`：针孔/鱼眼双目标定、校正映射和 Q 矩阵。
  - `multiview.py`：多组双目的全局外参图构建。
  - `planar.py`：通用平面板检测，支持棋盘格和圆点阵。
  - `auxiliary.py`：辅助单目内参和 AUX→RGB_L 跨模态外参求解。
  - `assessment.py`：采集数据和标定质量评估。
  - `models.py`：内参、双目标定结果、相机外参、rig 数据模型。

- `ui/`
  - `main_window.py`：主窗口、菜单、工具栏、导出入口、点云 viewer 懒加载。
  - `widgets/calib_panel.py`：采集、导入、标定、联合优化、点云生成控制。
  - `widgets/stream_view.py`：多路预览、检测 overlay、覆盖率热力图。
  - `threads/`：后台标定和图片保存 worker。

- `io/`
  - `session.py`：会话目录、图片和 rig 保存。
  - `export.py`：JSON、OpenCV YAML、Kalibr camchain YAML、点云导出。

## 3. 数据模型与会话

核心数据类型在 `calibration/models.py`：

- `CameraIntrinsics`
  - `camera_matrix`、`dist_coeffs`、`image_size`、`model`、`rms_error`。
- `StereoCalibResult`
  - `R`、`T`、`E`、`F`、`Q`、`R1/R2/P1/P2`、校正 map。
- `StereoPairCalibration`
  - 一组左右相机的完整内参和双目标定结果。
- `MultiCameraRig`
  - 多组双目标定结果、全局外参、辅助单目相机和参考相机。
- `AuxCameraCalibration`
  - 辅助相机 ID、模态、单目标定内参、`T_rgb_left_aux`、RMS、板型和参与帧数。

会话目录由 `runtime_paths.sessions_dir()` 决定：

- 开发环境：当前工作目录下的 `sessions/`。
- PyInstaller 打包后：exe 同级的 `sessions/`。

日志目录由 `runtime_paths.logs_dir()` 决定：

- 开发环境：当前工作目录下的 `logs/`。
- PyInstaller 打包后：exe 同级的 `logs/`。
- 主日志文件为 `multicamera.log`，采用 5MB 轮转并保留 5 个历史文件。
- UI 菜单 `工具 -> 打开日志目录` 可直接打开日志位置。

会话结构：

```text
sessions/<session_name>/
  session.json
  images/<pair_name>/left_0000.png
  images/<pair_name>/right_0000.png
  multimodal/rgb_left_0000.png
  multimodal/aux_0000.png
  results/rig.json
```

图片保存由后台队列串行执行，避免多个线程同时写 `session.json`。保存失败不会推进 metadata，并会在 UI 日志中提示。

## 4. 相机流与同步

`StreamManager` 管理所有双目相机组：

- `add_stereo_pair()`：注册真实 MJPEG 双目组，并创建 grabber。
- `add_virtual_stereo_pair()`：注册离线导入的虚拟双目组，不创建 grabber。
- `set_auxiliary_camera()`：注册融合辅助单目相机，辅助流与普通相机流一样通过 MJPEG grabber 访问。
- `get_sync_pair()`：返回时间戳偏差在 `DEFAULT_SYNC_TOLERANCE = 0.35s` 内的左右帧。
- `get_latest_pair_relaxed()`：严格同步失败时返回左右最新帧，用于采集回退。

发现服务：

- mDNS TXT 建议发布 `type/modality/camera_type` 和 `eye/side/role`。
- 支持识别 `rgb`、`ir/thermal/热成像`、`acoustic_rgb/声像仪`。
- 同一台多 sensor 设备可以在同一 host:port 下暴露多个 path；服务唯一键包含 `host:port/path`，不同数据源不会互相覆盖。
- 双目自动配对只处理 `rgb` 和 `ir` 的左右目；融合辅助单目从发现服务列表中单独选择。

`MJPEGGrabber` 默认参数：

- `READ_TIMEOUT = 10.0s`
- `RECONNECT_DELAY = 2.0s`
- `FPS_UPDATE_INTERVAL = 1.0s`
- `DEFAULT_MAX_DECODE_FPS = 20.0`

解码限频在 JPEG 解码前生效。过密帧会被跳过，避免 UI 只显示 15fps 时后台仍无限解码。

## 5. 采集与检测

采集按钮流程：

1. 从 `StreamManager` 获取同步帧，失败时尝试 relaxed 最新帧。
2. 使用原始帧分别更新左右 `IntrinsicCalibrator`。
3. 使用原始帧更新 `StereoCalibrator`。
4. 评估 sharpness、brightness、角点数等质量指标。
5. 将 PNG 保存任务提交到后台串行队列。
6. 更新采集引导、撤销栈、进度条和 UI 日志。

预览检测和采集检测是两条路径：

- 预览检测为轻量路径，限频并可缩小图像。
- 采集/标定检测使用原始帧，确保标定精度不受预览优化影响。

撤销上一帧：

- 标定缓冲立即回退。
- 如果图片保存还在 pending，保存任务完成后删除对应文件。
- 如果图片已保存，立即删除左右 PNG，并重算 `frame_count`。

## 6. UI 预览性能

`MultiStreamView` 默认参数：

- `UI_FPS_CAP = 15`
- `DETECT_FPS_CAP = 2.0`
- `DETECT_MAX_WIDTH = 960`

预览渲染策略：

- 每路相机最多 15fps 更新 UI。
- ChArUco overlay 每路最多 2fps 检测。
- 预览宽度大于 960px 时，缩小后检测，再把角点坐标映射回原图。
- `CameraView` 缓存当前 label 尺寸下的 pixmap，避免尺寸未变时重复平滑缩放。

如果预览 CPU 偏高：

- 先降低 `DETECT_FPS_CAP`。
- 再降低 `UI_FPS_CAP`。
- 如果仍偏高，降低 `DETECT_MAX_WIDTH`。

## 7. 标定线程模型

单组双目标定由 `PairCalibrationWorker` 在后台 QThread 中执行。

worker 输入：

- pair name、left/right camera id。
- 左右 `IntrinsicCalibrator`。
- `StereoCalibrator`。
- 已存在的左右内参缓存，可避免重复计算。

worker 输出：

- UI 日志信号。
- 左右内参。
- `StereoPairCalibration`。
- 失败错误信息。

UI 主线程负责：

- 禁用采集、导入、撤销、自动采集和标定按钮。
- 接收 worker 结果。
- 更新 `_intrinsics` 和 `_pair_calibs`。
- 生成 rectification preview。
- 刷新按钮状态和步骤状态。

“一键标定全部”使用串行队列，不并行跑多个 pair。这样可以避免 OpenCV 标定同时抢 CPU，也能让 UI 日志和状态更可预测。

图片导入由 `StereoImageImportWorker` 和 `AuxImageImportWorker` 在后台 QThread 中执行：

- 负责目录扫描、`cv2.imread()` 和标定板检测。
- 主线程只接收已加载图像和 detection result，并更新 calibrator / UI 状态。
- 离线双目导入会复用左右 ChArUco 检测结果，避免内参和双目缓冲各自重复检测同一帧。
- 多模态导入会复用 RGB_L/AUX 的平面板检测结果，后续辅助内参和跨模态外参直接使用缓存观测。

## 8. 点云与 Open3D 懒加载

启动路径不再导入 Open3D。

Windows 打包默认也不包含 Open3D：

- `scripts\build_windows.bat` 生成轻量包，保留标定、导出和多模态功能。
- `scripts\build_windows.bat --pointcloud` 生成点云完整包，会安装并打包 Open3D。
- 轻量包中点击点云生成/PLY/PCD 导出会提示需要 `pointcloud` extra 或 `--pointcloud` 包。

懒加载策略：

- 主窗口启动时只创建 3D 点云占位页。
- 第一次收到 `pointcloud_ready` 信号时，才导入并创建 `PointCloudViewer`。
- `CalibrationPanel` 顶部不导入 `fusion`、`pointcloud`、`stereo_matching`。
- 点击“生成融合点云”时才导入 `MultiViewFusion`、`depth_to_pointcloud`、`StereoMatcher`。
- 普通 JSON/OpenCV/Kalibr 标定导出不加载 Open3D。
- 只有 PLY/PCD 点云导出调用 `_require_open3d()`。
- SciPy 也按需加载：`least_squares` 只在多视图 bundle adjustment 时导入，`Rotation` 只在跨模态外参聚合时导入。

当前实测改善：

- 入口导入：约 `3.98s / 330MB` 降到约 `1.42s / 170MB`。
- GUI offscreen 烟测：约 `5.19s / 382MB` 降到约 `2.60s / 223MB`。

## 9. Windows 打包优化

默认构建目标是“标定工具轻量包”：

```bat
scripts\build_windows.bat
```

该包不安装、不收集 Open3D，适合现场采集、标定、JSON/OpenCV/Kalibr 导出，打包速度和分发体积都明显低于完整点云包。

需要点云功能时使用：

```bat
scripts\build_windows.bat --pointcloud
```

对应安装包脚本会透传参数：

```bat
scripts\build_windows_installer.bat --pointcloud
```

PyInstaller 优化原则：

- 不再对 `PySide6` 做全量 `collect_all()`，交给 PyInstaller Qt hooks 按实际导入收集。
- 默认排除 `open3d`、`torch`、`tensorflow`、`matplotlib`、`notebook` 等大依赖。
- `aiohttp`、`zeroconf` 仅收集子模块，避免把无关数据文件打进去。
- OpenCV 动态库和数据仍显式收集，保证 ChArUco/aruco 能正常工作。

## 10. 导出格式

标定完成后支持：

- JSON
  - 完整 `MultiCameraRig.to_dict()`。
  - 适合程序内部恢复和调试。

- OpenCV YAML
  - 由 `cv2.FileStorage` 写出。
  - 矩阵节点采用稳定扁平 key，便于 OpenCV C++/Python 读取。
  - 包含 camera matrix、distortion、pair R/T/E/F/Q、rectification 矩阵和全局外参。
  - 启用辅助单目时额外包含 AUX 内参、模态、板型、`T_rgb_left_aux`、`R_rgb_left_aux`、`t_rgb_left_aux`、帧数和 RMS。

- Kalibr camchain YAML
  - 生成 `cam0/cam1/...`。
  - 包含 `camera_model`、`distortion_model`、`intrinsics`、`resolution`、`rostopic`、`cam_overlaps`。
  - 从第二个相机开始写 `T_cn_cnm1`。
  - 针孔模型使用 `radtan`；鱼眼模型使用 `equidistant`。
  - AUX 会作为追加相机写入；该扩展用于融合标定结果交换，不声明等同 Kalibr 原生链式采集流程。

注意：Kalibr 导出的 `rostopic` 默认使用 `/<camera_id>/image_raw`，如需对接真实 ROS bag，可在后续增加可配置映射。

## 11. 多模态融合单目标定

第一版支持固定设备形态 `RGB_L/R + AUX`：

- `RGB 双目 + IR 单目`
- `RGB 双目 + 声像仪 RGB 单目`

参考相机固定为 `RGB_L`。多模态流程只生成可供融合使用的标定结果，不实现点云融合、声学融合或算法级多模态融合。

核心流程：

1. 先按现有入口完成 RGB 双目标定，得到 `RGB_L`、`RGB_R` 内参和双目外参。
2. 按多模态采图引导实时采集 `RGB_L/AUX` 配对观测，或导入辅助单目配对图像，目录命名为 `rgb_left_XXXX.*` 与 `aux_XXXX.*`。
3. 使用 `PatternBoardConfig` 和 `PlanarPatternDetector` 检测平面板点。
4. 用 AUX 图像检测结果执行单目标定，得到辅助相机内参。
5. 对每组 `RGB_L/AUX` 配对观测分别 `solvePnP`，得到板到两路相机的位姿。
6. 计算每帧 `T_rgb_left_aux = T_rgb_left_board @ inv(T_aux_board)`。
7. 按重投影误差筛除异常帧，对旋转和位移聚合，记录参与帧数和 RMS。
8. 将结果写入 `MultiCameraRig.aux_cameras` 并触发原有导出入口。

支持的平面板：

- `chessboard`：OpenCV `findChessboardCorners`，角点数来自 ChArUco 设置的 `cols-1`、`rows-1`。
- `circles_grid`：OpenCV `findCirclesGrid`，第一版默认对称圆点阵。

UI 和线程模型：

- 配置窗口新增“融合辅助单目”，可选择不启用、`RGB+IR` 或 `RGB+声像仪RGB`。
- 可从发现服务下拉选择热成像/声像仪 RGB 辅助流，也可手动设置辅助相机 ID、URL、平面板类型。
- `StreamManager` 保留 `AuxiliaryCameraConfig`，实时流会随 `start_all()` 创建 MJPEG grabber。
- 标定面板新增“多模态单目标定”步骤；AUX 内参和跨模态外参由 `AuxCalibrationWorker` 后台执行。
- 多模态步骤有独立采图引导，覆盖视场九宫格和中心倾角；“采集多模态一帧”会取 `RGB_L` 与 AUX 最新帧、检测平面板并写入标定缓存。
- 建议按引导采满 13 组；至少 3 组有效配对观测即可启用辅助单目标定。

失败条件：

- 没有完成 RGB_L 所在双目标定。
- 配对目录缺少同编号 `rgb_left_XXXX` 与 `aux_XXXX`。
- 有效平面板配对观测少于 3 组。
- AUX 内参或任一外参 `solvePnP` 有效帧不足。

## 12. 性能观测

`multicamera.perf.perf_timer()` 用于记录慢操作。

应用启动时会写入运行环境信息，包括 frozen 状态、exe 路径、Python 版本、平台、工作目录和进程 ID。日志同时捕获：

- 未处理 Python 异常和线程异常。
- Qt warning/critical/fatal 消息。
- mDNS/子网扫描发现到的服务、类型、目别和 URL。
- 每路 MJPEG 连接 URL、HTTP 状态、Content-Type、断开原因和 JPEG 解码失败。
- 用户触发的配置、连接、断开、导出、新建/加载会话等关键操作。

已接入位置：

- `MJPEGGrabber._decode_and_emit()`
- `CharucoDetector.detect()`
- `IntrinsicCalibrator.add_frame()`
- `IntrinsicCalibrator.calibrate()`
- `StereoCalibrator.add_frame_pair()`
- `StereoCalibrator.add_detection_pair()`
- `StereoCalibrator.calibrate()`
- `CalibrationSession.save_frame_pair()`
- `MultiStreamView` 预览检测和渲染
- `CalibrationPanel` 采集按钮流程

默认日志只在超过阈值时输出，便于发现异常慢操作，而不会在正常帧率下刷屏。

## 13. 验证步骤

推荐使用项目虚拟环境：

```bash
.venv/bin/python -m compileall -q src scripts
```

离线标定验证：

```bash
.venv/bin/python scripts/test_offline_calib.py
```

多模态导出最小验证：

```bash
.venv/bin/python -c "import tempfile, numpy as np; from pathlib import Path; from multicamera.calibration.models import CameraIntrinsics, CameraModel, StereoCalibResult, StereoPairCalibration, MultiCameraRig, AuxCameraCalibration; from multicamera.io.export import export_rig_json, export_rig_opencv_yaml, export_rig_kalibr_yaml; K=np.eye(3); d=np.zeros((1,5)); intr=CameraIntrinsics(K,d,(640,480),CameraModel.PINHOLE,0.1); stereo=StereoCalibResult(np.eye(3),np.array([[0.1],[0],[0.]]),np.eye(3),np.eye(3),0.2); pair=StereoPairCalibration('rgb_pair','RGB_L','RGB_R',intr,intr,stereo); T=np.eye(4); aux=AuxCameraCalibration('AUX','ir',intr,T,0.3,'chessboard',5); rig=MultiCameraRig(pairs={'rgb_pair':pair}, aux_cameras={'AUX':aux}, reference_camera='RGB_L'); tmp=Path(tempfile.mkdtemp()); export_rig_json(rig,tmp/'rig.json'); export_rig_opencv_yaml(rig,tmp/'rig.yml'); export_rig_kalibr_yaml(rig,tmp/'camchain.yaml'); print(tmp)"
```

GUI 启动烟测：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c "import sys; from PySide6.QtCore import QTimer; from PySide6.QtWidgets import QApplication; sys.path.insert(0, 'src'); from multicamera.ui.main_window import MainWindow; app = QApplication([]); w = MainWindow(); w.show(); QTimer.singleShot(1000, app.quit); app.exec(); print('gui smoke ok')"
```

可选检查，需要开发依赖：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src scripts
```

当前环境中如果 `pytest` 或 `ruff` 未安装，应先安装 dev 依赖：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

## 14. 手工回归清单

相机配置：

- mDNS/子网扫描能发现设备。
- 自动配对能生成左右 ID 和 URL。
- 手动添加相机组后保存，右相机 ID 不被自动改写。

实时采集：

- 多路 MJPEG 连接后预览稳定刷新。
- FPS 显示正常。
- ChArUco overlay 低频更新但不阻塞 UI。
- 点击“采集一帧”后进度、覆盖率、质量提示正常更新。

离线导入：

- 导入 `*_left.png/jpg` 与 `*_right.png/jpg` 后出现 `offline` 虚拟相机组。
- 离线帧可前后浏览。
- 离线标定完成后可生成 rectification preview。

标定：

- 单组标定期间 UI 不冻结。
- “一键标定全部”按组串行执行。
- 标定失败时 UI 日志有错误提示，按钮能恢复。

会话与保存：

- 新建会话后采集图片写入 `sessions/<session>/images/<pair>/`。
- 快速撤销 pending 保存帧后，不残留对应图片。
- 保存失败不推进 metadata。

导出：

- JSON 导出可重新反序列化。
- OpenCV YAML 可用 `cv2.FileStorage` 读回核心矩阵。
- Kalibr YAML 包含 `cam0/cam1`、内参、畸变、分辨率、overlaps 和 `T_cn_cnm1`。
- 启用辅助单目后，JSON/OpenCV/Kalibr 导出包含 AUX 内参和 `T_rgb_left_aux`。

多模态：

- 配置窗口可启用 `RGB+IR` 或 `RGB+声像仪RGB`。
- 导入 `rgb_left_XXXX.*` 与 `aux_XXXX.*` 后有效配对数正确显示。
- 未完成 RGB 双目标定时点击辅助单目标定，应提示先完成 RGB_L。
- 辅助单目标定期间按钮禁用，完成后恢复并启用导出。

## 15. 当前边界与后续方向

当前边界：

- 点云生成、视差、Open3D 融合本身尚未做性能优化。
- 多模态功能仅产出内参和跨模态外参，不包含点云/声学融合算法。
- 性能参数还是代码常量，尚未提供 UI 配置。
- 没有正式单元测试覆盖线程 worker 和导出格式。

建议后续优先级：

1. 给导出格式增加自动化测试。
2. 将性能参数暴露到配置 UI。
3. 为 `FrameSaveWorker`、`PairCalibrationWorker` 和导入 worker 增加最小单元测试。
4. 为 `AuxCalibrationWorker`、平面板检测和跨模态外参增加正式单元测试。
5. 如果需要继续降启动内存，可继续拆分 UI 模块导入路径。
