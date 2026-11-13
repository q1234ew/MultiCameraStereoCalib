# MultiCamera Stereo Calibration

MultiCamera 是一个多相机标定工具，基于 Python、PySide6 和 OpenCV。支持多组双目相机的实时采集和离线标定，也可标定 RGB 左目与红外、声像仪 RGB 等辅助单目之间的外参。

[Windows 下载](https://github.com/q1234ew/MultiCameraStereoCalib/releases) · [构建状态](https://github.com/q1234ew/MultiCameraStereoCalib/actions/workflows/build-windows.yml) · [技术文档](docs/TECHNICAL_DOCUMENTATION.md) · [更新记录](docs/CHANGELOG.md)

## 功能

- **采集与预览**：相机发现、MJPEG 多路预览、左右图像源交换、RGB 参数调整，以及支持设备的采图前白平衡检查。
- **双目标定**：ChArUco 标定板，针孔和鱼眼模型；支持实时采图、离线图片导入、采集引导、逐对复核和数据筛选。
- **辅助相机标定**：独立配置棋盘格或圆点阵，计算辅助单目内参及其相对 RGB 左目的外参，支持原图复核、联合细化和留出验证。
- **会话与导出**：保存原图、恢复会话，导出 JSON、OpenCV YAML 和 Kalibr YAML，并记录结果文件的 SHA-256。
- **点云**：可选安装 Open3D，用于点云查看与融合。默认 Windows 轻量包不包含 Open3D。

## 安装与运行

### Windows 下载

在 [Releases](https://github.com/q1234ew/MultiCameraStereoCalib/releases) 中选择版本，预发布版本也在此列出。常用附件为：

| 文件 | 用途 |
| --- | --- |
| `MultiCamera_Setup_<version>_x64.exe` | Windows x64 安装包 |
| `MultiCamera_portable_<version>_x64.zip` | 便携版，完整解压后运行 `MultiCamera.exe` |
| `SHA256SUMS.txt` | 下载文件校验值 |
| `calibration_source_test.json`、`calibration_self_test.json` | 源码及最终 EXE 的标定自检报告 |

安装器面向 Windows 10 及以上 x64。程序默认在 EXE 同级写入 `sessions/` 和 `logs/`，请使用可写目录。安装目录受限时，可选择便携版。早期版本的附件命名可能不同，以对应发布说明为准；发布说明也会注明构建提交及是否包含 Open3D。

### 从源码运行

需要 Python 3.10 或以上。克隆仓库后，在仓库根目录执行：

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m multicamera
```

**Windows PowerShell**（Windows 构建使用 Python 3.11）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c packaging/windows/constraints.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m multicamera
```

需要点云功能时，在同一虚拟环境中额外执行 `python -m pip install -e ".[pointcloud]"`。不同平台的相机驱动和桌面环境需分别验证。

## 双目标定

1. 在“相机配置”中设置左右相机、镜头模型和 ChArUco 板参数。板尺寸应与实物一致：界面长度单位为 **mm**，内部计算和结果平移使用 **m**。
2. 连接视频流，在“采集”页创建会话；或直接导入离线左右图片。采图前核对左右角色，必要时交换来源。
3. 采集覆盖画面不同位置、距离和倾斜方向的图像。可参考自适应引导补拍，再逐对检查、排除不合适的图片。
4. 在“标定”页选择策略，执行单组或全部标定。计算在后台进行。
5. 在“结果”页检查质量提示、逐帧误差和校正残差，再查看自动保存结果或手动导出。

“自动校验择模”会检查模型有效性，并在数据充足时比较候选模型的留出误差。采集引导和低重投影误差可辅助判断，实际精度仍需用独立图像或已知尺寸、距离验证。

### 离线图片

导入时会递归查找并配对，例如：

```text
images/
  0001_left.png
  0001_right.png
  0002_left.png
  0002_right.png
```

也支持 `left_0001/right_0001`、`0001_L/0001_R` 和 `left/`、`right/` 对应子目录。图片格式支持 PNG、JPEG、BMP、TIFF；同一相机的有效观测分辨率需一致。导入后核对配对编号，未配对项目可在日志中查看。

## RGB 与辅助相机标定

先完成参考 RGB 左目所在组的双目标定，或读取已有双目参数，再进入融合采集和标定。使用的图像组合是 **RGB 左目 + 辅助单目**。

1. 配置辅助标定板。默认红外棋盘格为 **7 列 × 5 行格子、每格 80 mm**，对应 **6 × 4 内角点**；也可独立配置圆点阵或导入板参数。
2. 采集两路都能看到同一块物理板的图像。离线数据按 `rgb_left_0001.png`、`aux_0001.png` 同编号配对。
3. 通过“复核已保存原图”或离线导入，逐对确认物理原点、行列方向、来源和板规格，明确纳入至少 6 对图像。
4. 选择辅助相机模型，执行固定 RGB 内参的联合细化，检查双向留出误差和全视场结果。

输出包括辅助内参和 `T_rgb_left_aux`，其方向为 **辅助相机坐标 → RGB 左目坐标**。对称棋盘格尤其需要核对角点顺序，倒序匹配也可能得到较低误差。软件配对不等于硬件同步；这里生成的是标定参数，热红外或声学融合算法需另行实现。

已有数据可通过“加载会话”恢复；只有参数文件时，使用“文件 → 读取双目标定文件”，支持项目完整 JSON 或 OpenCV 双目 YAML/XML。读取 OpenCV 文件时须明确选择平移量的 m/mm 单位。需要补拍时，从结果页选择“继续 RGB+红外采集”，核对设备、左右顺序、镜头和输出设置后再连接。

## 结果与日志

源码运行时，数据默认写入当前工作目录；打包程序写入 EXE 所在目录。典型会话结构如下：

```text
sessions/<会话名>/
  session.json
  images/<相机组>/         # 双目原图
  multimodal/             # RGB 左目与辅助原图
  results/
    latest.json
    <本次结果批次>/
      rig.json
      rig_opencv.yml
      rig_opencv_standard.yml
      rig_kalibr.yaml
      calibration_manifest.json
logs/
  multicamera.log
  multicamera_fault.log
```

具体导出文件随相机组和所选格式变化，以界面目录及本轮 `calibration_manifest.json` 为准。第三方程序读取双目参数时，优先使用“标准 OpenCV”格式，并核对模型、图像尺寸、左右相机 ID 和变换方向。Kalibr 格式无法无损表达某些高阶畸变参数，此时会拒绝导出并提示原因。

从“工具 → 打开日志目录”可查看日志。反馈问题时，请附上版本或 Release 标签、复现步骤及完整报错；公开日志前移除私有相机地址等信息。

## 开发与构建

在虚拟环境中执行：

```bash
python -m pytest -q tests
python -m ruff check src scripts
python -m compileall -q src scripts
python -m multicamera --self-test-calibration --report calibration_self_test.json
```

诊断工具的测试位于 `scripts/tests/`，可用 `python -m pytest -q scripts/tests` 单独运行。自检和回归测试用于验证程序行为，硬件精度需另做实测。

Windows 本地构建使用 `scripts\build_windows.bat`；安装包使用 `scripts\build_windows_installer.bat`，需要 Inno Setup 6。脚本支持 `--pointcloud`，可将 Open3D 打入包中。

版本号维护在 `src/multicamera/__init__.py`。本地发布准备可先运行 `python scripts/prepare_release.py --dry-run` 查看计划，完整约定见 [AGENTS.md](AGENTS.md)。GitHub Actions 支持手动构建和 `v*` 标签构建；通过源码回归、源码自检及最终 EXE 自检后发布附件，可下载版本以 Releases 为准。

## 更多文档

- [技术文档](docs/TECHNICAL_DOCUMENTATION.md)：架构、标定模型、参数、数据格式和操作细节。
- [更新记录](docs/CHANGELOG.md)：功能变更、验证记录与已知问题。
- [Jetson 服务设计](docs/JETSON_SERVICE_DESIGN.md) / [Orin 部署记录](docs/JETSON_ORIN_DEPLOYMENT.md)：独立于桌面应用的设备预览、双目校正和深度点云服务。部署记录按阶段保留，使用时请查阅相应的后续更新。
- [RV1126B 服务](cpp/rv1126b/README.md)：板端标定参数与校正服务。
- [mrcal 调研](scripts/mrcal_validation/README.md)：实验工具，不作为主程序的运行时标定后端。
