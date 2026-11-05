# MultiCamera Stereo Calibration

基于 Python、PySide6 和 OpenCV 的桌面标定工具，支持多组双目相机实时采集、离线图片标定，以及 RGB 左目与红外/声像仪 RGB 辅助单目的联合标定。

[Windows 下载](https://github.com/q1234ew/MultiCameraStereoCalib/releases) · [构建状态](https://github.com/q1234ew/MultiCameraStereoCalib/actions/workflows/build-windows.yml) · [技术文档](docs/TECHNICAL_DOCUMENTATION.md) · [修改记录](docs/CHANGELOG.md)

## 当前功能

- 相机服务发现、MJPEG 多路预览、连接状态提示、左右图像源交换和预览关闭/恢复。
- ChArUco 双目采集与离线导入，支持针孔和鱼眼模型、后台标定、数据筛选和逐帧误差检查。
- 首版自适应采图引导：根据已采配对图像推荐欠缺的位置、画面占比和倾斜方向，使用 QPainter 线稿示意；可切回固定计划。
- RGB 参数调整及支持相机的采图前白平衡稳定/锁定；左右 RGB 配对应用，离线流程不访问设备。
- RGB 左目 + 红外单目、RGB 左目 + 声像仪 RGB 单目的内参与跨模态外参标定，辅助平面板独立配置/导入。
- 标定结果自动保存，以及 JSON、OpenCV 完整 YAML、单组双目标准 OpenCV YAML、Kalibr YAML 分类导出。
- 图像后台保存、会话记录、结果清单与 SHA-256、滚动文件日志和原生异常日志。
- 点云查看与融合为可选功能，Open3D 懒加载；默认 Windows 轻量包不包含 Open3D。

## Windows 下载与运行

在 [Releases](https://github.com/q1234ew/MultiCameraStereoCalib/releases) 中选择所需版本：

- `MultiCamera_Setup_0.2.0_x64.exe`：Windows x64 安装包，文件名中的版本号以实际发布为准。
- `MultiCamera_portable_x64.zip`：便携版，完整解压后运行 `MultiCamera.exe`，不要只复制 EXE。
- `SHA256SUMS.txt`：下载文件校验值。
- `calibration_source_test.json`、`calibration_self_test.json`：源码环境及最终 EXE 的标定自检报告。

发布说明记录构建提交和是否包含 Open3D。同名文件可能出现在不同 Release 中，应以 Release 标签、提交和校验值辨别版本，而不是只看安装包文件名。

**推送源码或触发构建不等于已生成新版安装包。** 只有 Actions 成功且对应 Release 已发布下载文件，才表示该构建可下载。预发布版本也在 Releases 中，可能不出现在 Latest 标签下。

当前安装脚本面向 Windows 10 及以上 x64。程序默认在 EXE 同级创建 `sessions/` 和 `logs/`，请选择具有写入权限的位置；安装目录不可写时，建议改用用户可写目录中的便携版。

## 源码安装

需要 Python 3.10 或以上；Windows 构建工作流使用 Python 3.11。以下命令均在仓库根目录执行。

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m multicamera
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c packaging/windows/constraints.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m multicamera
```

开发环境需要点云功能时，可在对应虚拟环境额外安装 `pip install -e ".[pointcloud]"`。不需要点云时无需安装 Open3D。其他平台源码可安装不代表所有硬件及桌面环境已完成验收。

## 双目标定流程

1. 在“相机配置”中设置左右相机、镜头模型和 RGB ChArUco 板参数。实物尺寸必须准确；界面长度单位为 mm，内部计算和结果平移使用 m。
2. 连接视频流，在“采集”页创建会话并采图；也可直接导入离线左右图片。采集前确认左/右角色，必要时交换来源；离线交换不修改原文件。
3. 按引导补充不同位置、大小和倾斜方向的图像。自适应冷启动没有可靠内参时需手动补采基础样本，不会假定倾角已经到位。
4. 前往“标定”，选择数据筛选/标定策略，执行单组或全部标定。后台计算期间界面可继续响应。
5. 在“结果”中检查质量提示、逐帧误差和校正残差，查看自动保存目录，或选择对象与格式手动导出。

自适应推荐是可解释的启发式补缺，不是数学意义的全局最佳姿态。临时引导内参不写入最终标定文件；采图数量、引导进度或低训练 RMS 均不能单独证明标定精度合格。

白平衡默认采用“自动稳定（不锁定）”：设备已经自动时仅检查，不重复写模式；设备处于手动时，采图准备仅开启自动，不再自行关闭。每次采图重新检查，只放行本次，不承诺参数固定或颜色准确。另保留显式“手动冻结验证（可能变色）”选项，仅供已验证设备使用，不代表通用锁定能力。此策略避免再次触发自动转手动变色，不是设备已有偏绿的根因修复；不稳定、接口失败或取消仍拒绝采图。

### 离线图片命名

双目导入支持以下配对命名，以及在父目录内递归查找：

```text
images/
  0001_left.png
  0001_right.png
  0002_left.png
  0002_right.png
```

也支持 `left_0001/right_0001`、`0001_L/0001_R` 和 `left/`、`right/` 对应子目录。支持 PNG、JPEG、BMP、TIFF；同一相机的有效观测分辨率需一致，板配置必须与图片中的实物一致。导入后核对配对编号及未配对日志。

## RGB + 红外 / 声像仪 RGB 标定

先完成参考 RGB 左目所在组的双目标定，再进入融合采集和标定。融合预览使用 **RGB 左目 + 辅助单目**，不以 RGB 右目代替红外。

- 红外默认棋盘格为 **7 列 × 5 行格子，每格 80 mm**，对应 **6 × 4 内角点、24 点**。格子数量与内角点数量不能混填。
- 支持棋盘格和圆点阵，辅助板参数独立于 RGB ChArUco 板，可导入 JSON/OpenCV YAML。
- RGB 与红外需能看到同一块物理板；离线融合图片按 `rgb_left_0001.png`、`aux_0001.png` 同编号配对。
- 输出辅助内参及 `T_rgb_left_aux`，方向为 **辅助相机坐标 → RGB 左目坐标**。

对称棋盘格存在跨模态角点顺序歧义，必须保证两路对应同一物理角点；低重投影误差不能排除倒序误匹配。软件配对不等于硬件同步。本流程产出融合所需标定参数，不包含热红外或声学融合算法本身。

## 结果、导出与日志

源码运行默认相对于当前工作目录；打包程序默认相对于 EXE 所在目录：

```text
sessions/<会话名>/
  session.json
  images/<相机组>/left_0000.png
  images/<相机组>/right_0000.png
  multimodal/rgb_left_0000.png
  multimodal/aux_0000.png
  results/
    rig.json
    rig_opencv.yml
    rig_opencv_standard.yml
    rig_kalibr.yaml
    calibration_manifest.json
logs/
  multicamera.log
  multicamera_fault.log
```

上图为示例：多组会按组生成标准 OpenCV/Kalibr 文件，文件是否生成取决于结果和导出条件。以 UI 显示的实际目录、日志和本轮 `calibration_manifest.json` 中的成功文件/哈希为准；目录内保留的旧文件不一定属于本轮。标定完成后自动保存，手动导出还可指定其他目录。

第三方读取双目参数优先选择“标准 OpenCV”格式，并核对镜头模型、图像尺寸、左右相机 ID 和变换方向。Kalibr `radtan` 不能无损表达非零高阶畸变时会明确拒绝，不会静默丢弃参数；多组完整相机链需要相应组间外参。融合扩展不保证等同 Kalibr 原生采集流程。

菜单“工具 → 打开日志目录”可直接定位日志。反馈故障时提供对应日志、版本/Release 标签、失败操作和完整报错；提交日志前检查是否含不希望公开的相机地址等信息。“加载会话”目前不等于恢复全部内存观测/标定状态，历史照片可通过离线导入重新计算。

## 验证与打包

在对应虚拟环境执行：

```bash
python -m compileall -q src scripts
python -m pytest -q tests
python -m ruff check src scripts
python -m multicamera --self-test-calibration --report calibration_self_test.json
```

版本号只在 `src/multicamera/__init__.py` 的 `__version__` 中维护；Python 包元数据、Windows 安装器及本地版本化文件名自动读取。修改版本后重新执行对应平台的可编辑安装命令，刷新安装元数据。

本地发布准备命令（不提交、不创建标签、不推送）：

```bash
python scripts/prepare_release.py --dry-run           # 预览默认 PATCH 升版
python scripts/prepare_release.py                     # PATCH：0.2.0 → 0.2.1
python scripts/prepare_release.py --bump minor        # MINOR：0.2.0 → 0.3.0
python scripts/prepare_release.py --preview --bump patch  # 新 PATCH 版本的首个测试构建
python scripts/prepare_release.py --preview           # 当前目标版本的下一次测试构建
python scripts/prepare_release.py --bump none         # 将当前已验证版本准备为正式版
```

以上为不同用途的示例，并非依次全部执行。实际准备要求工作区干净；`--dry-run` 允许未提交修改且不写入任何文件。预发布按上海日期，从本地标签和 Git 元数据目录内的准备记录取最大序号加一；不会自动访问远程。多人或多机发布前需先同步标签，本地编号不能视为远程预留。版本发生变化后，应更新修改记录、重新安装元数据、完成验证并提交，再创建输出中的建议标签。预发布目标版本如已存在正式标签，必须选择新的升版目标。

发布前执行 `python scripts/check_release.py --installed --require-clean`，再创建正式标签 `vX.Y.Z` 或预发布标签 `vX.Y.Z-build.YYYYMMDD.N`，执行 `python scripts/check_release.py --installed --require-clean --tag <标签>` 核对版本及标签指向。日期按上海时区，序号为当日正整数；检查不会自动创建或推送标签。CI 在构建前重复检查，并对比源码与最终 EXE 的版本、提交和标签。已存在的 Release 禁止覆盖，修复构建必须使用新标签。

Windows 本地构建入口为 `scripts\build_windows.bat`；安装包入口为 `scripts\build_windows_installer.bat`，需要 Inno Setup 6。添加 `--pointcloud` 可选择包含 Open3D 的构建，具体参数见脚本和技术文档。

安装器默认使用速度优先的 `normal` 压缩；Actions可选 `installer_compression=max/ultra64`，本地可设置 `MULTICAMERA_INSTALLER_COMPRESSION`。CI在最终EXE自检通过后并行生成ZIP与安装器，耗时及体积见新构建的Release附件 `packaging_metrics.json`；压缩更快可能使下载包略大，不影响程序功能。

GitHub Actions 支持手动构建或推送 `v*` 标签触发构建。标签构建包含安装包，默认不包含 Open3D；发布前执行源代码回归、源码自检和最终 EXE 标定自检，通过后将安装包、便携 ZIP、报告及校验值发布到 Release。

## 文档与维护

- [技术文档](docs/TECHNICAL_DOCUMENTATION.md)：架构、线程、标定模型、参数默认值、数据格式及验证边界。
- [修改记录](docs/CHANGELOG.md)：功能、兼容性、实际验证及已知限制，采用追加式维护，不能覆盖历史条目。
- 当前 mrcal 调研不属于主程序运行时标定后端；本项目不默认依赖 mrcal。

本机测试、自检成功不替代 Windows 最终 EXE、真实设备、独立图像集或已知尺寸/距离的精度验收。
