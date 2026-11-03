# MultiCamera Stereo Calibration

基于 Python、PySide6 和 OpenCV 的桌面标定工具，支持多组双目相机实时采集、离线图片标定，以及 RGB 左目与红外/声像仪 RGB 辅助单目的联合标定。

[Windows 下载](https://github.com/q1234ew/MultiCameraStereoCalib/releases) · [构建状态](https://github.com/q1234ew/MultiCameraStereoCalib/actions/workflows/build-windows.yml) · [技术文档](docs/TECHNICAL_DOCUMENTATION.md) · [修改记录](docs/CHANGELOG.md)

## 当前功能

- 相机服务发现、MJPEG 多路预览、连接状态提示、左右图像源交换和预览关闭/恢复。
- ChArUco 双目采集与离线导入，支持针孔和鱼眼模型、后台标定、数据筛选和逐帧误差检查。
- 首版自适应采图引导：根据已采配对图像推荐欠缺的位置、画面占比和倾斜方向，使用 QPainter 线稿示意；可切回固定计划。
- RGB 参数调整及支持相机的采图前白平衡稳定/锁定；左右 RGB 配对应用，离线流程不访问设备。
- RGB 左目 + 红外单目、RGB 左目 + 声像仪 RGB 单目的内参与跨模态外参标定，辅助平面板独立配置/导入。
- RGB+辅助原图逐对复核、辅助独立模型联合细化及双向留出验证；在线采集后复用同一复核和求解流程。
- 标定结果自动保存，以及 JSON、OpenCV 完整 YAML、单组双目标准 OpenCV YAML、Kalibr YAML 分类导出。
- 图像后台保存、会话记录、结果清单与 SHA-256、滚动文件日志和原生异常日志。
- 点云查看与融合为可选功能，Open3D 懒加载；默认 Windows 轻量包不包含 Open3D。

## Windows 下载与运行

在 [Releases](https://github.com/q1234ew/MultiCameraStereoCalib/releases) 中选择所需版本：

- `MultiCamera_Setup_0.6.0_x64.exe`：Windows x64 安装包，文件名中的版本号以实际发布为准。
- `MultiCamera_portable_<version>_x64.zip`：后续便携版统一包含实际产品版本号，完整解压后运行 `MultiCamera.exe`，不要只复制 EXE。已发布的0.1.2预览包仍使用 `MultiCamera_portable_x64.zip`，历史附件不改名。
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

“标定”页的优化策略支持“自动校验择模”：检查模型反算、单调性和校正视野，在数据充足时比较同类候选模型的留出误差；无合格候选时拒绝生成新结果。策略在通过检查的候选中优先保留足够好的标准模型，自动择模不代替独立测量验收。

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

先完成参考 RGB 左目所在组的双目标定，或读取已有有效双目参数，再进入融合采集和标定。融合预览使用 **RGB 左目 + 辅助单目**，不以 RGB 右目代替红外。

- 红外默认棋盘格为 **7 列 × 5 行格子，每格 80 mm**，对应 **6 × 4 内角点、24 点**。格子数量与内角点数量不能混填。
- 支持棋盘格和圆点阵，辅助板参数独立于 RGB ChArUco 板，可导入 JSON/OpenCV YAML。
- RGB 与红外需能看到同一块物理板；离线融合图片按 `rgb_left_0001.png`、`aux_0001.png` 同编号配对。
- 输出辅助内参及 `T_rgb_left_aux`，方向为 **辅助相机坐标 → RGB 左目坐标**。

在线点击采集后，先进行所选白平衡准备，再等待双路新帧、原图预检及连续稳定检测；可取消，稳定等待15秒超时不保存、不增加计数。本次原图仍重新验收，“已保存”“验收通过”“引导采用”分别统计。采集结论在主提示区保留5秒，完整回执常驻；有效非重复观测更新参考动作，重复或待复核会说明不推进原因，不把保存数量当作完成姿态数。

已在线保存的图像通过“复核已保存原图”进入后续流程；离线数据则导入对应目录。逐对核对物理0点和行列方向、确认来源及板规格，明确纳入至少6对，再选择辅助独立模型，执行固定RGB内参的联合细化、三折双向留出和全视场检查。加载已有会话可恢复匹配的RGB标定结果；不会仅凭重新检测把历史稳定性失败改写为通过。辅助模型不跟随RGB全局模型强制切换，不默认强制鱼眼。

- **已有 session**：通过“加载会话”自动检查文件，按原相机组后台恢复双目原图和可识别的融合配对图；缺少、未配对或损坏的项目记录在日志。已有有效双目结果可直接沿用，恢复原图后也可重新执行双目标定。
- **只有标定文件**：通过“文件 → 读取双目标定文件”读取项目完整JSON或OpenCV双目YAML/XML。缺少模型、尺寸等关键字段时会报错；OpenCV文件须明确选择T的m/mm单位。导入后先离线打开，不因逻辑ID相同而自动绑定实时设备，也不把缺失RMS显示为实测0。
- **继续补拍**：结果页选择“继续RGB+红外采集”，确认相同设备、左右顺序、镜头及输出设置，再配置视频地址。专用续采路径保留RGB内参和兼容的辅助观测；同一会话新增编号避开旧图，历史原图仍需逐对复核后纳入联合求解。

此流程输出**待实物验证候选**。留出RGB/辅助误差分别使用各自原始像素，不能混为一个单帧PnP误差；仍需独立实物对齐和新数据验收。

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

上图为示例：多组会按组生成标准 OpenCV/Kalibr 文件，文件是否生成取决于结果和导出条件。以 UI 显示的实际目录、日志和本轮 `calibration_manifest.json` 中的成功文件/哈希为准；目录内保留的旧文件不一定属于本轮。标定完成后自动保存，手动导出还可指定其他目录。

第三方读取双目参数优先选择“标准 OpenCV”格式，并核对镜头模型、图像尺寸、左右相机 ID 和变换方向。Kalibr `radtan` 不能无损表达非零高阶畸变时会明确拒绝，不会静默丢弃参数；多组完整相机链需要相应组间外参。融合扩展不保证等同 Kalibr 原生采集流程。

菜单“工具 → 打开日志目录”可直接定位日志。反馈故障时提供对应日志、版本/Release 标签、失败操作和完整报错；提交日志前检查是否含不希望公开的相机地址等信息。“加载会话”会预检相容的RGB标定结果，并后台读取已识别的原图；文件检查报告保留在日志及会话按钮提示中。缺少辅助配置、文件编号歧义或图片损坏时需先处理对应提示；读取原图不自动确认物理角点或标定精度，缺失或不相容的RGB结果不能作为融合标定前提。

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
python scripts/prepare_release.py                     # PATCH：0.6.0 → 0.6.1
python scripts/prepare_release.py --bump minor        # MINOR：0.6.0 → 0.7.0
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
