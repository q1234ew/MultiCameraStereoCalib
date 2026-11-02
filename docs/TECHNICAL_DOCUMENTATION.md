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
  - `configuration.py`：配置变更影响分析，按相机组及共享相机依赖判定失效范围。
  - `widgets/export_dialog.py`：手动导出对象、格式、内容清单及失败时保留旧文件。
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
  results/latest.json
  results/calib_<时间戳>_<批次ID>/rig.json
  results/calib_<时间戳>_<批次ID>/rig_opencv.yml
  results/calib_<时间戳>_<批次ID>/rig_opencv_standard.yml
  results/calib_<时间戳>_<批次ID>/rig_kalibr.yaml
  results/calib_<时间戳>_<批次ID>/calibration_manifest.json
```

双目图片保存由后台队列串行执行，避免多个线程同时写 `session.json`。保存失败不会推进 metadata，并会在 UI 日志中提示。辅助配对图片目前仍在采集槽中同步保存。

标定完成时，主窗口自动保存结果；如果没有当前会话，会先自动创建 `calib_<时间戳>` 会话。单组、全部标定和多模态标定均走这条保存路径，不需要再执行联合优化或生成点云。状态栏和操作日志显示实际结果目录，文件日志逐条记录成功写入的绝对路径。

- `rig.json`：完整 rig，可重新读取。
- `rig_opencv.yml`：OpenCV 完整 YAML。
- `rig_opencv_standard.yml`：单组双目的 OpenCV 标准文件。
- `rig_kalibr.yaml`：所有相机之间存在所需变换时，保存完整 Kalibr 链。
- 多组双目分别生成 `rig_pair_01_opencv_standard.yml`、`rig_pair_01_kalibr.yaml` 等文件，序号按 rig 中相机组顺序排列；OpenCV 文件内记录 `pair_name` 和左右相机 ID。缺少组间外参导致完整 Kalibr 链失败时，各组文件仍独立生成，并在操作日志提示具体失败原因。

每次自动保存创建独立批次目录，每个文件先在该批次的临时目录生成后发布。JSON 保存失败会弹窗提醒，并保留内存结果供手动导出；某个 YAML 格式失败不阻断其他格式，失败格式不会出现在当前批次目录。旧批次及旧版根目录文件均不删除，也不复制进新批次。结果页、状态栏、打开/复制目录均指向本次实际批次目录。

保存完成后原子更新 `results/latest.json`，记录批次相对目录和 generation ID；索引更新失败保留上一索引，UI 明确提示，本次成功文件及其目录仍可访问。`CalibrationSession.result_directory` 和 `load_rig()` 优先按索引读取，无索引时兼容旧版 `results/rig.json`；索引非法、指向越界/缺失批次时拒绝读取，不静默回退到历史文件。批次清单记录本批次成功文件的SHA-256及失败格式，索引不计入标定文件数量。低层 `save_rig()` 默认仍只写旧版根目录JSON，供既有调用兼容；应用完整保存必须使用 `save_calibration_results()`，不应混用两种写入入口。

自动创建会话只保证标定结果落盘，不会补存此前在无会话状态下采集的原始图片。需要保存完整原始数据时，仍应在第一次采集前新建会话。

### 3.1 配置变更与已有工作保护

配置对话框先构建候选配置，完成 ID、相机组名和地址校验，再由主窗口计算影响；确认之前不会停止原视频流或修改相机配置。取消确认不会改变连接、缓存或结果。自动采集、图片导入、后台保存或标定尚未结束时，暂不允许打开配置修改，避免工作线程使用变更中的数据。

- 无实质修改：保留采图、标定结果、预览及视频连接，不重置、不重连。相机角色/分组的内部规范化不算来源改变。
- 仅修改辅助 ID、URL、模态或板参数：清理辅助配对缓存、辅助内参和跨模态结果，保留 RGB 内参、双目结果及双目采图。
- 修改某组左右目来源、类型或删除该组：清理该组的检测缓存和结果，保留其他独立组；共享相机的组按依赖关联失效。若涉及 RGB 参考相机，还会清理依赖它的辅助数据和结果。
- 修改全局 ChArUco 板或镜头模型：按全局几何设置变更处理，清理全部相关采图缓存和结果，需重新采集或导入后计算。

只在受影响范围内确有缓存/标定结果时弹出确认，逐项列明失效对象；默认不应用。所有上述清理均针对内存，不删除磁盘历史图像或文件。后续标定成功在会话 `results/` 下新增独立批次，既有结果无需另行导出即可保留。

实时来源变化会按之前是否正在连接来决定是否重连；修改板/模型但来源未变时不重启视频连接。重连保留未受影响的预览覆盖统计，受影响的统计重置。空 URL 的离线虚拟双目组在重新打开配置后仍会保留。

会话 `session.json` 更新当前 `board/pairs/auxiliary_camera/camera_model`，并追加 `configuration_history`，记录变更前配置、时间及生效时的双目/多模态帧计数。既有图片和帧计数不重置。配置记录采用临时写入后替换，失败时提示日志并保留上一版磁盘 metadata，内存仍反映已应用的配置。这不是结果版本管理或完整的逐帧配置索引；历史标定结果不会因修改配置自动重算。

### 3.2 相机与标定板配置窗口

配置窗口分为三页，最小 900×640，默认 1080×760。每页单独滚动，底部校验摘要、保存和取消按钮固定；填写过程不修改运行中的配置，也不会发起视频连接。

各页 `QFormLayout` 显式使用 `AllNonFixedFieldsGrow`，不依赖系统样式的默认字段伸展策略；参考相机为只读可复制单行框，完整值可悬停查看，长 ID 不参与多行高度估算。原生 macOS 与 offscreen 样式均需检查，避免离屏可用而原生输入框收缩或文字遮挡。

- **相机连接**：设备发现、左右流选择和双目相机组表格。类型使用 RGB/红外下拉编辑；名称和 ID 自动避开已有默认编号。手动新增的 URL 默认为空，作为离线组；左右 URL 需同时填写或同时留空。选中行才启用删除/交换来源，交换仍只改 URL，不改角色 ID。
- **融合相机**：模态、已发现辅助流、辅助 ID/URL 和实际 RGB 左目参考；无参考时显示缺失状态，但允许先保存离线辅助配置。进入融合配置入口会定位本页，板设置入口转至标定板页。选用发现服务保留已填写 ID，仅在 ID 空白时生成名称。
- **标定板**：全局镜头模型、RGB ChArUco 和独立辅助平面板。模型仍由双目与辅助内参共用，本轮不新增每路模型选择。RGB 与辅助尺寸均以 mm 填写，在保存边界换算为内部 m；不会改变已有 JSON/YAML 的长度单位。

RGB 板列/行按格子数量填写，显示内角点总数与不含边距的板面尺寸。Marker 边长必须小于方格边长，ArUco 字典容量必须容纳全部标记；不合法时不构造 OpenCV 板对象、不应用配置。

辅助棋盘默认按**格子数量**填写 7×5、80 mm，保存为 6×4 检测点及 0.08 m。可切换“内角点数量”，切换时自动换算而不是重新解释同一数字。圆点阵始终按圆心数量填写，禁用棋盘计数方式；非对称选项只对圆点阵生效。导入文件中的 `cols/rows`、`board_cols/board_rows` 仍然是检测点数，不因界面默认按格子数而改变。恢复红外预设或导入参数不修改 RGB 板。

字段变更后更新校验摘要，区分未修改、有未应用修改和具体错误。非法 URL、只填一路 URL、重复组名、冲突 ID、同组相同视频来源、辅助与双目复用 ID/来源、无效板参数均阻止保存；直接触发保存也会复验并定位错误页。实时视频仅接受 HTTP(S) 地址，语法有效不代表设备在线；实际可达性仍由连接流程判断。旧配置中无效地址可以打开窗口修正。已采数据的失效范围确认仍遵循上一节。

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
- 双目自动配对只处理同一主机、同类型（`rgb/ir`）且各有唯一一路的已知左目和右目；不跨主机配对，也不猜测未知目别。同一主机存在多路同侧候选时交给手动选择。已配置的正反同源配对不重复添加；手动选择仍允许已确认的跨主机双目。
- 发现服务持续到达时保留下拉框中的既有选择，无候选时显示空状态；辅助单目按模态从发现服务中单独选择，不作为双目自动配对对象。

`MJPEGGrabber` 默认参数：

- `READ_TIMEOUT = 10.0s`
- `RECONNECT_DELAY = 2.0s`
- `FPS_UPDATE_INTERVAL = 1.0s`
- `DEFAULT_MAX_DECODE_FPS = 20.0`

解码限频在 JPEG 解码前生效。过密帧会被跳过，避免 UI 只显示 15fps 时后台仍无限解码。

## 5. 采集与检测

### RGB 启动偏绿排查记录

2026-10-24 对现场 `192.168.31.188` 两路 RGB 做只读取证及经授权的左路单变量对照：

- 直接访问 8080/8082 的 `/snapshot`，绕过 Qt 预览，JPEG 仍偏绿；两路 API 均为手动白平衡（`white_balance_automatic=0`、`white_balance_temperature=4254`），曝光手动。客户端连接代码不主动写入 RGB 控制。
- 仅向左路 POST `{"white_balance_automatic":1}`，右路不写入，分别在 2 秒、7 秒采样。左路 7 秒样本已无原先明显绿色偏色；恢复左路原模式与色温后，偏绿重新出现。恢复后的 API 值已校验，右路所有控制值与测试前一致，全程未重启服务或设备。
- 此对照确认**当前偏绿与相机端手动白平衡设置存在可复现因果关系**，不是本应用显示转换造成；不代表已经确认开机时谁写入该设置，也不保证固定色温在所有现场均适用。自动模式下 API 仍返回 4254，因此不能仅用色温读数判断输出颜色。
- 本地对应 RK1126B 服务源码的启动脚本未指定白平衡，底层默认保持原状态，这是初始化策略的风险线索；设备实际启动命令与驱动日志因 SSH 登录未通过认证而尚未验证。尚未修改客户端启动策略、设备脚本或持久参数。

对照报告和原始 JPEG 临时保存于 `/tmp/multicamera-awb-probe-20261024/`，其中 `report.json` 记录各阶段返回参数、恢复结果和右路不变检查。该目录可能被系统清理，不是会话数据，不随发布包分发。后续测试必须先保存原参数，并在 `finally` 中恢复、回读校验；不可在未经授权时自动开关设备白平衡。

### RGB 启动与采图前锁定（2026-10-25）

按已确认方案分为两端实现：

- **服务端**：本机 RK1126B 仓库 `FusionSensor/ustreamer-ir/start_fusion.sh` 的 dual 左右 RGB 启动命令均显式增加 `--white-balance "$RGB_WHITE_BALANCE"`，默认 `RGB_WHITE_BALANCE=auto`；环境模板同步新增此项。它在设备打开/重新打开时生效，不改 IR 和 stitch 取流。自定义启动命令必须分别加参数；IR-only systemd 单元不会因此凭空变成 RGB 服务。
- **标定端**：采集页默认勾选“采图前稳定并锁定 RGB 白平衡”。首次实时采图先后台准备当前 RGB 组的左右两路，包括融合只使用 RGB 左目时对应组的右目；不向红外或声像辅助单目下发 RGB 设置。离线导入和离线标定不访问设备。

`streaming/white_balance.py` 为不依赖 Qt 的控制逻辑，`ui/threads/white_balance_worker.py` 提供 worker 和采图守卫。准备流程：

1. 读取控制能力，向所有目标 RGB 下发 `white_balance_automatic=1`；记录准备前参数。
2. 最少预热 7 秒，每 0.5 秒读取 `/snapshot`，缩小到 160×90，排除近黑/近饱和像素，计算平均 `R/G` 与 `B/G`。连续 4 个样本两项绝对跨度均不超过 0.025 才认为统计稳定，稳定期限默认 20 秒。
3. 仅下发 `white_balance_automatic=0`，**不回写自动模式下可能陈旧的色温，不修改曝光或增益**。等待 1 秒后再次采样两次，与锁定前均值的两项绝对差均须不超过 0.05，并回读自动模式仍为 0。
4. 成功后丢弃关联相机缓存 0.8 秒，再等待 1.1 秒恢复原请求的采图；恢复时重新读取最新帧，不使用准备前保留的图像。自动采集准备时暂停，成功且仍在原采集步骤时恢复；失败不恢复自动采集。

任何控制/图片接口失败、画面太暗/过曝、未稳定、计算期间取消或锁定后颜色跳变都会拒绝这次采图，不新增标定缓存、图片保存任务或帧计数；尝试将本次已处理的目标恢复自动白平衡，恢复失败写异常日志，不能宣称设备已恢复。采图前创建的空会话可以保留。关闭窗口若仍有准备线程运行会先请求取消，并提示稍后再次关闭，避免销毁运行中的线程。

固定操作区显示准备阶段（读取参数、预热、检查稳定、锁定、复核、等待新画面）与累计秒数，每 250 ms 刷新；提供“取消准备”。计算期间取消通过线程中断请求退出和恢复流程，网络调用返回前可能仍需等待；完成计算但尚在等待新帧时取消只终止采图，明确提示已经锁定的参数保持不变，不宣称恢复自动。取消不会恢复原拍照请求或自动连拍；主按钮在忙状态显示“白平衡准备中…”。

锁定记录只在当前应用连续使用期间缓存。连接/断开事件、新建或加载会话、应用内 RGB 参数修改及选项切换使它失效；不重新启动网络连接。后台准备期间保护配置、导入、标定及 RGB 参数面板，不让冲突操作交叉写设置。若准备期间切换了任务/会话，原采图请求不会跨上下文执行。无 RGB 控制接口的通用 MJPEG 相机可以显式取消勾选，日志记录跳过，此时不保证颜色锁定；关闭选项本身不写入设备参数。

**边界**：颜色统计稳定不是色彩校准，也不证明左右颜色完全一致。某些 UVC 固件在关闭自动模式时会恢复历史增益，此时必须被锁定后检查拒绝，不能把模式回读成功当作真正冻结；本型号“只切开关、不回写色温”的硬件行为仍需实机验证。外部客户端在采集期间改变白平衡不会被当前缓存实时监测，应避免多客户端竞争。

本次修改是源码实现，不等于设备已部署。2026-10-25 对现场地址读取参数即超时，在首次写入前退出；没有下发参数、上传脚本或重启设备。服务端改动位于 `/Users/fuqiang/Dev/RK1126B/FusionSensor`，不包含在 MultiCamera 单仓库提交中；部署时须同时更新服务脚本/实际启动配置。

### 原图采集与离线数据

采集按钮流程：

1. 实时采图必须先有可保存原图的会话；首次操作显示“创建会话并采集”，通过主窗口同步创建会话，创建失败则停止采图和自动连拍。随后实时 RGB 按启用选项通过白平衡准备守卫，再从 `StreamManager` 获取同步帧，失败时尝试 relaxed 最新帧。
2. 使用原始帧分别更新左右 `IntrinsicCalibrator`。
3. 使用原始帧更新 `StereoCalibrator`。
4. 评估 sharpness、brightness、角点数等质量指标。
5. 将 PNG 保存任务提交到后台串行队列。
6. 更新采集引导、撤销栈、进度条和 UI 日志。

预览检测和采集检测是两条路径：

- 预览检测为轻量路径，限频并可缩小图像。
- 采集/标定检测使用原始帧，确保标定精度不受预览优化影响。

离线双目导入支持递归扫描所选目录，可直接选择数据集父目录。支持常见命名：

- `0001_left.png` / `0001_right.png`
- `left_0001.png` / `right_0001.png`
- `0001_L.png` / `0001_R.png`
- `left/0001.png` / `right/0001.png`

重复导入会重置 `offline` 虚拟相机组和旧的离线标定缓冲，避免把不同数据集追加混在一起。

左右图像源校正：

- 设备服务可能出现 metadata 标记为 left/right，但实际 MJPEG 图像内容相反的情况。
- 配置窗口的相机组表格提供“交换左右图像源”，仅对调左右 URL，保留 `*_L/*_R` 相机 ID。
- 使用该按钮后重新确认配置、连接、采集和标定，导出的 `M1/D1` 会对应修正后的左目图像源。
- 离线导入后，标定面板提供“交换离线左右图”。该操作会重建 `offline_L/offline_R` 的检测和标定缓冲，保留相机 ID 不变，但让 `offline_L` 使用原右图、`offline_R` 使用原左图；交换后需重新执行双目标定和导出。

撤销上一帧：

- 标定缓冲立即回退。
- 如果图片保存还在 pending，保存任务完成后删除对应文件。
- 如果图片已保存，立即删除左右 PNG，并重算 `frame_count`。

## 6. UI 预览性能

### 6.1 默认工作台与操作入口

默认采用第 1 版设计方向：中性深灰工作区、克制的蓝色主操作、绿色连接状态；左侧预览和配对图像，右侧当前任务。最小窗口为 1000×680，默认 1440×960；右侧默认约 430 px，可在 360～600 px 间调整，内容可纵向滚动，主操作区固定在底部。

- 顶部工具栏保留连接、断开、相机配置、RGB 参数和会话目录入口；连接汇总显示实际连接路数。生成标定板和点云工具移至“工具”菜单，原有计算能力保留。
- 任务选择区分 RGB 双目与 RGB+红外/声像仪 RGB；“采集 / 标定 / 结果”切换当前任务的操作面板，不清空数据或重新计算标定。
- 默认“并排”显示相机。融合任务可切换“红外优先”（声像仪为辅助优先），辅助画面位于主工作区，RGB 左目移至右侧上方对照；普通双目不混入 AUX。切换只影响显示，流连接和采集来源不变。
- 画面保持原始宽高比，允许留白；仅叠加检测角点，尺寸、FPS、角点数量位于画面下方，采图姿态引导移至右侧。长相机名称省略显示，悬停查看完整名称。
- “采集”底部固定原图保存状态、采图/导入和前往标定；“标定”集中策略与计算操作；“结果”显示真实 RMS、对象数量、保存路径/异常，并提供导出和打开结果目录。导出仍使用对象范围选择对话框，未生成结果不能导出。
- 操作日志位于预览下方，默认收起；失败日志自动展开。界面最多保留 1000 个文本块，磁盘日志策略不变。已有后台任务互斥、配置影响确认、离线左右交换及预览关闭/恢复继续生效。

### 6.2 配对缩略图与只读回看

`PairGallery` 直接引用当前采集/导入流程持有的原图，不为缩略图再次复制整批原图。按窗口宽度每页显示 1～4 对，仅缓存当前页图标；收起时释放图标缓存。实时双目索引关联现有标定器图像缓存，离线与辅助任务复用各自图像列表。

点击缩略图进入明确标识的“回看”状态，暂停主预览的新帧显示；后台接收和最新帧缓存仍更新。返回实时、切换任务、替换当前数据或触发采图会结束回看。采图始终从 `StreamManager` 获取原始最新帧，而非回看图或带角点的显示图。预览关闭记录在回看期间同样有效；缩略图不是删除数据入口。

### 6.3 渲染参数与调优

`MultiStreamView` 默认参数：

- `UI_FPS_CAP = 15`
- `DETECT_FPS_CAP = 2.0`
- RGB ChArUco检测不设缩小上限，直接使用收到的原图分辨率。
- `PLANAR_DETECT_MAX_WIDTH = 960`（仅融合辅助平面板预览，保留原策略）

预览渲染策略：

- 每路相机最多 15fps 更新 UI。
- ChArUco overlay 每路最多 2fps 检测。
- RGB双目预览和无历史检测记录的离线重检直接使用原图像素，显示缩放只影响pixmap，不影响检测输入。
- 融合辅助平面板预览超过960px时仍按原策略缩小检测并映射坐标；本轮不改IR检测策略。
- `CameraView` 缓存当前 label 尺寸下的 pixmap，避免尺寸未变时重复平滑缩放。

如果预览 CPU 偏高：

- 先降低 `DETECT_FPS_CAP`。
- 再降低 `UI_FPS_CAP`。
- 不通过缩小RGB检测输入换取性能；高分辨率设备需另行评估后台调度或原图ROI，不能将缩小图零检测直接视为未放入标定板。

### 6.4 右侧任务工具栏

右侧按第 1 版草案收敛为“当前任务 → 标定板 → 采图引导 → 固定操作区”，不嵌套装饰卡片：

- 标题为当前任务，右侧显示采集/标定阶段。融合任务用独立行显示传感器组合、RGB 左目内参、实时配对和融合标定状态；状态依据原有就绪检查，异常详情位于悬停提示，不再将两种状态挤入长段落。
- RGB 内参缺失时显示带提示的跳转箭头，直接进入 RGB 双目标定页；不自动计算、不清空 AUX 数据，不绕过已有标定条件。
- 标定板默认显示简短规格，采集阶段收起类型、物理规格、毫米间距、检测点数明细，点击箭头可展开；标定阶段默认展开。标题旁“配置”调用原有配置影响确认流程。小窗口优先保留 RGB 内参状态、采集就绪、引导及有效配对，板配置移到引导之后，详细异常仍可通过悬停与日志查看。
- 紧凑采图引导移除重复的进度数字与圆点条，保留当前姿态编号、视场位置/姿态图、前后导航及跳过。九宫格列标签有独立纵向空间，不覆盖姿态标题。有效配对数量只在细进度条上方显示，建议采图数不代表已自动验证姿态覆盖。
- 底部 `_ActionPages` 按当前页计算高度，保持主操作固定在底部，不为隐藏页面预留多余按钮高度。没有会话时主按钮为“创建会话并采集”，创建成功后才继续准备和采图；创建失败不生成新观测。提示“保存已启用”不表示待保存任务已经落盘，实际保存错误仍通过日志显示。

融合配置页直接提供“发现 / 刷新”和“使用发现服务”，区分未扫描、正在扫描、扫描完成但无匹配流与已发现数量。已填写的 URL 与发现结果分开表述，不把未发现当作无法连接；地址的实际连通性仍以连接操作结果为准。

结果页先显示质量建议和最多三条风险，展开“质量指标详情”可查看完整风险、RMS 及已记录参与帧数；文件保存状态独立显示。`quality_summary.py` 复用 `CalibrationEvaluator` 的现有内参/双目阈值与建议，不改变标定算法或拦截导出。缺少逐帧误差要求人工复核，无效指标不能判为可接受；融合结果始终要求目视检查跨模态对齐，不以低 RMS 保证合格。结果目录显示会话名与 results，完整路径位于提示中，可复制或打开实际目录。

双目采集页进度位于引导下方，以“有效配对 N / 最少帧数”明确计数；多组选择取所选组最少有效配对，未创建观测的组按零计。零样本隐藏空进度条，产生有效配对后显示 8 px 细条，采集记录数保留在提示中。没有质量观测时显示简短空状态并隐藏评分及空建议，不再显示多组 `--` 占位符；重置时清除过期评分。最少帧数、采集目标变化后立即刷新展示，不改变求解器的就绪或质量阈值。小窗口收起重复板规格及会话路径，板规格保留在采集目标提示中，原图保存状态保留在固定操作区。

相关回归覆盖原有就绪/忙状态、真实板规格、RGB 前置步骤跳转保留数据、精简引导继续导航、底部随阶段适配。原生 macOS 在 1440×1024、1280×800、1000×680 检查并排/红外优先及标定状态；合成图仅为测试素材，未改变运行时视频来源。

P1/P2 专项回归位于 `test_workbench_ui.py`、`test_configuration_ui.py`、`test_white_balance.py` 和 `test_quality_summary.py`，覆盖创建会话失败不采图、运行中/等待新帧时取消、阶段反馈、质量等级/缺失指标、路径复制和发现空状态。2026-10-25 最终完整 161 项、原生窗口专项 6 项通过；源码数值自检报告 `/tmp/multicamera-priority-self-test-20261025.json`。这些检查不代表 Windows 或真实设备已验收。

### 6.5 图标与姿态清单

应用图标采用“标定格”方案：`assets/logo.png` 为选定原图，Qt 应用和主窗口沿用 `logo_png_path()` 读取；`assets/logo-legacy.png` 保留旧设计。Windows EXE 与 Inno Setup 使用 `assets/logo.ico`，含 16、24、32、48、64、128、256 px。修改主图后运行 `.venv/bin/python scripts/build_app_icon.py` 重建 ICO，再重新打包；只替换 exe 旁的 PNG 不会修改 EXE 或安装器内嵌图标。图案来源、生成提示词及尺寸记录见 `assets/APP_ICON.md`，资源检查见 `tests/test_app_icon.py`。本机资源检查不等同 Windows 任务栏/安装器实机验收。

工作台使用 `ui/icons/` 内的 Lucide 线性 SVG 子集，完整授权和上游固定版本见该目录 `LICENSE` 与 `SOURCE.md`。`theme.app_icon` 以图标名/颜色缓存 QIcon，使用现有 PySide6 QtSvg 渲染 1x/2x/3x，显式提供普通、激活、选中及禁用状态；无需外部图标包或联网。`standard_icon` 为已有调用映射统一图标，未映射系统图标保留原有回退。wheel package-data 与 PyInstaller collect_data_files 均包含图标和授权，增加资源时必须同步检查发布产物。

采图引导右上姿态清单支持键盘选择与直接跳转，每项包含编号、位置/倾角、已采/跳过/待采；“下个待采姿态”循环查找未完成记录。菜单复用原有导航信号，查看下一姿态、跳转、跳过均不触发采图或保存。九宫格绿色数字是该区域的已采引导步数（含不同倾角），不是自动测得的真实覆盖率；悬停和可访问描述显示完整记录统计。已导入图像沿用现有引导推进规则，不将记录状态解释为实际姿态已验证。

遍历最后一步但存在未采或跳过时显示“引导已浏览完”，而非采图完成；只有所有计划步记录为已采才显示“采图计划已完成”。计划结束后仍可跳回补拍。现有实时检测、有效配对判定、撤销和标定求解不变。专项测试为 `tests/test_icons_and_guide.py`，完整回归增至 164 项；这些检查不取代真实设备与 Windows EXE 验收。

引导展示进一步分离当前姿态与记录状态，固定右侧状态区域，避免长标题挤掉“待采/已采/已跳过”。板面朝向使用无外框示意与细分隔线，底部步骤条表示当前、已采、跳过和待采（文字状态始终同时可见，不只依赖颜色）。紧凑示意高度约 200 px，空间充足时约 260 px，导航按钮高 32 px，不恢复重复圆点条。

侧栏默认分配约 430 px（实际受主窗口和分隔器分配约束），最小宽度 360 px、最大 600 px，可拖动分隔器调整。小窗口允许滚动内容，优先可读性，不再将引导、统计和全部配置强塞入首屏；固定采图操作不随内容滚动。标题处“展开采图引导”图标打开 760×520、最小 620×400 的可调整大小非模态窗口，共用当前 `CaptureSequence`，区域选择、导航、外部采图和导入更新均同步显示，不复制完成状态，也不提供另一路拍照入口。原引导隐藏或切换任务时关闭放大窗口，防止展示过期上下文。

九宫格可直接选择区域；命中检测与绘制复用 `grid_rect()`，优先跳转区域内首个待采目标，无待采时选择区域首个目标供回看。点击只发出导航信号，不调用采图、不修改目标完成状态；与姿态菜单使用相同路径更新视频引导。悬停说明区域已采记录数，并明确“远/中/近”为计划提示，不是实时距离测量。原生检查覆盖 300/360/460 px 侧栏及不同窗口大小，测试图为合成数据，不代表真实姿态测量。

## 7. 标定线程模型

单组双目标定由 `PairCalibrationWorker` 在后台 QThread 中执行。

针孔内参使用通用 `cv2.calibrateCamera`：每帧按照 ChArUco ID 从 `CharucoBoard.object_points` 取对应三维点，并与该帧图像角点保持同一顺序，支持只检测到部分角点和非连续/乱序 ID。不依赖部分 OpenCV Python 构建缺失的旧 `cv2.aruco.calibrateCameraCharuco` 接口。日志记录 OpenCV 版本、求解接口、样本数和图像尺寸；鱼眼模型继续使用 `cv2.fisheye.calibrate`。

worker 输入：

- pair name、left/right camera id。
- 左右 `IntrinsicCalibrator`。
- `StereoCalibrator`。
- 可选的左右内参缓存；当前 UI 的单组标定和全部标定均不传入旧缓存，每次按当前样本和策略重新计算内参。

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

补拍和重新标定：

- 已完成的相机组保留可点击的“重新标定”按钮；“一键标定全部”也包含已有结果且数据就绪的组。
- 补拍新增有效观测或撤销采集后，相关相机组的旧内参和双目结果失效，保留其余采集样本。全局 rig 和依赖旧内参的辅助相机结果同时失效，导出入口暂时禁用。
- 直接点击该组“标定”或“一键标定全部”，不必重置或重新采图；计算会使用当前数据筛选策略选出的全部有效样本，完成后自动更新当前会话结果文件。
- 标定期间禁用优化策略和数据筛选选择，避免计算与 UI 状态不一致。
- 质量评估显示或校正预览失败时会记录错误，不影响已生成结果的发布和自动保存。
- 重新打开应用后，可使用“导入图片”载入已保存的左右图再标定。当前“加载会话”只加载会话元数据，并不自动恢复内存中的检测缓冲。

“一键标定全部”使用串行队列，不并行跑多个 pair。这样可以避免 OpenCV 标定同时抢 CPU，也能让 UI 日志和状态更可预测。

标定质量评估和优化策略：

- 采集阶段持续评估角点覆盖率、分布均匀性、可用帧数、清晰度和亮度，并在界面给出补采建议。
- 双目标定面板提供“标准”“稳健”“联合精修”三种优化策略。
- 标准策略：分别求左右内参后固定内参求双目外参，适合大多数数据集。
- 稳健策略：固定切向畸变和高阶径向畸变，降低自由度，适合样本较少、画面噪声较大或结果抖动的数据。
- 联合精修策略：双目阶段允许基于已有内参初值微调内参，适合数据量充足后做最终对比。
- 双目标定面板提供“全部使用”“自动质量筛选”“角点丰富优先”三种数据筛选策略。
- 全部使用：所有已通过检测并进入缓冲的图像都参与标定。
- 自动质量筛选：按角点数量、清晰度和亮度综合评分保留较优帧，默认保留约 80%，且不少于最少帧数。
- 角点丰富优先：按 ChArUco 角点数量/双目共视角点数量保留较优帧，适合离线数据中有遮挡或边缘丢角点的情况。
- 数据筛选只作用于本次标定计算，使用临时 calibrator，不删除原始采集或导入图片。
- 切换策略会清空已有标定结果和导出状态，但保留已采集/导入的检测数据，可直接用同一批图重新标定。
- 标定完成后显示左目内参、右目内参、双目外参的 RMS、等级、分数、关键指标和优化建议。

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
- `scipy` 只补充 `optimize/spatial` 相关子模块，不全量复制 data files。
- OpenCV 动态库、数据及发行包版本 metadata 显式收集；是否可用由最终 EXE 自检确认，而非仅以文件存在判断。

GitHub Actions 打包策略：

- 手动触发默认只生成便携 ZIP，不安装 Inno Setup。
- tag 构建或手动勾选 `installer` 时才生成 Inno 安装包。
- 使用 Windows runner 预装的 GitHub CLI，将 `MultiCamera_portable_<version>_x64.zip`、可选的 `MultiCamera_Setup_<version>_x64.exe` 和 `SHA256SUMS.txt` 上传到 GitHub Release 附件，不再调用 `actions/upload-artifact`，不占用 Actions Artifact 存储额度。历史 Release 的无版本号便携包保持原名。
- 手动勾选 `pointcloud` 时才把 Open3D 纳入 CI 包。
- PyInstaller 打包前安装测试依赖并以离屏 Qt 执行 `python -m pytest -q tests`；测试失败会阻止构建和发布。
- 项目依赖限制 OpenCV 为 `>=4.9,<5`，Windows CI 和本地构建脚本另使用 `packaging/windows/constraints.txt` 固定为已验证的 `opencv-contrib-python==4.13.0.92`。升级该约束时必须重新通过源码和 EXE 自检；不要混装四种共享 `cv2` 命名空间的 OpenCV 发行包。
- 构建前生成 `calibration_source_test.json`，PyInstaller 完成后执行真实 `MultiCamera.exe --self-test-calibration`，生成 `calibration_self_test.json`。后者使用临时工作目录、不依赖相机，不创建 Qt 界面或加载 Open3D。
- 两份报告必须均通过，且 OpenCV 版本、原生库 SHA256、API 清单及 contrib 发行版本一致；EXE 还必须从 `_MEIPASS` 包内加载 OpenCV。缺报告、非零退出、120 秒超时、接口或数值检查失败均阻止压缩和发布。
- 自检报告随 Release 附件发布并纳入 SHA256 清单；失败时尽可能写入报告和 Actions job summary，不依赖 windowed EXE 的标准输出。若进程在 Python 入口前因 DLL 缺失而退出，则无 JSON，CI 仍按退出码/缺报告判失败。
- 手动构建可填写 `audit_release`（例如 `v0.1.0-build.20261020.2`），下载旧 ZIP、核对其校验清单，并在独立 Windows Python 进程加载旧包中的 `cv2.pyd`，输出 `previous_opencv_audit.json`。此报告检查旧原生绑定，不代表直接运行了旧 GUI EXE；探针环境的 Python 包 metadata 与归档二进制版本分开记录。
- 发布 job 使用仓库自带的 `GITHUB_TOKEN` 和 `contents: write` 权限，不需要新增 PAT。相同 ref 的构建串行运行，避免同时修改同一个 Release。
- tag 构建以当前标签创建 Release；手动构建自动使用 `windows-build-<run_id>` 标签，指向本次构建的精确 commit。带 `-` 的标签和手动构建标记为预发布，不覆盖正式 Latest。
- 新 Release 先创建草稿，附件上传成功后再发布。已存在的Release拒绝覆盖；修复后使用新的标签或新的手动构建运行，不移动已发布标签或替换同名二进制。
- 下载入口为仓库的 `Releases` 页面，运行摘要同时输出本次 Release URL。下载 EXE/ZIP 附件，不要将自动生成的 Source code 压缩包当作可运行程序。
- Release 发布失败时 job 保持失败状态，可查看 GitHub CLI 的权限、网络或附件上传错误。旧构建的“Re-run jobs”仍使用旧工作流；切换发布方式后必须用新提交触发构建。

## 10. 导出格式

标定完成后支持：

- JSON
  - 完整 `MultiCameraRig.to_dict()`。
  - 适合程序内部恢复和调试。

- OpenCV YAML
  - 由 `cv2.FileStorage` 写出。
  - 顶层写出通用双目标定字段：`M1`、`D1`、`M2`、`D2`、`R`、`T`、`E`、`F`、`R1`、`R2`、`P1`、`P2`、`Q`、`image_width`、`image_height`。
  - `camera_matrix` 和 `distortion_coefficients` 作为左相机单目标定别名，兼容常见 OpenCV 单目标定读取代码。
  - 多组双目时，顶层通用字段指向第一组 pair；每组 pair 还会写 `pair_<name>_M1`、`pair_<name>_D1`、`pair_<name>_M2`、`pair_<name>_D2` 等别名。
  - 继续保留稳定扁平 key，例如 `camera_<id>_camera_matrix`、`pair_<name>_R`、`extrinsic_<id>_T_world_cam`，便于程序完整恢复多相机结构。
  - 包含 camera matrix、distortion、pair R/T/E/F/Q、rectification 矩阵和全局外参。
  - 启用辅助单目时额外包含 AUX 内参、模态、板型、`T_rgb_left_aux`、`R_rgb_left_aux`、`t_rgb_left_aux`、帧数和 RMS。

- OpenCV 标准 Stereo YAML
  - 作为独立导出选项，面向第三方 OpenCV C++/Python 程序和工具。
  - 只导出一组双目的通用顶层字段，避免项目自定义多相机字段干扰外部读取。
  - 字段包含 `M1`、`D1`、`M2`、`D2`、`R`、`T`、`E`、`F`、`R1`、`R2`、`P1`、`P2`、`Q`。
  - 同时写 `cameraMatrix1`、`distCoeffs1`、`cameraMatrix2`、`distCoeffs2`、`K1`、`K2` 等常见别名。
  - 手动导出必须明确选择单组双目，不再隐式选择第一组；完整多相机结构使用 OpenCV 完整 YAML 或 JSON。底层兼容 API 省略 `pair_name` 时仍保留原先选择第一组的行为。

- Kalibr camchain YAML
  - 生成 `cam0/cam1/...`。
  - 包含 `camera_model`、`distortion_model`、`intrinsics`、`resolution`、`rostopic`、`cam_overlaps`。
  - 从第二个相机开始写 `T_cn_cnm1`。
  - 针孔模型使用 `radtan`；鱼眼模型使用 `equidistant`。
  - AUX 会作为追加相机写入；该扩展用于融合标定结果交换，不声明等同 Kalibr 原生链式采集流程。

注意：Kalibr 导出的 `rostopic` 默认使用 `/<camera_id>/image_raw`，如需对接真实 ROS bag，可在后续增加可配置映射。

### 10.1 手动导出对象与内容预览

“文件 → 导出标定结果”先打开对象/格式对话框，再选择保存位置。清单列出实际包含和排除的相机、参考相机、双目/辅助数量和建议文件名；操作不修改内存标定结果。

- 全部标定结果：包含当前可用的全部双目与辅助结果，可选 JSON、OpenCV 完整 YAML、满足相对外参链条件的 Kalibr。
- 指定双目组：只包含明确选择的该组左右目，参考系重建到该组左目；可选四种格式。即使选择第二组或其他组，也不会静默导出第一组。
- 融合结果：选择某路辅助相机及其参考 RGB 双目基础参数，保留 RGB 右目以完整表达已有双目模型，但跨模态配对仍是 RGB 左目与 AUX。清单明确列出这三路相机，JSON/OpenCV 完整 YAML/Kalibr 均按所选范围输出。

标准 Stereo 只允许“指定双目组”，并明确不含红外/辅助字段。由该模式切换回全部/融合对象时，格式切换为 OpenCV 完整 YAML，内容清单同步更新。Kalibr 在保存前验证相对外参链；互不连通的多组双目不能输出伪造的完整链，可改选单组或连通融合对象。

文件格式由明确选项决定，不再依据用户手输扩展名猜测格式；扩展名不匹配时提示失败。未填写后缀时自动补齐，若补齐后的目标存在则再次确认覆盖。手动导出先在目标目录创建临时文件，写出成功后替换目标；失败弹窗和日志保留原因，不显示“已导出”，原目标文件不被截断。自动保存及现有脚本的导出入口和文件结构不因本对话框改变。

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

- `chessboard`：OpenCV `findChessboardCorners`，使用辅助板独立的内角点列数、行数与格子边长，不再读取 RGB ChArUco 板尺寸。
- `circles_grid`：OpenCV `findCirclesGrid`，辅助板独立设置圆心列数、行数、间距与对称/非对称排列。

### 11.1 红外标定板独立配置与导入

本项目当前红外板规格已确认：**7 列 × 5 行黑白格子，每格边长 80 mm**。OpenCV 检测配置为 **6 列 × 4 行内角点，共 24 点**，内部长度单位为米，即 `square_size=0.08`。不能将格子数 `7×5` 直接填入内角点数量，也不能沿用 RGB 板的 `14×9 / 40 mm`。棋盘上的三维点间距为 0.08 m，第一行第一个到最后一个内角点相距 0.40 m，第一列相距 0.24 m。

在“相机与标定板配置 → 融合相机”选择 `RGB 双目 + IR 单目`，再到“标定板”页编辑独立辅助板。棋盘可按实际格子数或内角点数填写，圆点阵按圆心数填写，间距单位 mm；点击“红外预设：7 × 5 格子 / 80 mm”恢复已确认规格。声像仪 RGB 也可独立设置其实际使用的平面板。多模态步骤显示当前检测板参数；修改 RGB 板不再改变辅助板几何。

“导入板参数...”接受以下文件，不新增 YAML 解析依赖：

- 独立 JSON 参数文件，或含单路 `aux_cameras.<id>.board_config` 的结果 JSON。
- 新会话 `metadata.json` 中的 `auxiliary_camera.board_config`；无辅助相机记录时可从 `multimodal_result.board_config` 读取。
- OpenCV FileStorage YAML 的顶层字段或 `board_config` 映射。
- 兼容 `ustreamer-ir` 红外内参文件中的 `board_cols=6`、`board_rows=4`、`square_mm=80`；其中列/行均为**内角点数**，毫米仅在导入边界转换为米。

项目和分发包包含模板 `assets/boards/ir_chessboard_7x5_80mm.json`：

```json
{
  "pattern_type": "chessboard",
  "cols": 6,
  "rows": 4,
  "square_size": 0.08,
  "asymmetric": false
}
```

兼容的 `ustreamer-ir` OpenCV YAML 板参数示例：

```yaml
%YAML:1.0
---
board_cols: 6
board_rows: 4
square_mm: 80.
```

此入口**只导入板规格**，不套用参考设备的 `K/D`、图像尺寸、Homography 或外参。RGB_L 与 IR 必须在配对图像中观测同一块板；RGB 单独标内参仍可使用原有 ChArUco 板。没有 URL 的辅助相机配置可供离线标定使用，不创建 grabber，也不显示额外的待连接视频通道。

导入时检查点数量为整数、棋盘格每方向至少 3 个内角点、间距为有限正数、单位字段不冲突以及圆点阵类型；界面间距范围为 0.001–10000 mm，精度 0.001 mm，不能表示的导入值会报错而非悄悄截断。错误文件不会覆盖现有参数。多个 AUX 的结果 JSON 不自动猜选相机，应提供独立板文件。缺少历史板参数的旧结果/会话无法凭空恢复实际板规格，需手动核对。

新会话记录 `auxiliary_camera.board_config`；多模态计算结束记录 `multimodal_result.board_config`；结果 JSON 的 `aux_cameras.<id>.board_config` 保留同一套参数。完整 OpenCV YAML 增加 `aux_<id>_board_cols`、`aux_<id>_board_rows`、`aux_<id>_board_square_size_m`、`aux_<id>_board_asymmetric`。标准 Stereo YAML 和 Kalibr 相机链保持原格式，不混入板参数字段。旧结果缺少 `board_config` 时仍可读取，不向历史文件反填新规格。

修改板参数后，需要重新采集或重新导入原始配对图片再标定；检测结果保留板参数快照，参数不一致时拒绝继续标定，导入期间更改参数则丢弃旧任务的检测结果。标定计算期间更改辅助相机、参考相机或板规格，也不会将旧结果应用到新配置。旧标定文件不会自动修正。本轮不自动解决跨模态棋盘格的 180° 角点方向歧义、热对比不足、伪彩处理或硬件同步问题。仿真验证通过不代表这些实拍条件已满足。

### 11.2 RGB 左目与中间红外预览

在顶部选择 RGB+红外/辅助任务，或操作其采集、导入、标定及采图引导时，预览切换为两路：并排模式左侧为已配置 RGB 相机组的左目，右侧为辅助配置中的中间红外相机；红外优先模式主区域显示红外，右上角显示 RGB 对照。RGB 右目不参与该配对，后台收到右目帧时也不会重新插入右目预览。多个 RGB 组存在时沿用配置顺序中的第一组 RGB 左目，不选择 IR 双目作为 RGB 参考。

两路预览均使用辅助板配置检测，当前红外预设显示 `24/24` 而非 RGB ChArUco 的 `126` 点；低分辨率红外保留紧凑提示。预览仍按 15 fps 渲染、2 fps 检测，超宽图缩小后检测，角点映射回原图。采集检测与落盘使用原始 RGB 左目、红外图像，不含预览叠加，也不使用缩小图。两路均检测成功才计入有效配对及覆盖度。

步骤内提供相机配置入口，并明确显示两路相机 ID。辅助 URL 应选择设备发现的真实红外服务地址；中间安装位置不会自动推导 URL，不能把 RGB 右目地址填为红外。辅助相机 ID 和完整视频流地址不得复用 RGB 左右目，但不同通道允许共用同一设备主机。配置应用时若原先正在连接/预览，则重连更新后的流；原先未连接则保持未连接状态。

融合步骤即使缺少红外配置也保留红外占位并提示配置/离线导入；连接失败显示原因并清除旧画面，不以 RGB 右目替代。普通双目步骤不会为无 URL 的离线辅助配置创建连接。RGB 双目与多模态离线预览分别缓存，切换步骤恢复对应图片及覆盖度；连接实时流会清空旧离线预览。进入融合步骤会停止双目自动采集，避免误采右目。

所有相机预览标题栏支持关闭，关闭的是显示而非连接：继续保留原图缓存、采集、保存和覆盖度，新帧/连接信号/离线导入/步骤切换不会重新打开该画面。可用“视图 → 恢复已关闭预览”恢复当前步骤的画面；融合步骤仍只允许 RGB 左目与 AUX，其他已恢复相机在返回普通步骤后显示。关闭后网格自动收拢，全部关闭时显示空预览状态；直接重连保持关闭记录，清空预览或应用新配置则重置。关闭记录仅存在于当前界面，不写入会话；隐藏不停止解码或减少网络带宽。布局刷新采用归属预览控件的单次定时器，避免窗口销毁后刷新失效控件。

预览布局重排、放大和延迟删除期间，控件保持由预览容器作为 Qt 父对象持有；使用取出布局项、隐藏和重新插入来切换，不临时变成独立顶层窗口。相机组状态行按名称复用，连接状态刷新不再反复销毁重建未变化的行。

### 11.3 UI 和线程模型

- 配置窗口“融合相机”页可选择不启用、`RGB+IR` 或 `RGB+声像仪RGB`，独立板规格在“标定板”页编辑。
- 可从发现服务下拉选择热成像/声像仪 RGB 辅助流，也可手动设置辅助相机 ID、URL、平面板类型及独立尺寸。
- `StreamManager` 保留含 `board_config` 的 `AuxiliaryCameraConfig`，有 URL 的实时流会随 `start_all()` 创建 MJPEG grabber；离线配置不创建连接。
- 顶部任务选择提供 RGB+红外/辅助流程，其采集和标定分别显示；AUX 内参和跨模态外参由 `AuxCalibrationWorker` 后台执行。
- 多模态步骤有独立采图引导，覆盖距离、视场位置和倾角；“采集多模态一帧”取 `RGB_L` 与 AUX 软件配对原图，在后台检测并验收。仅质量通过且保存成功的配对纳入标定缓存；详细阶段、计数及失败处理见第48节。
- 建议按引导采满 13 组；至少 3 组有效配对观测即可启用辅助单目标定。

### 11.4 融合采图与标定就绪检查

融合步骤分别显示采图状态和标定状态，而不是仅以有效配对数启用全部操作。状态每 250 ms 更新，连接/断开事件和数据变更也立即刷新；只读取帧时间戳与检测缓存，不额外执行板检测。

- 实时采图：需要配置 RGB 左目与辅助实时地址，两路均未报告断开、存在 1 秒内的新帧，且帧到达时间差不超过 `StreamManager.sync_tolerance_seconds`（默认 350 ms）。缺帧、陈旧帧、非法时间戳或时间差过大时禁用采图并显示原因；即使直接触发采集槽也会再次检查，不保存不合格帧对。
- 辅助标定：需要已有 RGB 左目内参、至少 3 组有效配对、检测板参数与当前辅助板一致；RGB 配对图分辨率需与左目内参一致，辅助图需保持同一分辨率。离线计算不要求视频在线，不能把缺少实时连接作为禁止离线标定的理由。
- 后台导入或标定中禁用冲突操作，完成/失败后重新计算就绪状态，不无条件恢复按钮。配置窗口也受正在执行的任务保护。
- 双目与辅助任务互斥保护其采图、导入、标定和策略修改入口；融合“标定”按钮独立成行，避免窄侧栏中与采图/导入挤在同一行。步骤标题与就绪区共用状态，避免条件不足时仍显示“可标定”。
- 新增有效辅助观测或替换辅助导入数据后，旧辅助/跨模态结果失效，但 RGB 双目结果仍可导出，需重新计算辅助标定后才再次提供融合导出对象。

关键常量位于 `CalibrationPanel`：`AUX_MAX_FRAME_AGE_SEC=1.0`、`AUX_READINESS_INTERVAL_MS=250`。时间差依据接收端时间戳，仅提供软同步保护，不等同硬件曝光同步。此前超出时间容差仅写日志仍保存的行为已改为拒绝采集；设备应先恢复稳定帧流，再继续采图。姿态引导目前仍主要按有效数量推进，不代表已经自动验证全部视场/姿态覆盖。

失败条件：

- 没有完成 RGB_L 所在双目标定。
- 配对目录缺少同编号 `rgb_left_XXXX` 与 `aux_XXXX`。
- 有效平面板配对观测少于 3 组。
- AUX 内参或任一外参 `solvePnP` 有效帧不足。

## 12. 性能观测

`multicamera.perf.perf_timer()` 用于记录慢操作。

应用启动时会写入运行环境信息，包括 frozen 状态、exe 路径、Python 版本、平台、工作目录和进程 ID，以及 OpenCV 版本、加载路径、发行包版本和关键接口是否存在。普通启动不计算大体积原生库哈希；哈希与完整 OpenCV build information 仅在显式自检时生成。日志同时捕获：

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

交互保护、导出对象及就绪门槛回归（无需相机）：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_interaction_safety.py
```

覆盖无改动保留状态、红外独立失效、取消确认零变更、按组及参考相机依赖失效、全局设置确认、任务中配置阻断、离线组保留、会话配置历史、重连覆盖度保留、缺少内参/陈旧帧/错时/断开/分辨率不匹配、离线可标定、手动导出第二组双目、标准 Stereo 范围约束、所选对象 JSON/OpenCV/Kalibr 回读、外参链缺失拦截和写入失败保留旧文件。

2026-10-24 本地验证：上述专项 38 项，含原有测试及新增预览父对象所有权回归的完整测试共 101 项，连续三个独立进程均通过；编译、Ruff、差异空白检查及源码 `--self-test-calibration` 数值自检通过。已检查离屏渲染的导出对象/格式清单和融合就绪界面。`scripts/test_offline_calib.py` 本地找到 0 对图片，不能算实拍离线流程通过；真实相机与 Windows 最终 EXE 仍需验证。

融合预览及相机路由回归（无需相机）：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_multimodal_preview.py
```

覆盖两路角色及显示顺序、右目帧隔离、24 点检测与缩放坐标、缺少红外占位、连接失败清图、离线步骤切换、采集原图逐像素回读、引导切换停止双目自动采集、辅助配置入口、配置后重连以及 ID/URL 冲突拒绝。合成图界面检查不代表已连通设备。实机应在融合步骤核对 RGB 左目和中间红外画面，遮挡每个镜头确认角色，检查红外服务失败时不出现 RGB 右目替代，并用热对比足够的同一块板采集配对图像。

红外板独立配置与导入回归（无需相机）：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_aux_board_config.py
```

覆盖 7×5 格子到 6×4 内角点的预设、256×192 合成棋盘检测、JSON/OpenCV YAML 与 `ustreamer-ir` 字段导入、毫米换算、非法参数不覆盖、独立设置重开、RGB 板隔离、离线无空连接、metadata/结果兼容和完整 YAML 板参数回读；使用 80 mm 板的仿真观测运行真实辅助标定 worker，检查内参和米制外参尺度。真实热成像数据与最终 Windows EXE 的多模态运行仍需另行验证，现有 EXE 烟测仅覆盖 RGB 针孔双目。

自动保存与补拍重标定回归（无需连接相机）：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_calibration_workflow.py
```

该测试使用仿真投影观测调用真实标定 worker，检查未建会话自动保存、重复写入、OpenCV 文件回读、补拍参与内参重算、撤销失效、已有数据直接重跑、预览失败隔离、多组独立导出和保存错误反馈。

OpenCV ChArUco 内参接口兼容性回归：

```bash
.venv/bin/python -m pytest -q tests/test_intrinsic_compat.py
```

使用 24 帧部分/乱序角点观测验证标准与稳健策略在旧 ArUco 标定接口不存在时仍可求解，并与旧接口可用时的结果做数值对照。Windows CI 同时执行这些测试和 UI 工作流回归，不再仅以 EXE 能生成作为发布通过条件。

推荐使用项目虚拟环境：

```bash
.venv/bin/python -m compileall -q src scripts
```

### 13.1 最终 EXE 标定自检与 OpenCV 版本排查

源码环境执行同一套自检：

```bash
.venv/bin/python -m multicamera --self-test-calibration --report /tmp/calibration_source_test.json
```

在 Windows 便携包目录手动执行（也适用于安装后的 EXE）：

```powershell
$exe = (Resolve-Path .\MultiCamera.exe).Path
$report = Join-Path $env:TEMP "multicamera_calibration_self_test.json"
$p = Start-Process -FilePath $exe -ArgumentList @("--self-test-calibration", "--report", "`"$report`"") -PassThru -Wait
$p.ExitCode
Get-Content $report -Raw
```

不传 `--report` 时写入当前工作目录的 `calibration_self_test.json`，现场排查建议显式指定可写路径。退出码 `0` 表示通过，`1` 表示运行/接口/标定失败，`2` 表示报告无法写入。必须等待 windowed EXE 结束，不能只依据进程启动成功。

自检生成 ChArUco 图案并检测 126 个角点；使用 24 帧部分、乱序且左右不完全相同的角点观测，调用真实内参、双目和校正代码，检查 RMS、已知内参、0.12 米基线与有限校正矩阵；导出四类文件并回读 JSON/OpenCV 矩阵和左右 ID，检查 Kalibr 链字段。所有模拟数据保存在临时目录，不修改用户会话。该烟测不替代真实图片质量评估，不覆盖鱼眼数值标定、点云、视频连接或安装器安装过程。

本次问题排查的实测证据：旧 Windows 构建 #10 的依赖日志安装 `opencv-contrib-python 5.0.0.93`，而本地验证使用 `4.13.0.92`；原 `>=4.9` 未限制主版本。Windows 构建 #12 下载并校验旧 Release `v0.1.0-build.20261020.2` 的 ZIP，直接加载其 `cv2.pyd`，测得版本 `5.0.0`，旧 `aruco.calibrateCameraCharuco=false`，通用 `calibrateCamera/stereoCalibrate=true`。旧 ZIP SHA256 为 `7fe8e4c3703a29b21fa42b20ceea65475a0fc38859ad044d1cbbb1f0e7e43bb8`；其原生库 SHA256 为 `ae833b9b5771f5932200bd3b8ad0f483f4fbec92cbdbd26d21d38fe524c60e66`，与从 PyPI 下载的 `opencv_contrib_python-5.0.0.93-cp37-abi3-win_amd64.whl` 内 `cv2/cv2.pyd` 完全一致。结论为跨主版本更新后的旧接口不可用，并非 OpenCV 未打包，也非 PyInstaller 裁剪了该函数。

新发布包 `windows-build-39475861438`（应用源码 `a51f264`）已通过最终 EXE 自检，报告 `frozen=true`、contrib `4.13.0.92`、OpenCV `4.13.0`。源码和 EXE 原生库 SHA256 同为 `ac679c0d19b50c03a0f8237ee6a1b87be3bf9cf9c9c94d206d9ccb7f5956de75`；左右内参和双目 RMS 分别约 `0.00001938`、`0.00001852`、`0.00014035` 像素，恢复基线 `0.11999995` 米。构建 #12 总用时 4 分 42 秒，已发布约 93 MB 安装器、143 MB 便携包及三份诊断报告。构建证据：[Actions #12](https://github.com/q1234ew/MultiCameraStereoCalib/actions/runs/39475861438)；下载：[Release](https://github.com/q1234ew/MultiCameraStereoCalib/releases/tag/windows-build-39475861438)。这些数值来自确定性仿真，不代表用户实际采图质量。

升级 OpenCV 时应保留源码基线、EXE 报告和构建链接，确认产物版本与实际使用接口；只重跑源码测试或检查 EXE 是否存在不足以验收打包兼容性。

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

工作台交互专项及离屏截图（合成数据，不连接设备）：

```bash
MULTICAMERA_UI_QA_DIR=/tmp/multicamera-workbench-qa .venv/bin/python -m pytest -q tests/test_workbench_ui.py
```

覆盖 1440×1024、1280×800、1000×680 的并排/红外优先/标定状态，及结果保存摘要、缩略图回看、最新原图采集、任务切换、预览关闭恢复、日志折叠和有限缓存。截图是 Qt 真实控件离屏渲染，不能替代 Windows DPI、真实视频流或最终 EXE 验证。

配置专项：`MULTICAMERA_UI_QA_DIR=/tmp/multicamera-config-qa .venv/bin/python -m pytest -q tests/test_configuration_ui.py`。覆盖三页在 1080×760、900×640 的横向适配和固定按钮、毫米重开、格子/内角点换算、圆点阵与预设、非法参数不应用、旧错误地址修复、离线组、保留发现选择、同机唯一配对与去重。测试用服务对象不发起真实网络扫描。

macOS 原生布局专项：`QT_QPA_PLATFORM=cocoa MULTICAMERA_UI_QA_DIR=/tmp/multicamera-config-layout-native .venv/bin/python -m pytest -q tests/test_configuration_ui.py::test_fusion_fields_expand_without_reference_overlap tests/test_configuration_ui.py::test_configuration_pages_fit_and_footer_stays_visible`。会短暂显示测试窗口，检查两个尺寸下的宽字段、长参考 ID、相邻控件无重叠以及三页无横向溢出，不连接真实设备；截图按 Retina 实际像素输出。

白平衡专项：`.venv/bin/python -m pytest -q tests/test_white_balance.py`。使用受控参数客户端和时钟验证双路稳定后锁定、不回写色温、锁定后颜色跳变/不稳定/单路接口失败/取消时恢复自动、空目标、worker 失败不恢复采图及过期结果作废。工作台回归另验证融合请求准备两路 RGB 而非 IR、准备中不新增观测、新会话清理锁定记录和显式关闭选项。服务脚本检查：`bash -n /Users/fuqiang/Dev/RK1126B/FusionSensor/ustreamer-ir/start_fusion.sh`。这些测试不代替真实 UVC 锁定验收。

当前环境中如果 `pytest` 或 `ruff` 未安装，应先安装 dev 依赖：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

## 14. 手工回归清单

工作台：

- 默认并排；任务与“采集 / 标定 / 结果”切换保留已有数据，普通双目不显示辅助路。
- 红外优先时 RGB 左目进入右上对照，关闭/恢复、放大/退出不出现独立残留窗口。
- 窄窗口右侧可滚动，主操作不被滚动隐藏；相机名称悬停可见完整 ID。
- 点击配对缩略图显示回看标识，返回实时恢复；回看时点击采图保存的是最新原图而非历史图片。
- 新增/替换数据不会继续显示旧结果为当前结果；结果页保存异常明确可见，导出和打开目录对应实际结果。
- 日志默认收起，保存/连接失败自动展开；生成标定板和点云入口可从工具菜单访问。

相机配置：

- 三个页签和固定保存区在小窗口可用；切换页面不丢失尚未应用的输入，取消不改动实际配置。
- RGB 40 mm / Marker 30 mm 重开后仍为同一尺寸；红外 7×5 格子与 6×4 内角点切换保持 24 检测点和 80 mm 间距。
- 相机发现刷新不改变已选服务；未知目别、多候选或不同主机不会被自动拼成一组；手动确认可跨主机添加。
- 错误尺寸、容量不足或地址错误显示具体原因，修正后恢复保存；没有视频地址的离线组不产生连接。

- mDNS/子网扫描能发现设备。
- 自动配对能生成左右 ID 和 URL。
- 手动添加相机组后保存，右相机 ID 不被自动改写。
- 不改设置直接确认，不重连也不清空数据；只改辅助地址，RGB 双目标定和数据仍保留。
- 有工作时修改来源会先列出影响，选择不应用后连接/数据不变；确认后只清理相关组，磁盘原图仍存在。

实时采集：

- 多路 MJPEG 连接后预览稳定刷新。
- FPS 显示正常。
- ChArUco overlay 低频更新但不阻塞 UI。
- 点击“采集一帧”后进度、覆盖率、质量提示正常更新。

离线导入：

- 导入左右图后出现 `offline` 虚拟相机组。
- 可选择数据集父目录，程序递归识别 `*_left/*_right`、`left_*/right_*`、`*_L/*_R`、`left/right` 子目录。
- 再次导入会清空旧的 `offline` 缓冲并替换为新数据集。
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
- 手动导出先选择对象，核对包含/排除清单；标准 Stereo 只允许单组，选择第二组后文件内 ID 与第二组一致。
- 融合导出清单包含参考 RGB 双目与所选 AUX；相机链不连通时不能导出完整 Kalibr 链。
- 手动导出到不可写目录或格式/后缀不匹配时有明确失败提示，已有文件保持不变。

多模态：

- 配置窗口可启用 `RGB+IR` 或 `RGB+声像仪RGB`。
- 导入 `rgb_left_XXXX.*` 与 `aux_XXXX.*` 后有效配对数正确显示。
- 未完成 RGB 双目标定时辅助标定按钮禁用，就绪区说明缺少 RGB 左目内参。
- 两路帧时间差过大、帧超过 1 秒或红外断开时禁用实时采图；已有合格离线数据仍可计算标定。
- 辅助单目标定期间按钮禁用，完成后恢复并启用导出。

## 15. 修改记录维护

项目修改记录维护在 `docs/CHANGELOG.md`。

需要同步记录的变更包括：

- 用户可见的功能入口、按钮、菜单、流程或提示文案变化。
- 标定算法、筛选策略、质量评估、导出字段或文件格式变化。
- 会话目录、metadata、rig JSON、OpenCV/Kalibr YAML 等数据结构变化。
- 日志、打包、启动性能、运行性能和平台兼容性相关优化。
- 修复会影响标定结果、导出结果或设备连接的问题。

每条记录应包含变更摘要、兼容性影响、验证方式和后续注意事项。提交代码前应检查修改记录是否需要同步更新。

`docs/CHANGELOG.md` 必须追加式维护。新增记录应插入为新的日期条目，或仅补充尚未提交的最新条目；不得覆盖、删除、合并或重写已有历史记录。

## 16. 当前边界与后续方向

当前边界：

- 点云生成、视差、Open3D 融合本身尚未做性能优化。
- 多模态功能仅产出内参和跨模态外参，不包含点云/声学融合算法。
- 性能参数还是代码常量，尚未提供 UI 配置。
- 已有仿真标定 worker、导出回读和离屏 UI 回归；真实设备、Windows 缩放/键盘交互及全场景线程压力仍需实机验证。

建议后续优先级：

1. 补充 Windows 最终 EXE 的真实网络流和多模态交互验证。
2. 将性能参数暴露到配置 UI。
3. 扩展保存队列、导入和标定任务并发/取消/退出场景的压力测试。
4. 使用实拍热像验证跨模态方向一致性、不同热对比与数据质量边界。
5. 如果需要继续降启动内存，可继续拆分 UI 模块导入路径。

## 17. RGB 双目采图工作台（2026-10-26）

双目采集页采用已确认的横向双路预览设计：顶部仅显示当前采图建议；下方依次为相机/板位置线稿、当前姿态、预览检查状态；覆盖分析与采集设置默认收起；配对缩略图和固定操作区位于底部。小窗口的预览及引导区允许滚动，采集、白平衡准备及取消按钮不随该区域滚动。切换到融合采集、标定、结果或点云页时恢复原侧栏；不更改这些任务的计算流程。

### 17.1 建议与测量的边界

- 当前仍为固定 18 步启发式 RGB 引导，不是信息增益最优选姿态。导航和跳过仍不等于有效采集，全部浏览完也不表示质量达标。
- 九宫格行标签为画面上/中/下，不能解释成远/中/近。青色虚线仅表示建议图像区域，不是已标定的三维板目标投影；黄色轮廓表示实际检测到的角点凸包，不是完整标定板外边框。线稿为非比例示意，不能用于量测。
- 当前板姿态仅在已有该路内参、检测有效且图像尺寸匹配时通过 `solvePnP` 估计；鱼眼观测先去畸变到归一化坐标。显示的米制量是板原点的相机 Z 深度，左右/上下倾角由板面法向计算，不包含面内旋转。没有内参时显示“待内参”，断流/失效时显示等待检测，尺寸变化不复用旧数值。当前不在采集中自动生成初始内参，不虚构“内参已就绪”。
- 多组任务的数值姿态和覆盖摘要展示首个所选组，自动采集检查所有所选组；详细目标与最少帧数仍可在折叠区设置。

### 17.2 预览与自动采集

复用预览已有的限频角点检测结果，不再创建检测线程或重复执行检测。UI 每 250 ms 更新状态；预览采样超过 1.5 秒或相机缓存帧超过 1.5 秒即不可用于就绪判断。清晰度以角点 ROI 的 Laplacian 方差估计（ROI 长边最多 320 px，阈值 50）；稳定性按相邻检测的至少 6 个共同 ID 角点位移判定（最大位移小于长边的 0.8%）。这些均为可调启发式阈值，不是精度保证。

自动采集默认关闭，每 2 秒检查：所有所选相机组的实时观测有效、清晰、稳定，角点中心接近建议区域（归一化中心距离小于 0.16），且通过现有软件时间戳配对接口。该条件不等于硬件同步，也不验证倾角是否达到目标，因此倾角步骤暂停自动触发并提示手动确认采集，避免固定板姿态被重复记为不同倾角。回看、切换标定/结果任务、断流或失效数据不得触发自动采集。白平衡准备后恢复自动任务时重新检查就绪条件。手动采图保留原始分辨率检测、会话创建、白平衡准备、保存队列和撤销路径。

### 17.3 回归验证

- `tests/test_stereo_capture_ui.py`：布局往返恢复、折叠详情、三种窗口尺寸、PnP 已知深度、缺失/不匹配内参、过期观测、回看、自动采集开关、源图不被覆盖层修改。
- 原生 Qt 截图检查尺寸为 1440×1024、1280×800、1000×680；验证小窗口可滚动、固定按钮未被压缩遮挡，按配置左右角色排列预览，而非按 ID 字符序排列。
- 未接入 mrcal/MC-Calib，不改变标定求解器；不包含 RGB–IR 联合优化或点云性能优化。真实相机、Windows EXE、高 DPI 和全场景长时稳定性仍需单独验证。

### 17.4 QPainter 动态相机/标定板示意

`ui/widgets/stereo_schematic.py` 替换原静态线稿，不引入 WebView、Open3D 或 3D 引擎。展开模式左侧为起始姿态、右侧为计划动作演示，紧凑模式只保留动作；两个视图使用同一套几何与固定观察视角。相机为一个机身、两个朝向板面的镜头，机身不随动画移动。窗口缩放仅对几何使用等比例投影，标题保持原生字体尺寸；各自绘图区裁剪，避免拉伸及线条越界。紧凑布局与滚动边界见 17.5。

- 正面步骤：从中心向计划九宫格位置平移；中央正面步骤保持静止，不启动动画计时器。画面上/下移动不解释为远/近。本轮不在 18 步计划之外增加距离或面内旋转任务。
- 倾角步骤：起始板已处于建议区域，仅演示旋转，不同时平移。左/右/上/下倾分别明确表述为对应边靠近镜头，以对侧边为固定轴。顶部建议和示意说明使用同一方向定义，倾角步骤仍需人工确认采图。
- 蓝色虚线为预设终点轮廓，黄色为移动边/运动箭头，板面网格只是定性线稿，不替代已配置的真实 ChArUco 检测板。示意中采用固定的任意尺度与示例转角（左右 30°、上下 25°），既不是内参投影，也不是实时测得姿态或强制验收阈值。
- `QTimer` 每 33 ms 推进单调时间插值；6 秒循环由起始停留、移动、目标停留、返回组成。支持暂停/继续和重播；同一步的 250 ms 状态刷新不重置动画，切换步骤才重置。隐藏页面、回看或没有计划时停表，恢复页面不累计隐藏时长；中央静止步骤禁用播放控件。只重绘示意区，不读取相机、不重复检测、不改变采集记录。
- 暂停、播放、重播使用离线 Lucide 图标，与既有图标同一固定版本及许可证，无网络依赖。真实 PnP 数值及自动采集判据仍按 17.1–17.2 执行；动画到达终点不触发采图，也不修改有效帧数。

验证：`tests/test_stereo_schematic.py` 覆盖四种倾角的固定轴、移动边深度/刚性距离、九宫格平移与全部计划点投影范围、计时器启停/暂停/重播/回看、像素非空与起始视图不随动画变化。配合 `tests/test_stereo_capture_ui.py` 检查三种窗口尺寸、滚动查看动作、真实姿态和原采集路径。可用 `MULTICAMERA_UI_QA_DIR` 指定截图目录；macOS 原生验证设置 `QT_QPA_PLATFORM=cocoa`，默认无屏插件下原固定操作按钮高度检查可能与原生不同。

### 17.5 评审改进：质量口径、紧凑布局和绘制减载

**质量与方向一致性**

- `IntrinsicCalibrator.compute_coverage()` 保留原接口和值，明确为出现过的板角点 ID 比例（板角点完整度），不再将其展示为视场覆盖。新增 `compute_image_coverage()`，统计 4×4 图像网格中至少出现过一个角点的格子占比；该量也不是连续面积或姿态覆盖率。完整板只在中央出现时，完整度可以为 100%，视场网格仍然很低。
- 数据评估分开展示完整度、视场网格、均匀度和可用帧数；评分的覆盖分量使用视场网格。均匀度改为相对均匀分布的归一化 L1 距离：所有点集中一格为 0，全格等量为 1，修正旧计算在严重集中时仍偏高的问题。数据评分可能比历史版本降低，不影响标定求解器、原始数据或 RMS。
- “基础画面检查”明确不含姿态。姿态单独显示“待人工确认”，无有效观测时为“待实时检测”；即使有已标定内参和 PnP 数值，也不自动宣称目标角度达标。紧凑摘要保留该状态，数值姿态可在摘要提示中查看，展开布局显示完整数值。
- `board_motion.py` 是 `BoardTilt`、动作文字与固定边旋转的公共定义。旧侧栏、放大引导及视频摆放示意的板四角从同一三维动作沿镜头方向投影，靠近镜头的一边应更大；原导入路径继续可用，没有改变采图计划或持久化格式。

**重复构图提醒**

`find_similar_detection()` 比较最近最多 50 个样本中相同 ID 的像素位置；要求图像尺寸相同，至少 6 个共同点且共同点不少于双方较多角点数的 60%，共同点二维 RMS 小于图像长边的 1%。双路均匹配时实时显示提醒；实际原分辨率采图完成后再检查一次并记录左右样本序号。提醒不删除、不拒绝采图，不改变保存队列、撤销或自动采集入口。它只判断近似构图，不证明图像重复、几何退化或已具备足够姿态多样性。

**小窗口布局和资源管理**

- 双目采集窗口高度小于 900 或宽度小于 1150 个逻辑像素时，采用左右区域分隔：左侧保留左右目预览，右侧为紧凑动作及质量摘要；正常尺寸仍采用预览上、完整对照引导下。分隔条可拖动，动作区有独立展开起始/动作对照按钮。极小窗口或展开配置时仍允许滚动，不通过缩小文字强塞内容。
- 紧凑模式收起图库但保留入口，底部白平衡/保存状态与采集操作横向组织；离开紧凑模式恢复图库原展开状态。复用原控件和信号，不创建第二套采集按钮；白平衡准备进度及取消入口保留。
- 相机、参考网格、起始板和固定文字缓存为带设备像素比的 `QPixmap`；动画帧只绘制动作板及目标/箭头。尺寸、字体、调色板、目标或模式改变时重建缓存，普通进度帧不重建。完整度/视场摘要以采集器 `data_revision` 缓存，增加、撤销或清空样本后失效。
- 动作滚出视口、隐藏页面、回看或手动暂停时停止动画；滚回后按单调时间继续，不累计不可见时长。切换建议保留手动暂停选择，重播才主动恢复。视口延迟检查使用控件持有的单次 `QTimer`，控件销毁后自动取消，避免关闭窗口后回调已释放的 Qt 对象。

**验证与下一步**

新增指标/重复构图/方向一致性回归，扩展静态缓存复用、滚出视口停表、暂停保持、小窗口预览与示意同时可见、重复采图保留及撤销等测试。原生 Qt 专项 27 项通过，完整回归结果见修改记录。本轮未接入下一最佳姿态算法或自动姿态验收；下一步应基于有效内参、重投影检查和已采姿态记录，评估距离/倾角分布与重复姿态，之后再考虑信息增益选点。mrcal 在独立离线任务中研究，不作为本界面的运行依赖。Windows EXE、真实设备及持续 CPU/延迟对比仍需单独验收。

本机局部离屏绘制对照（800×350 控件、120 帧）中，复用缓存平均约 0.51 ms/帧，强制逐帧重建约 2.30 ms/帧，普通动画帧没有额外重建静态缓存。这是同一实现的缓存开关对照，不是旧版整体性能基准，也不能外推为 Windows 多路相机 CPU 或端到端延迟改善比例。

### 17.6 双目相机可视化模型

相机形状由 `stereo_camera_model.py` 的轻量网格生成：单个倒角机身、两个沿相机 +Z 方向伸出的镜筒、外圈和镜片层次。按固定观察位置剔除背向面，再将表面按观察深度由远及近绘制，取代原来前后轮廓相互穿插的线稿。表面采用低饱和灰蓝填充和细轮廓，保留技术示意风格，不使用外部网格、Open3D 或实时 3D 渲染依赖。

- 固定观察位置为镜头前上方斜视，以同时看到两个镜头端面、机身顶盖和厚度。相机坐标 Y 向下，观察点 Y 必须为负；曾使用正 Y 的前下方观察点，导致展示底面而出现上下颠倒感，现已修正，不通过翻转像素掩盖视角问题。与此前后侧示意相比仅切换观察角度，不改变实际相机、预览或标定数据。模型与板面使用同一投影，完整 18 步仍检查投影范围。
- 两条短虚线及箭头表示平行光轴朝向标定板，不是视场锥，也不是从内参计算的真实光路。机身尺寸、镜头间距与外形均为示意，不从外观推导实际基线。
- L/R 表示左/右预览相机角色；由于从镜头前侧观察，L 可能出现在示意的屏幕右侧。板边方向仍以左目预览为准，不交换真实左右流。说明同时提供可访问描述和提示，角色字母以屏幕空间固定字号绘制。
- 模型网格在进程内复用，投影后的机身与光轴仍进入原有高 DPI 静态缓存，不在每一动画帧重新建立。动态板、暂停/重播、视口停表与采集操作保持原行为；本轮未提供相机型号导入或实际机身尺寸配置。

`tests/test_stereo_camera_model.py` 验证一个机身/两个镜头、平行前向镜筒、有效面法向、背面剔除、投影边界和 1×/2× 像素绘制，并断言可见的是顶面而非底面、相机与板面的垂直轴没有倒置。配合原几何、缓存、计时器及工作台专项检查；本轮组合原生 Qt 测试曾在后续窗口构造的图标渲染入口发生原生崩溃，独立专项结果不作为整体稳定性已解决的证明。详细结果及未验收平台见修改记录。

## 18. 真实离线素材的 UI 流程验证（2026-10-26）

### 18.1 离线与实时状态隔离

`StreamManager.is_virtual_pair()` 作为离线相机组的判据，不用连接状态或是否暂时收到帧猜测数据来源。导入完成自动选择 `offline` 组，已有实时相机组、数据和引导记录保留；切回实时组后恢复实时控制。仅选择离线组时：

- 预览明确标记“离线”，状态点不显示相机已连接，底部显示分辨率和检测角点数；原图上不绘制下一次实时采图的目标区域。
- 顶部和检查摘要显示加载数量、有效配对、角点完整度/视场网格及“姿态覆盖未验证”。不再对静态图像提示实时运动稳定性，隐藏动态采图动作和白平衡/自动采集/跳过按钮，以“前往标定”为主操作。
- 原图提示为保留在源目录，完整路径通过提示查看。导入本身不复制到新采集会话、不修改源图；标定结束仍沿用自动创建结果会话并保存结果的逻辑。
- 导入和交换只更新离线观测，不重置或推进实时 `CaptureSequence`，不把导入数量当作已验证的姿态步骤数。失败导入保留上一次有效图片、结果及来源，日志明确带“离线导入失败”，并展开错误日志。
- 少于五对时不会将标定器的最少样本数降到控件下限以下。修改“最少配对数”同步更新内参/双目标定器与按钮就绪状态，后台计算期间禁用该设置。

### 18.2 回看、配置与导出

- 图库使用导入记录的原始编号，例如素材缺少 `0009/0010` 时，下一组显示 `0011`，不重新编号成 `0009`。新导入默认从第一页及第一对开始；回看选择同步当前离线索引，结束回看或交换左右后仍定位到相同的素材。交换不修改文件，原有标定结果失效，必须重新计算。
- 用户在紧凑模式下主动展开/收起图库的选择会保留，切换标定/结果页不再将已收起的图库自动弹开。纯 RGB 任务隐藏无关的红外优先切换。翻页控件补充可访问名称，左右交换采用同版本 Lucide 双向箭头。
- 标定页展示当前针孔/鱼眼模型和 ChArUco 规格，并提供“模型与标定板”入口，复用原配置窗口。取消配置不丢失已导入数据；保存会影响观测的模型/板参数仍执行原确认和失效规则，不自动猜测镜头模型。
- 标定行“已计算”与质量合格分开：颜色按左右内参和双目误差中最低分项显示，不再对高误差一律显示绿色。加载数、可用数与就绪条件统一以“对”为单位。结果页保持原有质量风险，不因导出文件生成成功而显示为合格。
- 单组标定按钮使用最小宽度而不是短文案固定宽度，随“重新标定”等状态文案保留足够空间；离线状态回归检查按钮实际宽度不小于样式建议宽度。
- 导出窗口补充保存/取消图标和主按钮状态。标准 Stereo YAML 仍须明确选择具体双目组，禁用项提示选择条件；默认文件名包含所选对象 ID，替换文件名中的跨平台非法字符，降低多组文件混淆的风险。现有原子写入、覆盖确认和格式内容不变。

### 18.3 验证范围与复现

实际使用 `data/pairs` 中 16 对 1920×1080 素材，包含 `0008` 到 `0011` 的编号断档。原生 Qt 测试创建独立窗口，仅替换文件选择器的目录返回值；图片读取、检测、后台导入/标定、自动保存和标准导出使用实际代码。通过鼠标点击图库、切换页面、交换/恢复左右、打开并取消配置，以及空目录导入验证交互；截图由该窗口的 `QWidget.grab()` 生成。桌面控制连接未能读取正在使用的 Python 窗口，因此这不是对用户当前窗口的人工操作验收，也未关闭或替换其会话。

```bash
QT_QPA_PLATFORM=cocoa \
MULTICAMERA_REAL_PAIR_DIR="$PWD/data/pairs" \
MULTICAMERA_UI_QA_DIR=/tmp/multicamera-real-pairs-final \
.venv/bin/python -m pytest -q tests/test_real_pairs_ui.py
```

专项保存导入、回看、交换、标定、结果、配置、导出及失败保留数据等截图和 `observations.json`；覆盖 1000×680、1280×800、1440×1024 三种窗口尺寸，并读取导出的 `M1/T` 等矩阵。逐文件 SHA-256 校验确认 32 张源图未被修改。普通 CI 不要求本地素材，未设置环境变量时该项明确跳过；合成状态回归 `tests/test_offline_ui_state.py` 始终执行。

本组素材在默认针孔模型、默认板参数及标准策略下得到双目 RMS 约 26.972 px，界面标为“不建议直接使用”。这只证明功能链路可运行，不证明该组标定可用；应核对实际镜头模型、板参数和采图条件，不能通过调整 UI 质量阈值掩盖问题。没有进行第三方工具、Windows 原生文件对话框或真实相机验收；完整回归、依赖跳过和已有 Qt 原生间歇崩溃记录见修改记录。

## 19. 双目标定可靠性循环优化（2026-10-26）

### 19.1 范围与输入契约

本轮优先处理 RGB 双目标定的输入有效性、求解一致性、可追溯评估、保存及后台生命周期，不接入 mrcal/MC-Calib，不优化点云或改变 RGB–IR 求解算法。并行 mrcal 研究文件不由本轮维护。辅助标定仅修复会话记录失败后忙碌状态无法释放的问题。

- 内参与双目观测入库前校验坐标/ID 形状、数量、有限值、ID 整数范围和唯一性，以及正整数分辨率。同一路数据不混用不同分辨率；双目左右必须同尺寸，不能将尺寸不一致的观测交给 OpenCV。
- 内参拒收时返回 `False` 并记录 `last_rejection_reason`；双目格式错误抛出 `ValueError`，面板逐条记录且继续导入其余图片。无检测或点数不足仍属于普通不可用观测。
- 递归导入遇到重复配对编号时，以左图相对文件路径区分来源，不重排普通单目录编号；存在仅大小写不同、无法可靠识别的路径时拒绝导入。未配对图片数量在日志提示。原图路径不能仅由参与序号推断。
- 入库坐标/ID 由标定器持有独立副本。原图缓存与观测索引对齐，无原图时保留 `None` 槽位，避免质量评分和撤销错用其他图片。
- 就绪条件同时检查左右内参与双目有效配对。图片加载完成不等于可标定；最少数量不足时不会自动降低硬下限。

### 19.2 后台筛选与求解

本次标定用的质量评分和筛选在 `PairCalibrationWorker` 中执行，不在点击按钮的主线程计算整批 Laplacian。离线导入首次质量统计也由导入 worker 计算、主线程读取已完成缓存；交换左右同步交换质量记录，不重复计算整批图像。默认“全部使用”；“自动质量筛选”按角点完整度及清晰度/亮度的组合评分排序，双目配对使用同一源编号两路图像质量的较低者。排序策略保留约 80%，但不低于最少样本数；这是启发式排序，不是几何外点剔除、姿态最优选择或独立精度保证。无原图时使用中性图像分值，并保留原始检测数据。

标准策略固定单目内参；联合精修允许双目阶段更新内参。求解器只接收内参数组副本，返回的 K/D 经校正成功后才写回；校正矩阵、映射、导出的内参保持同一版本。联合精修改变内参后，用原选用观测重新估计板姿态并计算逐帧误差，不保留旧 K/D 对应的误差或位姿。

鱼眼和针孔的 OpenCV flags 必须分属各自命名空间。鱼眼稳健策略固定 `cv2.fisheye.CALIB_FIX_K3/K4`；针孔稳健策略约束切向及高阶畸变。这里的“稳健”是模型约束，不等同 RANSAC 或鲁棒损失。

### 19.3 鱼眼变长角点的显式兼容

当前 OpenCV 4.13 鱼眼双目实现要求不同帧的角点数量相同，而 ChArUco 部分检测天然可能变长。默认策略在求解前列出源编号/点数并明确拒绝，不再暴露 `matrix_wrap.cpp` 断言，也不静默丢帧、截断或复制角点。针孔没有这一跨帧等点数要求。

鱼眼模型可显式选择“鱼眼兼容：跨帧共同角点”：

1. 对全部有效配对的真实共同 ID 再求跨帧交集。
2. 每帧同步选择对应物点、左点、右点，保留原帧顺序与全部帧。
3. 交集少于 6 个点时停止，提示使用覆盖共同板区域的数据；不为了凑数制造观测。
4. `calibration_info.selection.stereo` 保存 `input_corner_counts`、`used_corner_counts`、`shared_corner_ids`。仅双目阶段减少点；单目内参仍使用原观测。原图和内存原数据均不改变。

同数量并不要求跨帧 ID 相同，但每帧内部三组点必须准确对应。此兼容模式明确选择 ID 交集以保证可审计性，不是支持任意变长数据的新优化后端。它会减少空间信息，应重新检查覆盖、残差及独立验证图像。

### 19.4 误差口径与逐帧诊断

新生成的逐帧误差为 `sqrt(mean(dx² + dy²))`，单位像素，与 OpenCV 的点距离 RMS 口径一致。旧实现 `sqrt(mean([dx²,dy²]))` 比该值小约 √2 倍；旧文件不重写、不直接跨版本比较。新增可选 JSON 字段 `frame_ids`、`per_frame_error_metric="euclidean_rms_px"`；旧文件缺失时仍可读取，但不能把参与序号当作源文件编号。

双目 JSON 的可选 `calibration_info` 记录 OpenCV 版本、策略、实际 flags、模型、板参数和各路选用/排除源编号。离线使用文件配对 ID，实时采集使用本会话落盘图片索引。原有字段及 YAML 核心矩阵保持兼容。Worker API 接收已有内参时在副本上运算，固定内参保留真实或未知来源记录，不冒充本次筛选所得；联合重算后标记 `refined`，区分初始来源和实际复核的观测编号。

“逐帧误差检查”窗口支持数值排序、左右单目误差、双目校正残差、各阶段参与状态和可靠匹配原图的跳转。无值显示横线，不代表零误差；旧文件未记录源 ID 或当前内存无图时，不猜测原图。窗口仅供检查，不自动删除/剔除图片。标定输入/结果失效或启动新计算时关闭旧诊断窗口。

`calibration_info.rectification` 使用选用双目观测的相同 ChArUco ID 计算校正残差，不使用 SIFT 匹配。根据 P2 判断极线方向：水平双目比较 y，垂直双目比较 x。保存样本数、逐帧及整体 RMS、P95、最大值、非有限点数；不裁剪异常值。统计位于校正像素坐标系，受校正尺度影响，不是物理空间距离，也不是独立验证精度。非有限结果判为无效；较高的有限残差给出复核提示，不改变原重投影 RMS 分级阈值。

结果评估同时检查有限数值、畸变向量、模型/分辨率一致性、SO(3) 旋转、三维平移及非零基线。负数、NaN、Inf 误差不会获得高分；计算完成和文件生成不等于质量合格。

### 19.5 会话、保存与关闭

- 标定、导入、保存、白平衡准备、自动采集或排队任务尚未安全结束时，拒绝新建/加载其他会话和关闭窗口，给出日志/状态提示。检查包括结果信号已发出、但 `QThread.run()` 仍未返回的尾部窗口；不调用 `terminate()`，也不在 GUI 中无限等待。
- `set_session()` 有归属校验兜底；空闲切换会话清理旧撤销记录。保存 token 和撤销条目明确绑定原会话，不在回调中取当前会话删除图片。首次自动采集先创建会话再启动定时器。
- 撤销时若磁盘删除失败，明确区分内存撤销和磁盘失败，仍完成旧标定结果失效、导出禁用及就绪刷新；不能因 I/O 异常留下与新内存观测不符的可导出结果。
- 相同秒内新建会话使用不同目录后缀，不复用旧目录。加载前检查 `session.json`，无效会话不创建新的子目录。
- 首次会话 metadata 即记录当前镜头模型，不等待后续配置变更补记。
- 图片先成对暂存，再提交并原子更新 `session.json`；保存失败回滚本次已提交图片和内存计数。已存在索引明确拒绝覆盖。若磁盘自身连清理都拒绝，日志保留未回滚路径，不宣称跨文件/断电 ACID 事务。
- 标定格式独立导出：一个格式失败不阻断其他结果。`calibration_manifest.json` 记录本次 generation ID、文件名及 SHA-256、失败清单。只有列出且哈希匹配的文件属于本轮；历史多组/单组文件不自动删除，不可按“目录里仍有文件”判断其为最新结果。清单自身写入失败也会提示。
- 暂存目录提交后的清理失败只记日志，不能反向删除已成功提交的图片或回退内存计数。清单哈希读取、生成和写入统一计入清单错误，仍返回此前已成功的结果文件。

### 19.6 第三方导出约束

标准 OpenCV YAML 在原 M1/D1/M2/D2/R/T 等字段之外，明确写入相机模型和左右畸变模型。鱼眼消费者必须使用对应 `cv2.fisheye` 或等价模型，不能因为文件可以 FileStorage 读取就按普通针孔处理。

Kalibr `radtan` 仅表达前四项；针孔非零 k3 或更高项无法无损导出时明确拒绝，提示使用 JSON/OpenCV 或在物理合理前提下重新进行约束标定，不静默丢参数。仅绝对值不超过 `1e-10` 的尾项允许视为零。已有 GUI 手动导出的暂存保护保留，底层标定导出函数也采用暂存替换，保护直接 API/脚本调用。拒绝非有限数值、同一相机 ID 的冲突内参和 OpenCV 键名清洗碰撞。

### 19.7 验证与持续迭代

```bash
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m ruff check src scripts
QT_QPA_PLATFORM=cocoa .venv/bin/python -m pytest -q -rs
QT_QPA_PLATFORM=cocoa MULTICAMERA_REAL_PAIR_DIR="$PWD/data/pairs" \
  MULTICAMERA_UI_QA_DIR=/tmp/multicamera-calibration-ui \
  .venv/bin/python -m pytest -q tests/test_real_pairs_ui.py tests/test_real_fisheye_ui.py
.venv/bin/python -m multicamera --self-test-calibration --report /tmp/calibration-self-test.json
.venv/bin/python scripts/stress_calibration_workflow.py \
  --source data/pairs --platform cocoa --cycles 1000 --duration-seconds 1800 \
  --dwell-seconds 30 --report /tmp/calibration-soak.json
```

`cocoa` 仅适用于 macOS，Windows 使用 `windows`；无人值守可选择 `offscreen`。持续测试仅在独立进程及临时会话中运行，源码/原图指纹、各轮模型/策略/尺寸、忙碌关闭保护、诊断原图跳转、清单校验及页面切换结果写入报告；临时会话结束后删除。源码若在测试期间变化会明确标记，不能把该快照的通过解释为所有后续修改通过。

最终可执行文件自检新增鱼眼单目、变长点友好拒绝、显式共同 ID 兼容、双目/校正及模型导出读回。与针孔自检共同决定退出码；保持不导入 Qt/Open3D，现有 Windows 构建脚本入口不变。本机源代码通过不等于最终 Windows EXE 已验收。

实际 `data/pairs` 的默认针孔训练双目 RMS 约 26.972 px；显式鱼眼共同 ID 模式保留 16 对、每帧 87 点，标准/联合训练 RMS 约 0.295/0.286 px。不同模型及观测集合不能仅以这组训练 RMS 宣称精度提升；板的真实 40 mm 尺度与独立验证数据仍需核实。

既有 Qt 原生间歇崩溃在本轮组合测试中仍复现，不能由分进程专项或单轮完整通过宣称根因解决。逐轮验证结果及未验收项追加至 Changelog；真实设备、Windows EXE、长期内存/延迟及独立精度验证不在本机结论中。

### 19.8 布局迁移崩溃与原生日志

在本轮后续循环中，进一步取得可独立复现的具体机制：按钮已在嵌套布局内时，直接向新布局 `addWidget()` 隐式迁移，可留下旧 `QWidgetItem` 包装缓存；新 QObject 的地址复用后，绑定层可能取到错误类型，出现 `SignalManager::retrieveMetaObject` 空指针崩溃。纯 PySide 最小示例不依赖本项目、pytest、标定线程或图片即可触发。

操作栏重组前现在递归定位实际原布局、显式 `removeWidget()`，再交给新布局，不依赖隐式迁移。后续紧凑/离线操作行切换原本就采用显式移除，保持该约定。新增最小几何/生命周期回归，不能通过关闭垃圾回收、跳过构造压测或禁用引导来掩盖问题。

实验对照的纯 Qt 显式移除 4,000 次、主窗口 600 次完成；正式代码的最终完整回归与循环结果见修改记录。修复覆盖已证实的布局项触发路径，不将缺少原生转储的每次历史崩溃都归为同一根因，也不代表跨平台长期稳定性已全部验收。

常规日志目录新增 `multicamera_fault.log`，通过标准库 `faulthandler` 捕获致命异常时的 Python 线程栈；启动日志同时记录 PySide6/Qt 版本。文件描述符保持存活，仅在启动且超过 5 MiB 时轮转，最多保留五份备份，不在运行中替换原生处理器持有的文件。启用失败写入普通日志，不阻止应用运行；窗口版无 stdout/stderr 仍可记录。它不能代替完整 C++ 转储，不能恢复已发生的原生崩溃。

### 19.9 后续标定优先项

- P1：建立与训练数据分离的验证图像集，验证重投影、校正、实测基线及目标距离偏差，核对实物板尺度；不按最低训练 RMS 自动决定镜头模型。
- P1：重新构建并验收 Windows 最终 EXE，执行扩展自检和反复导入/标定/关闭测试；继续跟踪原生崩溃日志。
- P2：在只读诊断之上增加可撤销的数据排除/恢复、方案对比和空间残差热图，持久化方案及选用原因，不删除原图。
- P2：补充输入文件清单、指纹及左右交换映射的持久化，进一步完善源数据追溯；本轮输出清单哈希不等同输入照片哈希归档。
- P2：明确“加载会话”的恢复语义。目前切换保存会话并不自动恢复全部内存采图/标定状态，后续应与离线导入区分并设计完整恢复流程。
- P2：大数据集的按需原图加载与缩略图内存上限；现有后台筛选不等于全部 UI/内存开销已消除。

## 20. 第一版自适应双目采图引导（2026-10-26）

### 20.1 使用路径与边界

双目采集页默认选择“自适应引导”，也可切换“固定采图计划”。已有上一张/下一张/跳过操作切回固定计划，不把浏览操作记作自适应姿态验收。多组同时采集暂用固定计划；单组自适应同时分析该组左右图像，不以单个相机的覆盖代替双目观测。离线导入/回看不运行实时到位验收。RGB+红外使用独立辅助标定板，不套用RGB ChArUco姿态统计；当前融合采图闭环及专用门槛见第48节。

本版是位置、尺度及姿态多样性的启发式推荐，不是信息矩阵优化、最优实验设计或标定完成判定；不自动选择镜头模型，不改变最终标定、原图保存、筛选或导出入口。推荐未使用实时深度或点云，不增加依赖。

### 20.2 数据、评分与临时模型

`calibration/adaptive_guidance.py` 接收左右相机的检测快照，按源帧 ID 配对；不配对、无效、分辨率混杂、冲突 ID、过度局部及双路重复观测不累计新证据。局部观测须覆盖至少 50% 的板物点凸包，合理时以平面单应估计板中心；鱼眼/大畸变下单应不可靠时，只有完整观测才退回角点均值。尺度为实际观测角点凸包面积/图像面积，不是物理距离或完整板面积。

候选包含 3×3 区域、正面/四方向倾斜、小/中/大占比。评分按两路位置次数（权重 3）、尺度次数（1）、姿态次数（0.7）及联合重复次数（0.7）惩罚采样集中；冷启动轮换倾斜建议，但明确其未验证。候选占比取 2.5%/7%/16%，估算矩形需满足 4% 图像边距、最小六像素角点间距；倾斜估算采用 25° 和 1.08 倍投影留量。候选几何约束仅是图像边界近似，并非未知双目外参下的真实共视保证；实际采集仍须通过双路可见检查。重复检测比较相同 ID，至少六点且不少于较多一路点数的 60%，二维点距离 RMS 小于图像长边的 0.5% 视为近似重复。

已有模型/尺寸兼容的内参可供引导使用。没有时，至少六个有效非重复视角、单应透视变化非退化后，在后台拟合临时内参，均匀抽取最多 30 帧；同时检查焦距/主点/畸变/有限值、RMS（不超过 2.5 px）、正深度及双方向姿态变化。失败不阻止手动补拍，显示待内参。临时内参只存在引导结果中，绝不写入最终内参、rig、会话标定文件或导出文件。该模型通过经验检查不等于真实精度已验收。

### 20.3 线程与状态一致性

`ui/threads/guidance_worker.py` 使用 Qt 线程池执行纯快照分析；同一工作台只保留一个在途任务。只有数据、配置、会话或可用内参变化时重新计算，不随预览每帧重标定。上下文键包含板对象、会话、相机组、镜头模型、观测修订、配对索引、内参矩阵及失败保存记录；旧结果返回后再次核对，不覆盖新数据的推荐。每次成功结果更新 QPainter 示意和实时目标框，当前目标不随预览抖动跳变。

保存队列未清空时不分析新增数据、不自动采图；保存失败的源帧不计入自适应证据。撤销依据剩余有效观测重新计算，不沿用已撤销样本的覆盖记录。分析异常显示回退固定计划并记录日志；自适应模式异常期间自动采图保持暂停，可手动切回固定计划。计算线程不操作控件、不修改求解器或结果；窗口销毁后 Qt 自动断开结果接收，线程池完成仅持有快照的在途计算。

### 20.4 到位检查与界面

QPainter 线稿及预览框显示推荐区域、定性倾斜和相对大小；它们不是标定后的真实 3D 投影。位置容差为归一化中心距离 0.16，尺度按角点凸包面积分为小（1.2%–4%）、中（4%–12%）、大（12%–45%），均为首版可调经验阈值。没有可靠内参时倾角不验收，冷启动需手动采集基础样本，不能一边显示“待内参”一边自动认定正面到位。

有可靠内参后通过 PnP 估计倾角；鱼眼先归一化去畸变，以原始畸变像素检查重投影 RMS（不超过 3 px）和正深度。正面两方向绝对角度均小于 15°；侧倾主方向 15°–40°、次方向小于 20°。右边近为正 yaw、下边近为正 pitch。自动采集还需两路清晰、稳定、匹配尺寸、软件时间配对可用，且构图不是已有双路重复。经验姿态到位不是标定精度或硬件同步保证。

界面显示推荐动作、推荐原因、当前验收状态和有效非重复样本统计；原固定计划仍可独立导航和回退，不将采图数量或计划浏览完成当作标定合格。

### 20.5 验证及后续

```bash
.venv/bin/python -m pytest -q tests/test_adaptive_guidance.py tests/test_adaptive_capture_ui.py
.venv/bin/python -m pytest -q tests/test_stereo_capture_ui.py tests/test_multimodal_preview.py
.venv/bin/python -m compileall -q src scripts
```

合成测试覆盖针孔/鱼眼位姿符号与临时内参、重复/无效观测；界面测试覆盖后台结果、无最终内参污染、撤销、保存失败、旧结果、无内参禁止自动验收、分析异常回退及三种窗口尺寸。完整验证结果追加至修改记录。后续使用独立验证集比较相同采图数量下的校正残差和参数稳定性，再引入信息增益排序；不能以训练 RMS 或本轮 UI 功能通过宣称精度提高。

## 21. RGB 左目与红外联合标定计算修正（2026-10-26）

独立辅助标定板保持 7×5 格、6×4 内角点、80 mm 间距。辅助内参及配对外参输入现在校验有限值、物点几何、点数、分辨率与内参模型，持有数组副本，拒绝坏观测；外参仍为 AUX 到 RGB_L 的 `T_rgb_left_aux`。

鱼眼不能将四项等距畸变直接交给普通 `solvePnP`：现在先用 `fisheye.undistortPoints` 生成归一化点再求位姿，并在原畸变图像上计算误差。针孔/鱼眼任意组合通过各自投影模型处理。辅助内参逐帧 RMS 及跨模态帧误差统一为二维点距离口径；两路帧误差按平方合并，不再取两个 RMS 的算术平均，辅助内参新结果标记 `euclidean_rms_px`。旧结果不重写，数值可能较旧坐标分量口径增大，不能直接认为精度变差。

本轮仍采用逐帧板位姿、重投影误差筛除及旋转/平移平均，未引入联合 BA；报告误差不是最终平均外参的联合重投影误差。对称棋盘格在 RGB/IR 中可能出现角点顺序歧义，低重投影误差不能证明物理点对应正确，仍需保证同一物理角点顺序；未采用盲目倒序或自动翻转掩盖问题。实际热红外设备、对比度、同步和独立外参精度尚需实拍验收。

## 22. Windows 配置布局与路径回归（2026-10-26）

`v0.1.0-build.20261026.1` 对应提交 `2898b76` 的 Windows 构建在源代码回归阶段失败（792 通过、2 失败、3 跳过），具体为配置页小窗口横向溢出 58 px 和结果目录路径的字符串分隔符差异。此时打包、最终 EXE 标定自检、安装包及 Release 发布步骤均未执行；这是源代码跨平台回归失败，不是此次已打包 OpenCV 缺失接口，也不是 Actions 存储额度错误。

相机配置的发现结果选择与添加/配对操作分为两行，避免按钮和多个下拉框的最小宽度相加后超出小窗口。标定板页监听滚动区视口尺寸：可用宽度扣除内容边距，若不足两组实际 `minimumSizeHint` 宽度、分栏间距及 24 px 预留之和，切为纵向排列；视口恢复充足宽度后再并排。使用视口而非已溢出的内容宽度决定方向，保留竖向滚动、原有输入控件、参数、保存/取消按钮和最小窗口大小。不强行关闭横向滚动条掩盖不可见控件。

Qt 的 `QUrl.toLocalFile()` 在 Windows 上可返回正斜杠路径，与 `Path` 显示的反斜杠路径指向同一目录。目录打开测试比较转换后的路径对象列表，仍验证只有一次打开且目标目录正确；复制路径测试维持实际本地字符串语义。不为测试修改生产目录或 URL 打开方式。

验证保留两个窗口尺寸的全部配置页及工作台流程，并增加强制窄视口纵向/宽视口并排恢复、较大字体下三页无横向滚动测试。运行：

```bash
.venv/bin/python -m pytest -q tests/test_configuration_ui.py tests/test_workbench_ui.py
.venv/bin/python -m pytest -q tests
```

本机专项通过不等于 Windows 门禁或最终 EXE 通过。修复提交后使用新的构建标签重新运行原 Windows 工作流，继续保留完整源代码回归及最终 EXE 标定自检，不通过重跑相同错误代码或跳过失败测试发布安装包。失败构建：[Actions #13](https://github.com/q1234ew/MultiCameraStereoCalib/actions/runs/40281324250)。

## 23. 白平衡锁定变色确认与兼容模式（2026-10-26）

### 23.1 已确认与未确认

现场对同一静止白纸场景记录两路 HTTP 原始 JPEG。初始自动模式下，白纸 ROI 的左路 R/G、B/G 约 0.9340/1.0545，右路约 0.9206/1.0630。仅左路关闭自动后为 0.6007/0.7829，右路约 0.9212/1.0632；左路恢复自动约 7 秒后为 0.8022/0.9530，右路仍近基线。公开控制数值均回到初始值，故“回读自动=1”不能作为颜色恢复证据。

经授权仅重启左路取流进程，保持原设备、分辨率、MJPEG、帧率和启动参数；右路及 IR 进程未变。重启后白纸 ROI 左路约 0.7997/0.9486，仍未恢复基线。实际启动命令和设备脚本缺少本地新版的显式自动白平衡设置，是部署差异，但本次不能将其断言为唯一根因。未执行 USB/模组复位、断电重启、固件升级或总服务重启。

随后第二组左路重启仅新增 `--white-balance auto`，设备日志确认实际应用自动=1；在早期采样约 42 秒后复查，白纸 ROI 左路约 0.7984/0.9544，右路约 0.9222/1.0632，左路仍未回到基线。该临时左路进程保留显式自动参数运行，右路/IR 不变、SSH 已退出、持久启动脚本未改；不将模式自动或单独重启成功解释为颜色已恢复，也不把“补启动参数”当作已验证的根因修复。

原始 JPEG 已变色，排除“仅 Windows Qt 显示偏色”作为本次现象的充分解释；服务日志显示 MJPEG 硬件编码输入透传。当前标准 V4L2 控制未列出专门的 3A 白平衡锁定或实时红蓝增益；供应商扩展能力仍待确认。本地服务 API 将自动开关直接映射为 `V4L2_CID_AUTO_WHITE_BALANCE`，不是厂商冻结增益接口。精确责任仍需驱动、模组 UVC 固件或 ISP 的进一步取证，不能以此报告宣称具体固件缺陷已定位/修复。

ROI 为人工确认在白纸内的 `(480,180,640,340)` 矩形，使用每阶段最后一帧的 JPEG 通道均值。纸张非标定灰卡，两路 ROI 不是完全相同物理区域，未测照明光谱；数值仅支持同路变色与参考路对照，不代表专业色彩精度。

### 23.2 客户端策略

默认严格锁定继续执行自动预热、稳定检测、关闭自动、锁定后颜色检查；失败恢复自动并拒绝采图。新增 `WhiteBalanceLockError` 和色度差/阈值日志，提示用户可显式切换“自动稳定（不锁定）”。不自动放宽原 0.05 锁定色度变化阈值，不写入 inactive 的历史色温。

兼容模式复用预热与稳定窗口（默认 7 秒、20 秒超时、4 次采样、稳定阈值 0.025），最后回读两路仍为自动，并再次检查色度变化。不发送关闭自动，不将结果称为已锁定；每次采图重新准备，`WhiteBalanceGuard` 只放行本次回调，回调未使用的许可也清除。策略切换、断线、会话变化或取消均使许可失效。当前版本兼容准备仍会显式下发自动开启，尚未实现“会话前仅初始化一次、逐帧无写入监测”的完整改进流程。

稳定仅表示近期颜色变化较小，不证明中性、左右颜色一致或拍摄期间参数冻结。兼容模式解除不兼容锁定导致的持续采图阻塞，无法根治相机输出偏绿；真实冻结需设备支持相应机制。操作区显示策略；离线导入/标定不显示或调用实时准备。新增选择行与复选框并排以避免压缩紧凑窗口采图引导。

### 23.3 诊断与验证

```bash
# 默认只读；输出目录必须尚不存在。
python scripts/diagnose_white_balance.py \
  --left-url http://CAMERA_HOST:8080/ --right-url http://CAMERA_HOST:8082/ \
  --output /tmp/awb-readonly

# 仅在明确授权、两路静止且左路已经自动时进行短时切换。
python scripts/diagnose_white_balance.py \
  --left-url http://CAMERA_HOST:8080/ --right-url http://CAMERA_HOST:8082/ \
  --output /tmp/awb-controlled --change-left-awb

python -m pytest -q tests/test_white_balance.py tests/test_white_balance_diagnostic.py \
  tests/test_scrollbar_theme.py tests/test_workbench_ui.py tests/test_offline_ui_state.py
```

脚本不重启服务，默认只读；控制实验仅写左路开关，不写右路/曝光/色温，并在 `finally` 尝试恢复自动及回读。报告中的 `restoration.verified` 只指模式回读，不表示颜色恢复。写报告失败不能阻止恢复命令执行；断电、网络持续失败或强制杀进程不保证恢复。现场原始证据保留在临时目录，可能被系统清理，不提交地址、原图和登录凭据。

滚动条补齐横/纵 `add-page/sub-page` 背景及横向按钮尺寸，避免 Windows 主题斜纹/亮块；必要滚动继续保留，不以关闭滚动条掩盖内容裁切。验证同时检查安全门禁、恢复路径、一次性许可、模式切换、离线状态和滚动条像素均匀性，实际执行结果见修改记录。

## 24. 默认自动白平衡采图策略（2026-10-26）

根据实拍对照，客户端当前默认策略调整为“自动稳定（不锁定）”，取代上节初次加入兼容选项时保留的默认严格锁定。默认实时采图准备不再发送 `white_balance_automatic=0`。左右分别读取：已为自动=1 时跳过模式写入；只有手动路才发送自动=1，再检查两路稳定及最终模式。失败/取消时只尝试恢复实际改动过的路，全部已自动的纯检查不会因失败而重新写模式。

白平衡管理默认勾选，采集页默认选中自动策略；不会仅因打开/连接客户端就修改相机参数。参数准备仍发生在采图之前，不承诺在设备服务启动时替代其启动配置。每次采图通过稳定检查后获得一次性许可，后续重新检查；不是永久锁定，也不保证颜色准确、左右一致或拍摄期间增益固定。

手动路径在 UI 明确命名为“手动冻结验证（可能变色）”，需要用户显式选择。保留原检测/恢复阈值，不证明 `V4L2_CID_AUTO_WHITE_BALANCE=0` 等同厂商真正冻结，未经设备验证不建议使用。底层既有显式锁定计算接口保持兼容，默认 UI 路径始终传入不锁定模式。切换策略、会话或连接变化仍使许可失效；离线输入不访问设备。本轮只修改客户端，不重启或调节设备，已有残留偏绿仍需设备侧处理。

### 24.1 后续实机整机重启对照

经用户另外授权，于 2026-10-26 继续执行三轮整机软件重启，以不同内核 boot ID 确认实际重启；不是已验证的模组断电或冷启动。重启实验不再关闭自动白平衡，不修改色温、曝光或持久配置。每轮网络就绪后，缺失的 RGB 服务以原设备、分辨率和帧率加显式 `--white-balance auto` 临时补起；不能将该实验等同于原部署脚本无干预自动启动。

- 第一轮恢复了此前左路关闭自动后残留的明显偏绿；第二、三轮补起后的早期采样亦接近第一轮恢复状态。各轮初次及延后只读检查均保存原始 JPEG、控制快照和哈希，每次三对图像；具体后续复查结果保留于现场报告。
- 同一画面白纸 ROI（x=480:640、y=180:340）在重启前左路 R/G、B/G 约为 0.798/0.954，第一、二轮恢复后约为 0.933/1.061，接近更早正常基线 0.934/1.055。此指标仅比较同相机同场景，不是标准色卡校色或左右一致性验收。
- 三轮均出现开机早期网络尚未就绪的故障：mDNS 无非回环 IPv4，视频服务报 `getaddrinfo: address family for nodename not supported` 及 HTTP 绑定失败；第一轮 RGB 失败，后两轮 RGB/IR 均失败。脚本仍输出 `All streams started`，未自动恢复。现有证据指向网络就绪与启动重试缺失，不能把 `Can't bind ...: Success` 直接解释为端口占用。
- 第一轮一次延后读取还出现 API/快照超时，后续不改参数、不重启即恢复。短时色度稳定不代表长期服务可靠性已通过。

设备端后续应独立修复网络就绪等待、失败重试、启动成功判定及日志；本次未部署这些修复。三轮补起后的自动模式未复现原明显偏绿，不代表自动模式永不偏色，亦未定位到具体 UVC 固件/驱动内部责任。客户端仍应保持默认不关闭自动，已发布的旧 Windows 包不自动包含本地策略修改。

### 24.2 客户端接入分阶段实机对照

同日一轮当前 macOS 源码组件实测：先只读建立基线，再用实际 `StreamManager/MJPEGGrabber` 接入双路约 12 秒、打开 `RgbSettingsDialog` 只读加载、执行默认 `stabilize_and_lock(lock=False)` 准备。四阶段各保存三对原始 JPEG 和控制快照，共 24 张，哈希复核通过。左纸面 R/G、B/G 分别约为 0.9325/1.0607、0.9328/1.0603、0.9330/1.0607、0.9328/1.0604，未出现此前明显偏绿；右路亦接近基线，全部公开控制值前后相同。

两路实际持续收到图像，准备成功且受审计参数客户端未发出控制写入，日志确认两路已自动并跳过写模式。测试结束关闭测试取流线程，设备服务不停止。该实验为离屏 Qt 组件级实机验证，不是完整主窗口点击/会话保存验收，也不是 Windows 最终 EXE；未应用 RGB 参数或重现旧手动锁定，未抓取设备全量 HTTP/ioctl。只能说明本轮当前默认路径未复现问题，仍须核对现场安装包版本及实际控制请求，不能将“接入后变色”直接归因为网络连接本身。

## 25. 接入、采集与结果交付交互（2026-10-27）

### 25.1 采集前状态

实时双目固定操作区区分等待首帧、画面过期、未满足软件配对及配对就绪。每 500 ms 本地检查缓存，不为刷新状态请求设备控制接口。悬停显示全部所选组状态、ChArUco 规格、白平衡策略和保存目录；多组时正文压缩显示，详情保留在提示中。顶部“已连接”仍表示网络连接，不代表可采集或通过质量检测。

`StreamManager.MAX_CAPTURE_FRAME_AGE` 默认 2 秒，检查接收时间戳；同步/宽松配对及配对信号均拒绝过期、非有限或未来时间戳，断流清理该路缓存。无可用双路图像时不创建会话、不启动白平衡准备。手动采集仍允许新鲜但超过同步容差的最新帧回退，并明确记录日志；自动采集原有配对要求不放宽。该检查不是硬件同步、曝光时间测量或镜头颜色校准。

白平衡默认沿用第 24 节的自动稳定、不锁定策略。连接/首帧状态不触发模式写入，不能因“配对就绪”就宣称白平衡、角点或清晰度已通过。建议姿态、检测指标和未验证项仍分别展示。

### 25.2 原图保存与恢复

双目保存仍采用后台串行队列。主按钮显示保存中的组数，底部单独显示本次已落盘/保存中/失败数量。“本次”是当前面板会话期间已成功写入的配对，不是会话全历史图像数量，也不是检测有效配对数。检测数据可先进入内存，但只要仍有 pending/失败任务，有效配对标题明确标记待落盘，禁止开始标定及下一批手动采集。

保存成功后记录落盘状态；失败保留该任务原图、相机组、原始帧号和所属会话，暂停自动采集，显示“重试保存原图”。重试复用同一帧号与图片，不再次检测、不增加观测、不覆盖其他帧；成功清除该帧失败标记。撤销支持 pending、已保存和失败三种状态，失败任务撤销会释放重试缓存。重试是进程内恢复，不是崩溃恢复或持久失败队列。

未解决的保存失败阻止会话切换、配置修改、导入替换和直接关闭窗口，提示重试或撤销；保存目录不可用时不能靠反复采图继续积累。显式重置会丢弃对应内存重试任务，不删除历史成功文件。多模态当前仍同步保存，但先确认保存成功再推进观测、图像缓存和采图计数，失败不增加记录；白平衡准备通过后重新取得新鲜配对。

### 25.3 补拍与交付

补拍仍失效旧计算结果、允许当前数据直接重标定，不复用过期内参/外参。结果页明确显示数据已变化、旧结果已过期及保留的历史文件目录；历史文件不删除，但不允许冒充当前结果导出。新结果生成后清除过期状态，显示完整可复制保存路径，区分计算成功与文件全部/部分保存成功。结果详情追加已记录的左右内参、双目参与/输入帧数和未参与帧 ID，缺少来源时显示未知，不虚构剔除原因。

### 25.4 版本识别与验证

状态栏及启动日志显示版本、提交与源码/安装包类型。源码首次读取 Git 状态并缓存，存在修改时明确标记。PyInstaller spec 在构建时生成内嵌 `build_identity.json`，记录提交、精确标签与 dirty 状态；旧包无元数据时显示未知，不从用户电脑上的仓库猜测安装包版本。此信息用于追踪来源，不替代最终 EXE 哈希或 Windows 运行验收。

专项：`python -m pytest -q tests/test_capture_readiness_and_saving.py tests/test_build_identity.py tests/test_stereo_capture_ui.py tests/test_calibration_workflow.py`。覆盖缓存新鲜度、断流清理、无帧不准备参数、后台真实保存失败、同帧重试、撤销、未落盘禁止标定、融合保存失败不推进、过期结果和版本缺失降级。布局检查沿用原生 Qt 截图，覆盖 1000×680、1280×800、1440×1024；完整回归及静态检查结果见修改记录。本轮不改变标定算法、不部署设备服务、不自动提交或打包；设备开机网络就绪问题仍单独处理，Windows 最终 EXE 仍须后续验收。

## 26. 离线采图引导回放（2026-10-27）

`calibration/guidance_replay.py` 对已采集双目图像按前缀回放，复用现有 `analyze_guidance()`，不更改实时推荐算法或 GUI 入口。第 N 步先以此前 N-1 帧生成建议，再用当前帧检测和历史内参核对位置、角点凸包占比与倾角；最后加入当前观测生成下一建议。每个前缀独立冷启动，不读取最终标定文件，也不把结束后的内参用于先前步骤。临时内参守卫不通过时保留“倾角未验证”。

运行（输出目录必须不存在）：

```bash
python scripts/replay_capture_guidance.py --input data/pairs --output /tmp/guidance-replay
python -m pytest -q tests/test_guidance_replay.py tests/test_adaptive_guidance.py tests/test_adaptive_capture_ui.py
```

默认使用项目素材对应的 ChArUco 15×10 格、40/30 mm、DICT_4X4_1000，其他板通过 `--board-config` 传入 `CharucoBoard.to_dict()` 格式 JSON；镜头模型可选 `--model pinhole/fisheye`（参数实际为其中一个值）。图片配对与自然排序沿用现有导入规则，编号不证明实际拍摄时间或硬件同步。未额外按清晰度筛图。

输出 `report.json`、`REPORT.md` 及 `index.html`，记录每步前后建议、历史引导内参可用性、当前测量/无法测量状态和去重样本数量变化。HTML 提供逐帧选择、前后切换和播放控件，直接引用原始图片路径，不改图、不写会话、不覆盖已有标定结果；跨电脑移动需保持原图路径可访问。源文件在分析前后核对 SHA-256，报告目录已存在则拒绝覆盖。源码测试验证前缀隔离和文件保护；未通过浏览器渲染时不能将 HTML 静态语法检查描述为交互验收。

覆盖统计为双路几何有效输入帧的3×3中心分布与大小分类，包含重复构图，不是精度、信息量或去重观测计数。静态图像不评价运动稳定、同步或白平衡；历史图没有按当前建议拍摄，不匹配不等于原图无效或算法错误率。

本轮 `data/pairs` 16对均产生几何有效观测，两路中心主要集中中央，下方三格无中心观测；两路均11对大占比、5对中占比。加入0006后曾获得两路临时内参；加入0012后仅右路保留，加入0013后均未通过，最终倾角未验证。新增观测后的重估可能退化，不能把早期内参可用当作后续永久有效，也不能直接将新增帧判为坏帧。结束后全量补充诊断的针孔内参训练 RMS 约5.233/11.615 px，高于引导守卫2.5 px，此诊断不回灌历史。与此前清晰度筛选后的11对不是同一输入，不能直接比较算法优劣。完整素材结论与原图哈希保留在临时报告，不提交原图。

## 27. 图像观测驱动的模型评估（2026-10-27）

### 27.1 使用流程与边界

在标定页选择相机组，点击“自动评估模型”。后台比较现有针孔与鱼眼模型，显示留出预测误差和推荐理由；没有可靠优势时不推荐切换。点击“应用推荐模型”后仍需确认：当前相机模型是全局配置，影响所有相机组与辅助标定，已有计算结果失效，但原始检测观测、源图和历史结果文件保留，可直接重新标定，无需重导入。鱼眼推荐同时选择共同角点兼容策略；针孔推荐恢复全部使用策略。

仅支持现有双目板点观测，不根据照片外观猜测镜头，也不自动识别板尺寸、物理焦距或同步关系。本轮不新增其他投影模型或RGB/IR自动选型，不涉及点云。留出比较属于同一序列内部交叉验证；相邻帧可能相关，结果不能替代独立验证集、实测基线或三维测量验收。

### 27.2 计算与安全约束

`calibration/model_selection.py` 要求至少9对左右及双目共同有效观测，最多按顺序均匀取24对。帧号重复、图像尺寸不一致、几何观测不足或两个倾角方向变化不足时拒绝评估。按索引模3固定分折，分别使用其他两折拟合内参和双目外参。两模型使用相同原始帧和留出角点；各训练折显式使用跨视图共同ID求双目外参。

留出帧只用左图求板姿，经训练外参预测右图原始像素，比较欧氏误差的汇总RMS，不使用右图反向优化该帧姿态。全部三折须成功，拒绝非有限值、负焦距或相机背后板点，不从失败折挑选好点。最佳候选留出RMS须不高于2px；两个候选均有效时，还须至少降低20%且绝对降低0.15px。只有一个完成全部检查时，提示另一个候选失败，仍要求人工确认。

`ModelSelectionWorker` 在独立Qt线程计算，使用不含原图的观测快照；结果和进度由信号回主线程。取消在计算阶段边界检查，不强行终止OpenCV调用。评估期间阻止数据修改、标定、配置与会话切换；数据版本或目标组变化使旧推荐失效。应用前先保存会话配置，写入失败不切换；最终结果记录对应的 `calibration_info.model_assessment`，包含输入/抽样帧、各折训练/留出ID、分数和失败原因。

### 27.3 展示口径与验证

校正预览仅代表对应已完成结果。切换模型、数据筛选、优化策略、重置、补拍失效和预览生成失败时清空并隐藏旧图，避免“待标定”状态显示历史校正结果。预览标题标明相机组、针孔/鱼眼模型、实时配对或该组最后导入配对；当前不随原图缩略图选择联动，不将无组别的历史图片作为回退来源。预览原图尺寸必须与内参标定尺寸一致，否则拒绝显示并记录原因。图片不作为侧栏横向最小尺寸，缩略图按侧栏可用宽度及160px高度上限等比缩放，避免出现图片后撑宽侧栏。参考线只是辅助目视检查，不是自动匹配正确的证据；实际残差仍以同ID角点诊断为准。这些显示修复不改变已保存的内外参或校正矩阵。

离线底部保存状态按字体高度及可用宽度布局，不再被压成窄条。预览标记“预览角点”，侧栏分列“当前预览角点（原图分辨率检测）”与“数据集累计：板角点ID/视场网格”；预览与点击采图不一定是同一帧，累计100%不代表当前帧完整、姿态充分或标定合格。部分格式导出失败时，结果摘要先显示完整绝对目录，打开和复制目录仍可用，再列失败详情。

验证覆盖不足数据、倾角退化、留出隔离、只读计算、取消、过期推荐、确认拒绝、会话写入失败、窄窗口状态标签、预览/累计口径及部分导出失败路径。完整测试844项通过、4项跳过，真实素材专项显式执行：16对 `data/pairs` 推荐鱼眼，确认后保留16对并完成标定，训练双目RMS约0.295px。真实素材测试通过环境变量 `MULTICAMERA_REAL_PAIR_DIR` 指定图片目录后运行 `tests/test_model_selection.py`，默认跳过该项。实际macOS窗口也完成导入、评估、确认、标定和结果查看，5个文件成功落盘，JSON含评估报告；部分导出失败路径由回归测试验证。该结果不代表实测三维精度或Windows最终EXE验收。

## 28. 结果批次隔离与失效状态一致性（2026-10-27）

### 28.1 批次交付与兼容

自动保存由固定同名文件改为第3节所列的独立批次目录。每次求解完成都产生新批次，包括同一数据的重标定；不将上一次失败格式的旧YAML留在本次目录。原文件、旧根目录及历史批次只保留，不迁移、不删除。第三方程序应使用结果页指向的本批次文件，或解析 `latest.json` 后读取相应批次，不能继续硬编码根目录 `results/rig.json`。旧会话没有索引时仍可加载。手工导出使用用户选择的路径，不受自动批次目录影响。

索引在本批次文件保存结束后原子发布，JSON本体失败不推进索引；单格式或清单失败会返回已保存文件和明确错误。索引失败不删除本批次，也不假装加载器已切换到新批次；`load_rig()`仍按旧索引读取。目录名称使用时间戳与UUID，避免同秒重标定冲突。此改动不改变标定模型、输出矩阵或单位，也不涉及点云。

### 28.2 模型评估与历史结果

模型评估仍绑定相机组、板对象、观测对象及版本、帧ID和会话。唯一可迁移场景为：无会话的首次离线标定完成后，主窗口自动创建保存会话，并显式调用 `set_session(..., preserve_model_assessment=True)`。只有原推荐在迁移前仍有效才重绑定会话标识，不重新计算或更改分数。普通新建/切换会话、已有会话之间切换、观测修改均不获得此例外。因而同一数据连续重标定能持续附带评估报告，真正更换数据仍阻止旧推荐应用。

`_mark_results_outdated()`统一保留过期状态及历史目录，供模型应用、策略/筛选变更和补拍失效使用。失效后禁止当前结果导出，结果页显示旧结果过期和原目录；多次改策略不遗忘首次目录。新计算发布后清除过期标识，显示新批次。重置则显式清理内存状态，但不删除磁盘文件。

### 28.3 验证范围

回归覆盖重复标定的旧文件字节不变、部分导出失败无旧格式混入、断开的多组相机导出、批次清单哈希、旧版会话加载、JSON失败和索引发布失败不推进最新结果、非法索引拒绝历史回退。UI回归覆盖首次自动建会话、连续重标定报告保留、真正换会话/改观测推荐失效，以及筛选/优化变更后的历史路径和重新发布状态。真实素材测试显式开启后验证导入、鱼眼推荐、两轮标定及批次隔离。Windows最终EXE与实际设备流不属于本轮验收；具体测试结果见修改记录。

## 29. 工具栏与完整信息展示（2026-10-27）

主工具栏固定停靠，分为连接控制（连接、断开、连接状态）、相机工具（相机配置、RGB参数）与会话操作。窗口宽度低于1200px时，配置和RGB参数仅显示Lucide图标，并保留完整悬停提示和可访问名称；连接主操作保留文字，断开为图标按钮。动作仍复用原QAction和信号，不改变连接流程或权限。会话主按钮行为不变，右侧菜单复用原新建/加载动作，长会话名随宽度省略，提示包含完整名称与目录。

侧栏换行标签不以最长单词作为最小宽度；结果摘要、路径/错误、模型建议和相机组状态根据当前宽度更新所需最小高度。仅允许垂直内容增长，长内容可在侧栏滚动查看，不用裁切来隐藏溢出。原始路径和文本保持不变，支持完整选择复制。标定状态分行展示内参RMS、双目RMS和基线，避免基线数值被挤到行尾。此布局调整不改变覆盖率、误差或质量判定口径。

状态栏短消息直接阅读；过长消息可悬停查看全文，`currentMessage()`和日志仍保留原值。版本标识最多210px，省略展示但提示保留完整身份，不将省略后的文字作为构建来源记录。此机制不替代结果页的可复制完整目录。

回归检查1000×680、1280×800、1440×1024窗口中的工具可见性、菜单动作复用、图标模式与长会话名提示；同时用180字符无空格路径及长错误说明验证侧栏无横向溢出，并以 `heightForWidth()` 核对实际分配高度，而不只检查标签是否可见。平台限制：本机Qt回归不替代Windows最终EXE及高DPI原生字体验收。


## 30. 标定板主视角动作示意（2026-10-27）

`StereoSchematic`默认使用 `board_view.py` 的板面视角，仍由QPainter绘制，无3D引擎或新增依赖。坐标遵循相机X向右、Y向下、Z向前，透视投影为 `(X/Z, Y/Z)`，与未镜像左目预览同向；板面上缘增加“上”标记。主画面是放大的板姿示意，不是实时相机画面或真实视场。每个目标按起止运动范围计算固定构图，播放过程中保持同一投影和放大比例，防止自动构图形成额外运动暗示；不同目标可有不同示意放大比例。格子仅表示平面方向，不读取或替代实际ChArUco/红外板规格。

选择“板面＋俯/侧”时，右侧辅助图依据倾斜轴选择：左右倾斜为俯视，相机固定在下方，靠近镜头表现为向下；上下倾斜为侧视，相机固定在左侧，靠近镜头表现为向左。辅助投影复用相同 `board_point()` 和固定边旋转几何，不独立编造第二套动作。白色固定边及圆点表示支点，黄色边和箭头表示移动侧，灰色轮廓表示起始，蓝色虚线表示终点。角度、距离和尺寸均为非实测示意，不参与采图质量判定。默认右侧立体辅助视角见第31节。

完整模式底部提供“板面＋立体 / 板面＋俯/侧 / 空间对照”视角选择，最后一项保留原前侧观察的一体式相机线稿；紧凑模式隐藏页脚，可通过展开恢复。视角切换只使画布静态缓存失效，不改变目标、进度、暂停状态、自动采集或观测数据。原隐藏/滚出可视区停表、重播和目标切换生命周期继续生效。18个双目目标和13个融合目标保持不变。

回归覆盖投影不镜像、四个倾角固定边不动及移动侧靠近相机、三档板面大小下所有18目标的完整运动位于构图区、两种视角缓存/非空/动态像素变化及视角切换保留状态。原生预览用于目视确认方向和标签；Windows最终EXE的高DPI显示仍须单独验收。此改动不包含相机参数、检测/标定算法或点云调整。

## 31. 右侧固定立体辅助视角（2026-10-27）

板面预览默认组合为“板面＋立体”：左侧仍按未镜像相机坐标展示放大板姿，右侧使用固定正交斜视投影 `u=0.8X+0.6Z`、`v=0.3X+sqrt(0.75)Y-0.4Z`。两投影基向量正交且为单位长度，不自动环绕观察方向。右侧使用同一 `board_point()` 运动、板纹理及边缘标记，配较小的相机模型说明朝向；相机及其双光轴静态缓存不随播放进度变化。此投影是示意，不是镜头成像模型或标定矩阵。

每个目标根据整段运动与辅助相机边界固定构图，三档板面大小及18个目标均测试不出界。相机模型仅在此小图省略角色文字，避免紧凑尺寸下L/R重叠；原空间对照仍保留。视角菜单可以选择“板面＋俯/侧”恢复俯视/侧视解释，或“空间对照”查看原先的双场景模型。菜单切换保留进度、暂停与目标，回归以像素检查板面主视图不因右侧切换而改变。

本轮不改变18/13目标计划、自适应推荐、真实相机参数、观测或标定结果；显示模式仍由 `StereoSchematic` 与QPainter实现。最终验证结果见修改记录。

## 版本来源与发布一致性

产品版本唯一来源为 `src/multicamera/__init__.py::__version__`，保持数字 `MAJOR.MINOR.PATCH`。`pyproject.toml` 使用 setuptools dynamic attr 生成发行包元数据；升级版本后需重新安装可编辑包。Windows 批处理与 Actions 调用 `scripts/check_release.py --installed --print-version`，将结果通过 `/DMyAppVersion` 传给 Inno Setup；未传版本时安装器编译直接失败。PyInstaller 同样校验元数据，并将版本、提交、标签和工作区状态写入内嵌构建身份。

`python scripts/check_release.py --installed --require-clean --tag <标签>` 检查单一版本来源、已安装元数据、工作区及标签指向 HEAD。正式标签为 `vX.Y.Z`，预发布沿用 `vX.Y.Z-build.YYYYMMDD.N`，检查有效日期和正整数序号。日期使用上海时区。普通本地构建允许未提交修改；CI 发布要求干净工作区，且最终 EXE 自检报告的版本、提交及标签与当前构建一致。

源码与最终 EXE 的标定自检报告增加 `build_identity` 字段；原 OpenCV 比对仍保留。CI 仅收集当前产品版本的安装包，Release 记录产品版本、标签、完整提交与构建链接。已存在的 Release（含草稿）拒绝重新上传，失败后需检查原发布状态并使用新标签，禁止覆盖历史二进制。此机制不自动升版、创建或推送标签。


### 本地发布准备命令

`scripts/prepare_release.py` 复用 `check_release.py` 的单一版本来源与工作区检查。默认增加 PATCH；`--bump minor` 增加 MINOR 并归零 PATCH；`--preview` 默认不升产品版本，支持组合 `--bump patch/minor`；`--bump none` 可将已验证的当前预发布目标准备为正式版。命令只修改 `__version__` 字符串，保留其余源码和换行格式；不会提交、建标签、访问远程或触发构建。

命令返回 JSON 计划，包含原版本、目标版本、建议标签、准备前提交、工作区状态和后续步骤。`--dry-run` 不创建锁、准备记录或文件，允许在有未提交修改时预览。正式准备必须有已提交 HEAD 且工作区干净，并在写入前再次检查。

预发布使用上海日期，同产品版本和日期下取本地标签及历史预留的最大序号加一。Git 公共元数据目录的 `multicamera-release.json` 保存预留列表和最近计划，同仓库的多个工作树共享编号记录及锁，`multicamera-release.lock` 防止并发写入。写入采用临时文件替换；先预留再改版本，改版本失败可能留下序号空缺，以防重复。准备记录损坏、锁已存在、正式标签已存在或正式计划已预留时直接失败，不覆盖旧记录。

编号仅限本地可见信息，不是远程预留；跨机器准备前应在授权范围内同步标签。版本更改后重新安装可编辑包，完成测试与文档更新后提交，再创建建议标签并运行严格发布检查。正式推送和 Windows 构建仍由后续授权操作完成。

## 32. 自适应采图实际流程验收（2026-10-27）

### 32.1 动态推荐与实时测量口径

推荐仍依据已采观测的位置、板点画面占比、倾角分布及重复构图进行启发式补缺，不是信息增益最优化，也不以引导完成数保证标定精度。实时预览与历史分析统一调用 `measure_composition()`：通过板点几何验证及单应性估计整板中心，避免仅检测到一侧角点时用角点均值造成位置偏差。板点范围不足、越界或几何不可靠时，不作为自动采集有效证据。没有可行候选构图时明确阻止自动采集。

固定计划与自适应模式共用严格的 `measure_pose()`。只有相机模型、分辨率、内参质量、正深度及重投影误差检查通过，才显示或验收倾角；鱼眼使用对应的归一化投影路径。界面距离标为“板中心深度”，不是板原点距离或欧氏距离。没有可靠内参时不自动确认自适应倾角，可手动采集基础样本；固定计划仍仅允许正面目标自动采集。

### 32.2 稳定窗口与原图复核

每路至少3次有效检测、跨越至少0.6秒才判为稳定。位移相对本窗口起点检查，公共角点至少6个且覆盖两次检测较多角点数的60%，最大移动小于图像较长边的0.8%；不采用只比较相邻帧的方式，以防缓慢累计移动被放行。检测时间及最新帧接收时间必须有限、不在未来且不超过1.5秒。板、会话、模型、相机源或回看状态变化后清空稳定证据。清晰度以板点区域缩小至最长边不超过320px后的Laplacian方差判定，默认阈值50；这属于图像质量启发式，不是运动模糊的物理测量。

自动采集在预览验收之后，对实际取到的原图再检测，每路每次只检测一次；复核可见板点、清晰度、分辨率、与稳定预览的位置差、目标姿态及重复构图。所有组先验收，再将通过的组入库和排队保存，避免第一组的待保存状态拦住同批后续组。未通过的组不新增观测、不分配保存帧号、不保存文件或推进采集计数，并显示原因。不同组允许部分通过，不表示跨组硬件同步。

自动路径只使用严格软件时间配对，取帧期间配对失效时跳过，不回退宽松最新帧。白平衡准备异步结束后重新走自动就绪检查。手动采集仍保留原来的宽松取帧与重复构图提示，不能将手动路径等同于自动验收。

### 32.3 保存、重试和撤销

原图待保存或保存失败时暂停后续自动采集；临时观测不代表原图已落盘。失败样本不推进自适应推荐的有效样本数。重试使用缓存原图，不重复检测、不追加观测；成功后再更新会话metadata与有效样本分析。撤销按批次移除对应观测及已保存文件，重算推荐；保存尚未完成时继续使用原有取消标记处理。自动定时器本身不作为后台任务阻止自身采集。

### 32.4 RGB＋IR衔接

辅助离线替换导入清空旧引导进度，不将检测成功视为已确认某一姿态；导入后补拍使用不冲突帧ID。状态区区分“离线原图位于源目录”与“实时采集已保存”，不承诺已把导入文件复制进会话。RGB左目内参是融合标定前置条件，不是采图前置条件；配对仍为软件时间配对。辅助数据变化只使辅助结果失效，保留RGB结果及历史目录；旧文件不删除，重新导出以当前有效对象为准。本轮不新增辅助撤销入口，不改变 `T_rgb_left_aux` 方向及7列×5行格子、80mm、6×4内角点配置。

### 32.5 验证方式与边界

专项测试位于 `tests/test_adaptive_workflow_followup.py` 与 `tests/test_aux_workflow_followup.py`，覆盖非法时间、部分板中心、累计移动、会话/源切换、原图拒收、取帧竞态、成功保存/撤销、失败重试/推荐重算及多组批次。Qt界面测试在1000×680、1280×800、1440×1024运行真实控件，图像和流来自受控输入，不访问设备。

实际素材可用 `.venv/bin/python scripts/replay_capture_guidance.py --input data/pairs --output <新目录> --model fisheye` 逐帧回放。每一步只使用此前观测，报告记录原图哈希并在结束复查；不修改源图片、会话或标定结果。回放没有实时运动、网络抖动及设备曝光过程，不能代替真机自动采集、Windows最终EXE、高DPI及硬件精度验收。本轮不涉及点云优化。

### 32.6 Windows命令行输出编码

Windows构建第15次运行在回归阶段因回放脚本的中文进度输出失败：重定向标准输出继承 `cp1252`，触发 `UnicodeEncodeError`；该次未进入EXE打包及自检。`replay_capture_guidance.py` 命令行入口现在显式将可重新配置的标准输出与标准错误设为UTF-8，保留中文进度、报告路径和异常。库函数导入不改变宿主进程的流配置；HTML、JSON及Markdown文件继续显式使用UTF-8。

Windows Actions作业同时设置 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8`，统一其Python子进程默认文本及标准流编码。通过管道调用回放CLI的程序应按UTF-8解码输出，不能依赖Windows区域代码页。回归主动关闭UTF-8模式并分别指定UTF-8、cp1252和ASCII标准流，验证中文源目录、输出目录、成功进度、错误输出、文件哈希和禁止覆盖行为；不只依赖CI环境变量绕过问题。此修复不改变图像颜色、白平衡、标定算法及结果格式。

构建#16已通过中文输出及报告生成，但在重复运行的错误路径断言失败：`FileExistsError` 在异常信息中使用文件名的 `repr` 表示，Windows反斜杠和路径内引号会转义。回归检查据此匹配 `repr(str(path))`，不将整个stderr做反斜杠替换或Unicode反转义；成功输出中的报告路径仍按原文检查。跨平台用例在任意宿主上启动子进程，分别对POSIX、Windows盘符及UNC路径主动产生真实 `FileExistsError`，包含中文、空格、引号和易与转义序列混淆的目录名。原有异常类型、无编码错误、源文件哈希及报告禁止覆盖检查全部保留。

## 33. 现场自适应引导反馈修正（2026-10-28）

### 33.1 复核证据与边界

以 `data/test2` 提供的会话、日志及4张截图复核。会话压缩包保留20对图片，metadata记录31次保存编号，缺失编号不补造观测。沿原针孔模型做前缀回放：旧推荐在20次后续建议中有16次为上边近倾斜，区域行号发生12次切换。部分现场截图虽然位置接近目标，但板点占比仍未达到小尺寸目标；多帧求解倾角也接近正面。不能据此认定所有实时“重复”提示都是误报：压缩包不包含全部预览帧、删除图片或设备运动过程。

旧评分优先选择最缺少的倾角，却只排除上一条完整目标；同一倾角换一个位置即再次成为最优候选，造成方向锁住和区域循环。实时近似检测使用图像长边1%的阈值，而后台去重为0.5%，小板即使姿态有变化也可能落在提醒阈值内；两眼还可能分别命中不同历史帧，并被错误组合成“重复配对”。

### 33.2 同高度成组调度

前6个去重样本的建议优先安排中央位置，改变倾角和大小；随后按中央行左右、上排右中左、下排左中右安排，每个周边区域先给出2次建议，再切换区域。该预算只控制操作顺序，不代表6组必能取得内参，也不把“已经建议过”记为真实姿态完成。

每次推荐仍使用两路实际位置、大小和可解倾角统计；另对重复建议方向和同区域重复方向施加调度惩罚，避免始终推荐同一个未达成方向。第一轮分组结束后继续按欠缺区域和高度移动成本补缺，不输出“已完成标定”。每次都过滤板面边距和角点间距不可行的候选。通过对去重历史重放推导调度状态，建议次数是前缀重放推导值，不是对既往屏幕展示日志的读取；重复输入不推进，撤销、失败保存排除和会话切换继续走原有重算机制。无可靠内参时不自动确认倾角。

位置已经接近时，标题使用“保持在”而非继续“移至”；实时反馈显示两路当前板点占比及目标区间，倾角不匹配时显示各路当前左右/上下倾角。采集时间、稳定性、清晰度、PnP重投影等原验收门槛不降低。文件日志在分析结果真正更新时记录相机组、去重样本数、推荐行列/倾斜/尺寸、内参可用性和候选可行性，不按预览刷新频率刷日志。

### 33.3 统一近似检测

`observations.similar_composition()` 统一实时提醒、原图验收、单目提示和推荐去重的单路几何判定。公共角点至少6个且占较多一方角点数60%，尺寸相同、坐标有限；共同角点RMS位移必须同时小于图像长边0.5%及两帧较小角点包围盒对角线0.6%。后一约束随板面像素尺寸变化，避免小板沿用过宽的整图阈值；依然允许亚像素检测噪声。此判定不是基于PnP信息增益的相似度，也不将微小像素移动视为高质量新姿态。

实时与自动原图重复判定要求左右同时命中同一个有效历史配对，排除失败保存记录；不再用“左目近似A、右目近似B”组合拒收。实时文案给出具体配对编号，手动采集仍可保留近似帧，自动采集继续拒绝确认重复的原图。离线回放位置测量统一使用整板单应中心，不再用局部角点均值代替。

### 33.4 验证

`tests/test_guidance_field_feedback.py` 覆盖针孔/鱼眼小板位移、倾斜变化与亚像素噪声、不同历史配对的错误组合、保存失败排除、未达成方向不持续占据建议，以及同高度分组调度。现场图片专项通过 `MULTICAMERA_FEEDBACK_IMAGES=<解压后的images/rgb_pair_1>` 显式启用；从压缩包解压至临时目录，仅校验原图，不把素材或设备地址提交仓库。

对同一20对素材，新版20次建议的方向为左4、右5、上5、下5、正面1次，区域行号切换2次。20对仍是去重样本且可解姿态，不表示20对都通过自动姿态验收。前后报告各自复核源图哈希；此对比仅证明推荐行为变化，不是Windows真机自动采集或独立标定精度验收。本轮不改标定求解器、相机服务、RGB–IR配置、点云、导出格式或原始采集数据。

## 34. 预览与采图检测结果可追溯（2026-10-28）

### 34.1 第一层边界

本节解决“预览126点、采图日志不是126点”时结果来源不清的问题，不改变ChArUco/平面板检测参数、原图标定点、求解器或数据筛选，不加入多尺度补检、角点平滑或补足点数。实时预览仍最多15fps、检测最多2fps；RGB检测输入现已按第54节保留原图分辨率，但预览与点击采图并非同一帧，检测数仍可能不同，这种差异不被隐藏。

### 34.2 采图快照与原图回看

`PreviewPair` 附带检测来源、左右检测结果、原板点总数、保存令牌与状态。双目采集在原图检测后保留独立检测快照，与采图日志使用同一次检测；已入库图像复用原有内存缓冲，未被标定器接受的原图也可回看，但不增加有效标定帧。新增“最近采图”入口显示编号、左右点数及保存状态，悬停提供完整相机组与来源信息；点击后回看原图和对应检测点，结束回看返回实时。采图后不自动冻结预览，自动采集入口与原有回看门禁不变。

保存状态区分“保存中、已保存、保存失败、仅内存原图”，保存回调及重试更新已打开的回看状态，不重复检测或重复新增观测。撤销移除对应快照并结束失效回看。图片保存仍只写无叠加的原图；查看时画出的角点不写回图像。快照在当前进程内使用，不新增会话JSON字段，也不宣称旧会话加载后能还原历史检测。

离线图像回看复用导入worker的全分辨率检测，并标为“导入原图”；交换左右角色时检测结果与原图同时交换。缩略图选择只更新离线索引，不再先触发一次缩放重检。RGB＋IR/辅助平面板同样保存采集或导入时的结果，失败检测也可查看；辅助保存失败仍不新增记录。没有原始检测快照的兼容路径标为“预览重检”，不能当作采图日志证据。

### 34.3 实时检测时效

只有本次检测对应的图像绘制该检测角点。后续未重新检测的新视频帧保留目标区域引导，但不叠加上一帧角点；点数明确标为“上次预览角点”、距检测时间及“非本帧”，新检测帧则标为“本帧预览角点”。因此实时角点并非每个15fps显示帧都有，不通过跨帧平滑制造稳定观感。

独立轻量定时检查在停止来帧后也能使超过0.75秒的预览检测过期，隐藏旧角点及点数并提示等待新帧，同时清除该路引导稳定证据。静态导入与原图回看不参与实时过期。过期检查不调用检测器、不改已采样本或累计覆盖；断开和清理仍使用既有连接状态流程。

### 34.4 验证与限制

`tests/test_capture_review_evidence.py` 覆盖日志/原图/快照一致、原图数组不变、检测快照持有副本、保存失败与重试状态、失败检测可回看、撤销、离线检测角色及旧角点时效。`MULTICAMERA_CORNER_IMAGES=<原图目录>` 可显式启用现场10对素材专项，验证0016左右原图114/122点在回看中保持一致且不触发缩放重检；三种窗口尺寸检查回看布局。

此层只保证显示来源一致与可追溯，不解决检测器的尺度敏感性，也不能仅凭静态图片诊断设备静止时的全部视频波动。Windows最终EXE、连续真实视频及硬件精度需另行验证；没有因界面结果一致就提高标定精度的结论。

## 35. 第一阶段：双目原图采集验收

### 35.1 操作及状态

RGB双目实时采图增加“验收：自动（允许部分角点）”和“验收：完整角点”，默认自动。此选项不更换相机模型，也不修改已有数据。两路原图各检测一次，验收、日志、内存观测及回看复用该结果；不得用15fps预览中的历史点数代替原图验收。

- **可采纳**：加入两路内参及双目观测，并推进有效计数和采图计划。ChArUco通过ID建立对应关系，允许部分角点；完整策略另要求左右均为当前板满点数。
- **待复核**：几何数据可读，但板面清晰度、亮度/对比度、共同点分布、时间配对或鱼眼模型兼容性存疑。不加入求解器，不推进有效计数或计划。当前不提供强制转为合格的操作，建议重拍或撤销。
- **拒收**：无有效检测、共同点不足、坐标/ID/尺寸无效、几何退化，或者完整策略下未满点。不计入有效样本。

手动采图的待复核/拒收照片仍保存无叠加原图，以便排查，独立显示“已保存”与验收结论；这不代表标定合格。自动采集未通过原图或质量门禁时不自动保存、不推进，避免连续保存不合格图片。底部持续显示实际左右点数、共同点数、结论及重拍原因，最近采图入口及回看也保留状态；不会因下一帧预览满点而覆盖采图结论。达到建议数量后持续提示“可继续补拍，不代表精度达标”。

窄窗口持续显示主要原因与问题总数，验收策略右侧的列表检查图标打开“验收详情”，可查看全部原因、相机组及原图编号；图标有悬停说明和可访问名称。避免将多项长原因全部塞进底部操作区而裁切文字或遮挡白平衡与采集按钮。

### 35.2 判定依据与边界

`calibration/capture_quality.py` 负责轻量、确定性的单对原图检查：复用 `prepare_observation` 验证ID、有限坐标与已有分辨率，检查角点在图内；左右至少6个共同点，图像分辨率一致，共同点不能共线或退化。共同点在物理板面的凸包面积不足全板内角点凸包的30%时待复核，避免少量集中点被当作完整覆盖。

仅在图像共同点凸包内部计算板面亮度和清晰度，减少背景占比影响：亮度10/90百分位，低端大于200、高端小于60或动态范围不足35时提示检查曝光/照明；Laplacian方差小于50时提示对焦、静止后重拍。黑白格本身的饱和像素比例不单独作为拒收条件。以上阈值为启发式质量预警，受分辨率、噪声和照明影响，不是角点定位误差或精度保证，也不自动调设备曝光。

针孔模型允许跨帧部分点；鱼眼采集核对已有各帧共同点数量，数量不一致时待复核，不静默裁点或切换模型。既有显式“鱼眼兼容：跨帧共同角点”筛选仍可用于历史/离线数据。实际场景是否适合针孔或鱼眼需多帧验证，此阶段不做自动模型选择。自动采图原有稳定性、姿态、重复构图和软件时间配对门禁继续有效；手动单张质量验收不能证明时间稳定性或硬件同步。

此阶段新增门禁只作用于RGB双目实时ChArUco采图；RGB+IR/声像仪的棋盘格、圆点板仍走已有完整网格检测流程，不宣称其新增同等质量门禁。原有离线图片无验收记录时维持原路径，不追溯改变历史数据。距离/姿态调度、多尺度补检、设备参数写入、点云性能均不在本阶段范围。

### 35.3 保存、导入与撤销

新增可选伴随文件 `images/<pair>/quality_XXXX.json`，记录schema_version=1、板配置、模型、策略、结论、原因及左右/共同点数；不更改原图像素或rig导出格式。后台保存worker先原子写验收记录，再按既有事务保存左右原图；只有图片保存成功才推进会话图片计数。保存失败仍按原队列保留原图重试，不能因质量可采纳就跳过保存失败门禁。

重新导入同目录 `left_XXXX/right_XXXX` 原图时，如有伴随验收文件且结论非“可采纳”，则跳过并说明原因；无法解析的记录使导入明确失败，不静默放行。无伴随记录按旧版导入路径处理。伴随文件是本应用的验收记录，不是第三方标定文件；复制图像排查时应一并保留。撤销移除对应原图、伴随记录和快照，仅回退实际加入过的观测；待复核或拒收不会弹出前一张合格观测。

### 35.4 验证

`tests/test_capture_quality_policy.py` 覆盖部分ChArUco与完整模式、亮度/清晰度、点分布、ID与尺寸、软件时间配对、鱼眼数量兼容、手动留图但不计有效、自动拒收不保存、导入隔离和撤销。补拍重新标定回归使用有亮度纹理的占位照片，纯黑占位图不再绕过真实采图质量检查。

现场10对原图离线复核均可通过自动基础验收，其中0016原图左114/右122点、共同110点；这仅验证部分角点不会被满点规则误拒收，不代表该组标定精度或连续视频稳定性验收。另检查1000×680、1280×800、1440×1024 Qt布局及源码标定自检；Windows原生和真实相机连续采集仍需现场验证。

## 36. 自适应引导的距离调度与前后移动反馈

### 36.1 中央先变距离，再分高度补边缘

自适应推荐在原有位置、倾角、重复构图及构图可行性检查之上，增加明确的中央距离阶段：中距离基础姿态3次、近距离2次、远距离2次，随后依次补中排左右、上排、下排；每个边缘区域优先连续安排2次不同姿态。同高度先完成局部动作，再调整支架高度，不采用逐个角点上下交替的路线。

上述次数是**建议预算，不是通过验收的距离覆盖数量**。只有新增、唯一且有效的双目配对才推进调度；重复图像、重复刷新、保存失败或已撤销记录不会被当作新增样本。重算使用现有数据快照，删除/撤销后回到对应推荐阶段。预算避免操作者手动保留不匹配姿态后某个建议无限占据推荐；跳过某距离阶段的建议不意味着该距离已采到。

真实距离分布另行统计同一配对中两路都落在同一档的样本，显示近/中/远数量；不会将两路各自不同帧的数量拼成有效配对。后续区域仍按欠缺姿态和尺度评分。靠近边缘时可能需要较小板面以保留边距，这不是要求全部后续照片都拍远距离。中央近距离阶段专门提供大板面观测，不能用边缘小板面替代。

每个候选继续检查两路分辨率下的图像边距（默认4%）、预测角点间距（默认至少6像素）和倾斜占位。不可实现的距离候选不强推，不静默放宽图像边距或角点间距；摘要列出受限的距离档位，并说明不计为已完成。无图像尺寸时可行性只能暂估，实时检测仍是门禁。

### 36.2 距离与画面占比的关系

近、中、远是相对采集尺度，不是固定米数。使用**内角点区域**的等效画面占比，不能与整个标定板含边框的占比或角点完整率混同：

- 远距离目标2.5%，实时允许1.2%–4%。
- 中距离目标7%，实时允许4%–12%。
- 近距离目标16%，实时允许12%–45%。

有可靠内参和通过重投影检查的PnP时，用板中心相机Z深度 `z`、内角点物理凸包面积 `A`、焦距和图像尺寸估计正视等效占比 `A * fx * fy / (width * height * z²)`。相同深度下仅转动板面不应被误认为远离；部分角点的PnP仍以已知ID恢复整板中心深度。此量是针孔正视等效尺度，鱼眼也使用该归一化近似，不宣称为真实畸变图中的可见板面积、欧氏测距或最优拍摄距离。

没有可靠双路PnP时，用观测角点图像凸包面积，按这些ID覆盖的物理板面比例作整板近似；明确提示“占比近似，非实测距离；先保持板面朝向”。该回退仍受透视、畸变和倾角影响，不以它证明真实距离，也不绕过自动采集已有的双路PnP姿态检查。一次反馈中不混用一侧PnP与另一侧无深度占比来作确定距离结论。

### 36.3 实时提示、自动采集及示意

任一路偏小且没有另一路偏大时，提示“向镜头靠近，增大板面；保持当前高度”；任一路偏大且没有另一路偏小时，提示“远离镜头，缩小板面；保持当前高度”；两路都达标提示“距离到位，保持前后距离”。一侧偏小而另一侧偏大时提示检查共视/相机配置，不取平均后指挥前后移动。只要任一路不满足目标档，自动采集不会因另一侧达标而放行；双路检测过期或姿态估计失败继续沿用原有门禁。

顶部横幅用简短距离动作和内参状态，避免小窗口多行测量挤占预览；悬停可见完整推荐依据、两路占比、目标范围及估计来源，右侧实时状态保留未达标原因。离线回看不显示实时移动指令，未启用自适应及后台分析失败时沿用固定计划。

QPainter线稿示意保持标定板物理尺寸不变，正面距离动作沿相机Z轴移动；近距离靠近、远距离远离，不再靠缩放板模型模拟距离。倾斜示意在目标距离处展示转动，不把已固定的转轴边同时平移。动画依旧为定性演示，无硬件控制或真实位姿测量功能。

### 36.4 回归与现场素材复核

`tests/test_adaptive_distance.py` 覆盖中央中→近→远阶段、同高度分组、唯一配对调度、针孔/鱼眼倾角与部分角点不改变同深度等效尺度、两路冲突/无效值、不可行远距离、实时靠近/远离/到位文案和自动采集门禁，以及固定尺寸板沿Z轴移动的示意模型。既有自适应、撤销/保存失败、原图验收、固定计划和线稿全姿态边界回归继续执行。

test2保留的20对现场图片重新离线回放：20次后续推荐包含中距离2次、近距离2次、远距离16次（不含采第一张前的中距离建议）；支架高度顺序中排→上排→下排，仅切换2次。回放照片并未按新建议重拍，实际双路同档统计仍是中距离20、近/远0，明确区分“推荐过”与“真正采到”。边缘小板面较多源于共视边距约束；这项回放不证明现场移动体验、Windows表现或标定精度提升。

## 37. 评审修复：撤销安全、距离闭环与回放一致性

本节修正第35–36节初版的边界。原有历史回放数据不重新解释为采集完成，最新行为以下述实现为准。

### 37.1 撤销与验收记录的删除顺序

`CalibrationSession.delete_saved_pair_frame()` 先删除左右原图，全部完成后才删除 `quality_XXXX.json`。任一路图片因文件占用、权限等原因删除失败时，异常继续返回UI，验收记录保留，防止尚在磁盘上的拒收照片失去标记后走旧格式导入路径。左图已删而右图失败时不伪装为完整事务成功；保留孤立原图与记录，后续可重试清理。此修复不承诺跨文件原子删除。

回归模拟左右两路分别被Windows式文件占用：确认记录仍在、未通过验收或缺失配对不会被读取为标定输入，解除占用后重试可清理剩余原图与记录。

### 37.2 客户端与回放共用距离判定

`adaptive_guidance.evaluate_distance()` 是实时提示、原图姿态门禁与离线回放的公共距离判定入口。两路均有可靠PnP时统一使用正视等效尺度；任一路缺少PnP时两路整体回退到整板占比近似，不混用两种来源。推荐器的双路距离统计也遵循相同回退规则。

离线回放测量增加 `distance_area`，逐帧判定增加 `distance_match`、`distance_message`、`distance_source`（pnp/composition/unavailable）。回放匹配与尺度覆盖统计不再使用原始角点凸包占比代替等效尺度；每一步仍只用此前图片建立的内参，禁止借用最终标定结果。构图候选不可行时不放行。静态回放不能评估同步、运动稳定或白平衡，因此“构图匹配”不是完整自动采图合格结论。

距离档位边界与统计统一：远距离 `[1.2%, 4%)`、中距离 `[4%, 12%)`、近距离 `[12%, 45%]`。过小或过大、超出所有验收档位的图像不能被当作距离覆盖；4%归中距离，12%归近距离，避免门禁通过但覆盖计数落到另一档。

### 37.3 实际距离缺口的补采闭环

初始中央中→近→远建议保持不变。初始7次建议后，按**实际去重后的双路同档样本**检查目标：中距离至少3组、近距离至少2组、远距离至少2组。尚缺的可行档位追加中央补采，优先近、远、中，顶栏明确标为“距离补采”。不同档位成组建议、组内变化倾角，不要求反复升降支架。

补采每个档最多追加2次、总计最多4次新增唯一配对机会。该上限用于避免操作者持续手动采入不匹配构图时陷入无限循环，**不是把未补齐距离标为完成**。达到目标后缺口消失，不再请求该档补采；持续不匹配达到上限时先继续同高度分区路线，摘要始终保留“距离待补”及未完成说明。不可行档位不强推，继续显示受限。撤销、删除或保存失败排除后，从剩余证据重算缺口与补采状态。

该闭环是有界启发式调度，不是最优实验设计或最终标定质量门槛；补采预算耗尽不等于可以忽略缺口。用户仍可通过现有手动导航切换固定计划补拍，随后重新分析数据。

### 37.4 原图完整性测试与验证

真实图片Qt流程测试通过 `_discover_stereo_pairs` 枚举所有实际使用的左右文件后计算哈希，支持 `left_XXXX/right_XXXX`、`XXXX_left/XXXX_right` 和左右子目录等导入命名，不再只用一种glob导致test3检查空集合。空集合直接测试失败，任一路内容变化必须被检测。

新增 `tests/test_capture_integrity_fixes.py`，扩展距离专项，覆盖删除失败、导入隔离、哈希命名、边界档位、真实覆盖清除缺口、单档与总补采上限、重复数据及撤销重算、客户端与回放的PnP/回退一致性。test3的10对原图重新跑Qt导入/回看/左右交换/标定/标准OpenCV导出，全部20张原图参与哈希校验；新的逐帧回放直接提供统一距离判定及未补齐的近/远缺口。上述验证是本机Qt和离线数据验证，不代替Windows EXE或真实相机连续采集验收。

## 38. 指定距离补采入口与单动作主提示

### 38.1 直接补近距离或远距离

RGB双目单组实时自适应采集时，右侧引导区顶部提供“补近距离”“补远距离”。点击后将活动目标改为中央、正面和所选距离，不需要从头浏览固定计划或手动等待推荐回到中央；点击本身不拍照、不增加有效数量、不创建验收记录，也不写设备参数。

指定补采会持续到用户点击“返回推荐”，不会因采入一张新图、后台分析更新或达到本档建议数量而自动跳走。达到建议数量后显示“本档数量已补足，可返回推荐”，这不等于姿态覆盖或最终标定质量通过。返回后恢复最新数据下的自适应目标，不回放过期推荐。

按钮仅在单组实时自适应采集时展示；离线导入、固定计划、多组同时采集、融合任务、结果页及原图回看不展示。保存中/失败、相机参数准备、导入/标定/模型评估或后台推荐尚未完成时禁用入口。自动采集定时器本身不禁用入口，仍需全部门禁通过才拍摄。已知分辨率下无法满足角点间距或边距的距离按钮禁用，悬停给出条件说明。

补采目标仅绑定当前相机组、URL、标定板、模型和会话，不写入rig或历史session。临时回看后保留目标；切换固定计划、任务、模型、标定板、相机配置或替换会话时退出指定补采。第一次采图创建会话时保留刚选定的补采目标，避免初次会话建立将目标重置；新数据/撤销会重新计算缺口，但不自动覆盖用户的指定距离。

预览目标、QPainter示意、实时姿态验收和原图自动采集使用同一个活动目标。指定补采不绕过可靠内参/PnP、双路可见、距离、位置、倾角、清晰度、0.6秒稳定、软件时间配对、重复构图或原图保存失败门禁；无可靠内参时提示手动采集基础样本，不能因为点击了补采就自动放行。

### 38.2 一次只强调当前动作

自适应模式顶部主提示按第一个未满足条件推进，不再同时要求前后移动、平移和旋转：

1. 保存失败、相机参数准备或任务进行中：先处理阻碍，保持当前位置。
2. 双路不可见/检测过期：先让标定板同时进入左右画面。
3. 距离未达标：只提示靠近、远离或双路冲突检查，保留当前高度。
4. 距离到位而位置未对齐：只提示向目标区域平移，保持前后距离。
5. 位置到位而倾角未达标：只提示调整板面方向；无法可靠求姿态时提示按示意手动采基础样本，不伪称已验证。
6. 构图到位后检查清晰度，再提示保持静止至少0.6秒。
7. 全部到位但重复构图或未满足软件配对：提示对应阻碍；均通过才显示“保持静止，可以采集”。

其他条件仍在右侧状态、紧凑摘要及悬停详情展示，不从实际门禁中移除。条件退化时回到当前最先缺失的步骤；后台数据重算时提示保持位置并等待，不用旧分析放行。固定计划保留原有展示，离线回看不出现实时操作指令。

### 38.3 验证边界

`tests/test_capture_actions.py` 覆盖补近/远及返回、数据更新期间保持指定目标、首次建会话/替换会话、模型/固定计划切换、回看保护、不可行分辨率、保存及参数准备禁用、自动定时器兼容、逐步提示和回退、PnP/重复构图/软件配对门禁，以及1000×680、1280×800、1440×1024布局截图。已有自适应和原图验收回归继续执行。

本轮不改变标定求解器、距离阈值、原图验收规则、设备曝光/白平衡参数、点云或mrcal研究代码。Qt截图和模拟状态不能证明真实操作者移动体验；近/中/远完整实拍与Windows最终EXE仍需现场验证。

## 39. Windows构建效率优化与耗时记录

### 39.1 保留验证，减少重复工作

Actions继续完整执行：安装依赖与pip check → 干净提交/版本/标签检查 → 全部应用回归 → 源码标定自检 → PyInstaller → 最终EXE标定自检 → 分发压缩 → 校验与Release发布。任何源码回归、EXE接口/数值/版本检查或分发任务失败均不能发布；不通过减少测试、复用旧EXE或跳过源码安装提速。

Python依赖使用现有pip下载缓存，缓存键只取 `pyproject.toml` 和 `packaging/windows/constraints.txt`，不再因 `MultiCamera.spec` 改动失效。运行库、构建工具和pytest合并一次安装，优先使用符合约束的wheel，不再在现代CI环境每次升级pip或测试前单独安装pytest。本地脚本仅在pip低于21.3、不支持所需PEP660 editable安装时升级pip，保留旧Python环境兼容。

SciPy仅对optimize/spatial继续收集所需子模块，显式扫描过滤tests/test/benchmarks，保留least_squares和Rotation等运行模块；不排除实际求解器、OpenCV接口或Qt运行插件。最终能否运行仍由新Windows构建的EXE验证，不以静态过滤测试代替。

### 39.2 分发压缩与并行调度

`scripts/package_windows.py` 使用标准库，不新增应用依赖。CI在最终EXE自检成功后才调用；便携ZIP与可选Inno安装器使用两个工作线程并行生成，仅共同读取同一份验证后的dist，等待两者结束。任一失败都会返回非零退出，不能降级为只发布另一个成功产物。

- ZIP使用Deflate level 1，文件名统一为 `MultiCamera_portable_<version>_x64.zip`，EXE位于ZIP根目录而非额外套一层目录；包含隐藏资源和中文名称。相比原有Compress-Archive减少压缩工作，下载体积可能增加。source不允许空EXE、指向包外的符号链接或把输出目录放回source中。
- Inno默认从 `lzma2/ultra64` 改为 `lzma2/normal`，保留SolidCompression。速度、构建内存和下载体积之间作明确取舍，不改变安装后的程序文件。手动Actions输入 `installer_compression` 可选normal/max/ultra64；标签构建默认normal。
- 安装器先寻找已有ISCC，找不到才执行Chocolatey安装。编译器仍必须实际存在，安装命令失败不会继续发布。
- ZIP与安装器分别在独立临时目录生成，成功验证后替换本地输出；失败不覆盖已有对应文件。ISCC即使返回0但没有产生新的非空EXE也会失败，不允许拿旧安装包冒充新产物。这里的本地输出替换不适用于已发布Release，后者仍禁止覆盖。

CI同时请求ZIP和安装器时并行；只请求便携版时不安装Inno。`build_windows_installer.bat` 本地有ISCC时使用同一工具的 `--installer-only` 保持只生成安装器的路径；没有ISCC时生成统一命名的便携ZIP作为替代，不重复执行一次PyInstaller。

本地选择压缩等级：

```bat
set MULTICAMERA_INSTALLER_COMPRESSION=max
scripts\build_windows_installer.bat
```

`normal` 优先构建速度，`max` 用于更紧凑的压缩，`ultra64` 保留旧高压缩模式；选择后仍需执行全部校验。直接调用 `package_windows.py` 只做分发压缩，不能代替完整构建/自检入口。

### 39.3 性能指标与发布附件

工具生成 `installer_output/packaging_metrics.json`：总压缩墙钟耗时、是否并行、每个任务耗时、压缩设置、产物字节数，以及ZIP源文件数/未压缩字节数。任务失败时尽可能记录失败阶段并返回错误。并行任务耗时之和不等于实际墙钟时间，比较时应看总耗时及相同Open3D变体/压缩设置。

Actions将该报告写入Job Summary，并随Release及SHA256SUMS发布；原来的源码和EXE两份标定报告继续保留。依赖安装、回归及PyInstaller阶段耗时仍查看Actions各步骤时间，不把压缩报告误称为全构建时间。

### 39.4 验证与限制

`tests/test_windows_packaging.py` 用标准ZIP往返校验根目录结构、隐藏和中文资源、原文件不变、实际并发启动、任一分发失败整体失败、旧ZIP/旧安装器保护、版本及输入检查、路径空格与中文、SciPy收集过滤和CI验证顺序；不调用真实ISCC，也不生成可运行Windows EXE。

本机可做Python回归、源码标定自检、YAML解析、编译及Ruff检查，不能执行Windows批处理、真实Inno编译及最终EXE验收。本轮未触发远端构建，暂无可靠的Windows优化前后耗时/体积对照，不承诺固定节省比例。下一次经授权构建应对照相同变体的Actions步骤时间、packaging_metrics及实际下载体积，再确认默认压缩取舍是否合适。

参考：[Inno压缩等级](https://jrsoftware.org/ishelp/topic_setup_compression.htm)、[PyInstaller收集接口](https://pyinstaller.org/en/stable/hooks.html)、[GitHub依赖缓存](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)。

## 40. 采集布局与结果信息展示改进

### 40.1 预览与引导分离

RGB双目采集保持“左右视频并排 + 右侧引导”的横向布局，不再因窗口高度达到900像素而切换为上下排列。引导使用独立的 `QScrollArea`，可滚动查看姿态状态和展开的数据分析，不带动视频及相机信息一起滚动。引导示意在紧凑布局下允许缩小到280像素宽，避免出现横向裁切。

底部采集操作保持整行紧凑布局，缩略图可展开/收起；大窗口不再为收起的图库和竖排操作区分配大片空白。切换采集、标定、结果及RGB/辅助任务时，沿用现有控件和信号归属，不重建相机连接。

相机图像区允许随可用高度缩小，始终等比适配，不用固定120像素的图像最小高度挤压多行状态文字。图像控件自身尺寸变化时重绘缓存帧；状态文字按当前宽度更新最小高度，并提供完整悬停提示。该行为同时覆盖实时帧和离线静态图片，不改变原图。

### 40.2 主提示、示意与回看一致性

自适应流程以 `_instruction_stage` 为当前操作阶段。示意区复用主提示文字：仅距离调整、倾角调整阶段允许动作动画；稳定检查、画质检查、配对、保存及等待分析阶段显示静态目标参考，禁用重播和暂停操作，避免主提示要求静止而示意仍要求移动。静态目标不是实时姿态测量或验收证据。固定计划仍可播放原有动作示意，用户暂停状态保持独立。

进入或结束图像回看时，通过 `review_changed` 同步刷新引导摘要，不等待250毫秒周期刷新。原图检测记录、检测来源及角点数在回看切换时一起更新；没有原图记录的图像仍明确标注预览重检，不冒充原图验收。

### 40.3 引导记录与有效图像分开计数

固定/融合引导标题改为“引导步骤 · 已记录 x/y”；融合数据区显示“图像配对 n组 · 建议13组”。前者表示步骤记录，后者表示有效配对数量，两者不互相推算。导入8组图像而没有对应步骤记录时，允许同时显示“已记录0/13”和“8组 · 建议13组”，并用悬停说明解释区别；不将导入数量认定为姿态覆盖已验证。

### 40.4 结果与导出状态

结果页默认展示各组双目RMS、左右内参RMS和基线毫米值，辅助结果展示融合RMS和参与组数；风险提示常驻，完整质量说明仍可展开。基线来自 `norm(T) * 1000`，不改变内部米单位和变换方向。

保存摘要按实际成功返回的标定文件逐项列出JSON、OpenCV完整、OpenCV双目标准和Kalibr等状态；校验清单不计入标定文件数量。某格式失败不会被其他格式成功覆盖，错误摘要仍保留，完整错误及文件列表放在悬停提示和操作日志中。

结果目录仅显示经过中间省略的本次目录名称，避免长绝对路径挤占质量信息；悬停和“复制路径”保留完整绝对路径，“打开结果目录”沿用真实本次结果目录。旧结果过期、没有结果、全部保存失败等状态继续分别处理，不把历史文件当成本次成功。

配置表格选中行使用深蓝 `#086ed3` 背景和白字，正文对比度达到4.5:1以上；不再使用浅蓝底白字。

### 40.5 验证边界

新增/扩展Qt回归检查：1000×680、1280×800、1440×1024窗口下预览及状态信息完整可见、视频与文字不重叠、图片缩放尺寸不越界、引导独立滚动、底部操作区紧凑、静止阶段动画停止、步骤/配对计数分离、关键结果常驻、部分导出失败与长路径可复制，以及表格选中色对比度。

可选真实图片流程使用 `MULTICAMERA_REAL_PAIR_DIR` 指向离线配对目录，`MULTICAMERA_UI_QA_DIR` 指定截图目录，运行 `tests/test_real_pairs_ui.py`；其中直接断言切换回看后原图检测摘要立即一致，并校验源文件哈希不变。test3用于本机流程验证，不代表真实相机实时同步、硬件颜色或最终Windows EXE显示验收。本轮不修改标定求解算法、设备参数或点云流程。

## 41. 第二轮界面复查修复

### 41.1 低矮窗口的辅助面板

窗口高度低于900像素时，配对图库和操作日志改为各自的非模态工具窗口，由原来的展开按钮打开，不阻塞主窗口采集。工具窗口复用原有缩略图、分页、选中状态和日志文本，不创建第二套数据或相机连接。两个面板同时打开也不会挤压视频和采集操作区。

关闭工具窗口或按Esc会同步取消对应展开按钮；再次展开仍保留原来的数据。高度恢复到900像素及以上时，面板自动返回原布局位置，工具窗口隐藏，展开状态保持。关闭主窗口时隐藏辅助窗口；白平衡资源或后台任务拒绝关闭主窗口时，不提前清理正在使用的面板。

### 41.2 动作方向与距离档位分离

自适应主操作同时产生阶段、提示文字和示意动作。动作使用 `approach`、`retreat`、`hold` 或固定计划的 `plan`，不再仅凭 `distance` 阶段启用动画。距离冲突、检测等待、稳定、配对及保存等情形使用 `hold`；只有明确需要靠近/远离时才播放相应动作。冲突时重播及暂停按钮不可操作，自动采集门禁不变。

距离动作的示意保持X/Y位置与板尺寸不变，只沿相机Z轴移动，靠近减少Z、远离增加Z，与目标处于近/中/远哪个档位无关。动作变化进入缓存键并重置播放进度，避免沿用旧方向的缓存或播放阶段；该模型仍是非实测示意，不替代PnP或标定质量判断。固定计划原有位移/倾角演示保留。

### 41.3 展开示意和长标签

侧栏内展开示意时增加可用高度并显示视角选项，不再把画布最小宽度强制恢复到360像素。侧栏保持280像素画布最小宽度，视角选择与说明纵向排列；展开/收起按钮的提示及可访问名称同步变化，仍可在原位置收起。

紧凑侧栏的数据分析按钮仅显示“数据分析”，完整配对计数及累计覆盖说明放在其悬停提示中；当前配对数量仍在摘要区显示。达到建议数量后的长提示不再作为不可换行按钮标题撑宽侧栏。

### 41.4 结束回看的刷新顺序

结束回看时先清理回看缓存、恢复实时或离线预览及检测计数，再发出 `review_changed`。主窗口收到信号时即可使用已恢复的计数、来源和图像更新摘要，避免将上一张采图原图的计数临时标注为当前预览检测。开始回看的原图记录同步流程不变。

### 41.5 回归验证

新增或扩展以下检查：三种窗口尺寸中图库/日志同时打开、工具窗口关闭/Esc/重开以及浮动与内嵌切换；展开示意时按钮可见且无横向滚动；已达建议数量的长文案不撑宽侧栏；三种距离档位分别核对靠近/远离Z轴方向、距离冲突动画停止及解除后的正确方向；结束回看返回后立即核对摘要和左右画面计数，不等待250毫秒刷新。

test3真实图片流程再次校验导入、回看、左右交换、标定、结果及标准OpenCV导出，原文件哈希保持不变。本机使用Qt离屏渲染及受控姿态状态验证，尚未替代Windows EXE实机窗口管理、真实相机或硬件精度验收。

### 41.6 Windows字体与补采操作自适应

Windows构建#20在1000×680窗口复现返回推荐按钮右边界溢出1像素。补采操作布局现使用按钮实际 `sizeHint()`、布局间距和侧栏视口宽度计算需求；三项操作放不下时，将返回推荐放到第二行，补近/补远保持首行。恢复空间或隐藏返回推荐时回到单行。父视口尺寸变化和布局请求均重新调度，不依赖固定字体、像素阈值或平台分支。保留严格的窗口可见性检查，新增大字体视口完整性检查。

## 42. Windows发行文件统一命名原则

后续可下载二进制统一包含产品版本与架构：安装器为 `MultiCamera_Setup_<version>_x64.exe`，便携包为 `MultiCamera_portable_<version>_x64.zip`，例如 `MultiCamera_portable_0.1.2_x64.zip`。规则写入AGENTS.md，打包、上传、下载说明、性能报告与SHA256SUMS必须引用同一实际文件名。

`scripts/package_windows.py` 的 `--version` 同时用于ZIP和安装器，接受纯数字MAJOR.MINOR.PATCH；省略时读取 `src/multicamera/__init__.py` 的唯一版本来源，不依赖当前工作目录。CI和本地Windows批处理显式传入已校验的PRODUCT_VERSION，包括仅生成便携包和无Inno时的回退路径。性能报告增加version字段，并记录带版本号的实际输出文件名。

新版本ZIP不覆盖其他产品版本或遗留的无版本号ZIP。Release上传只选取当前PRODUCT_VERSION的精确文件名，不通过通配符误用旧包。既有Release（包括windows-build-40741401157）不改名、不替换；旧OpenCV归档审计兼容版本化及遗留名称，要求下载后恰好存在一个便携ZIP和一条对应文件名的校验记录，存在歧义或校验不符即失败。

回归覆盖版本化ZIP往返、不同版本及旧文件并存、非法版本拒绝、跨工作目录默认版本、并行压缩和失败保护，以及CI/本地脚本中的名称与参数一致性。此次仅更新后续命名规则，不重新发布历史安装包。

## 43. PnP距离可信度与近中远自适应补缺

### 43.1 距离定义与适用范围

双目自适应引导复用 `adaptive_guidance.measure_pose()`。已知标定板物理尺寸、对应角点和同模型同分辨率的内参时，使用PnP估计位姿；鱼眼先反畸变到归一化射线求解，再回原始像素空间检查误差。深度使用完整标定板角点边界中心变换后的相机Z值，不直接使用依赖板坐标原点的tvec.z，也不使用部分角点的均值。返回 `depth`（光轴深度，米）及 `range`（板中心直线距离，米），引导以前者为准。物理单位来自标定板配置的米值，打印比例或板尺寸输入错误会直接影响距离。

沿用完整内角点凸包的正面等效占比 `A=C/Z²`，其中 `C=角点物理凸包面积*fx*fy/(图宽*图高)`，从而区分真实前后移动和倾斜/部分角点造成的面积变化。近中远不是通用的固定米数；每路相机按自身C将档位占比边界换算为目标深度范围 `sqrt(C/A_max)` 到 `sqrt(C/A_min)`。远、中、近分别沿用1.2%–4%、4%–12%、12%–45%的工作区间，属于可调工程参数，不是最优标定距离定理；检测间距及候选边距检查继续有效。

### 43.2 局部可信度检查

在原有内参有效性、角点空间分布、正深度、非极端侧视和重投影RMS不超过3像素检查之外，增加距离专用检查：

- 内参RMS不超过2.5像素；当前角点最近邻间距中位数至少6像素；重投影RMS不超过该间距的10%。
- 在板中心坐标系对旋转和平移进行有限差分，计算重投影雅可比J。用SVD检查秩；最小奇异值不超过最大值的1e-10时拒绝距离约束。
- 使用不低于0.5像素的观测噪声底值及残差自由度修正，计算局部协方差近似 `sigma²*(JᵀJ)^-1`，提取板中心Z标准差 `depth_std`。相对标准差不超过5%才通过局部深度检查。
- 位姿返回 `reprojection_rms`、`depth_std`、`distance_reliable`、`distance_reason`。不可信时可保留姿态供诊断，但不显示可信米数、不计可信距离覆盖、不通过自动采集距离门槛。

这里的“可信”仅指固定当前内参、板尺寸与点对应关系后的局部数值检查，不是完整测距精度证明；不包含内参不确定性、板变形、错误打印比例、系统误差或所有平面PnP多解风险。0.5像素、5%及间距误差阈值均为首版工程默认值，需要真实相机数据验证后调优，不是经过硬件统计标定的置信概率。没有新增深度学习模型或运行依赖。

### 43.3 双路验收与界面反馈

`evaluate_distance()` 是实时预览、原图自动采集复核、历史统计和离线回放的共用入口。只有两路都通过局部可信度检查才使用PnP口径；两路不混用PnP和占比。已有内参但位姿不可解/不匹配、单路失败或距离低可信时，返回 `uncertain`，说明原因，提供占比近似作为人工参考，暂不自动验收；无内参冷启动返回 `composition`，明确“非实测距离”，倾角自动验收仍不可用，可以手动采集基础样本。

两路必须同时处于目标区间，不对相反移动指令取平均。处于区间内但局部深度正负2倍标准差触及边界时，暂不通过距离验收，提示小幅调整至档位中部；动画保持静止，避免临界状态交替播放靠近/远离。此缓冲不是时间滤波，也不是完整95%置信区间。清晰度、至少3次检测且持续0.6秒的稳定门槛、软件配对、原图重检、去重及保存门禁不变。

侧栏姿态显示可信板中心深度或“距离待确认”；顶部推荐摘要显示两路可信深度，详细提示包含各相机目标米数范围。低可信或历史回看不沿用上次的可信米数。临时引导内参继续独立于正式标定结果，不写入会话标定文件。

### 43.4 距离分布与调度

`GuidanceAnalysis.distance_counts` 统计去重、双路同档、通过共同距离门槛的PnP样本；`approximate_distance_counts` 单独记录无可靠PnP时可用的占比近似样本。近似样本不抵扣可信距离缺口。内参产生或数据撤销后从现存原始角点重算，不永久固化冷启动结论；回放快照也输出这两组计数。

中央阶段中/近/远的可信样本目标分别为3/2/2组。有可信距离观测时，优先补尚未满足的中距离，再近距离，再远距离，跳过已满足档位；没有可靠位姿时保留原有有界冷启动请求计划，避免等待内参而无法开始采集。中央阶段仍最多7次有效构图调度，后续距离补采最多4次、单档最多2次；预算耗尽继续分区并保留缺口提示，不误报完成。保持先中央、同高度侧边，再上排、下排，避免反复升降支架。手动指定补近/补远、返回推荐及撤销路径不变。

### 43.5 验证与边界

新增 `tests/test_pnp_distance_guidance.py` 覆盖针孔/鱼眼、倾斜、偏轴中心深度与直线距离、部分角点、远距离误差增长、内参拟合偏大、角点过密、重投影相对误差、边界缓冲、单路失败、冷热启动重评估、按实际距离跳档及撤销/去重。扩展实时UI和回放一致性测试；原图自动采集仍调用同一门禁。运行方式：`PYTHONPATH=src python -m pytest -q tests`，并执行源码标定自检、编译与变更文件Ruff检查。

本轮不改变标定求解器、设备曝光/白平衡、RGB-IR流程或点云算法，不引入mrcal运行依赖。合成测试和Qt离屏验证不能替代真实相机卷尺距离对照、Windows EXE实机验收或完整标定精度评估。

本轮最终完整回归1084项通过、6项跳过；源码标定自检、Ruff、编译与差异检查通过。test3按匹配会话的15列10行、40毫米格子配置回放10组图片，原图哈希复查通过；引导模型重评估为10组中距离、近/远各0组，缺口持续提示。回放只使用此前样本产生的引导内参，不载入最终标定内参，不能据此证明实际米数精度。可信与低可信状态的Qt截图检查确认详细数值不会挤占紧凑摘要，冷启动标识和补足后返回推荐提示保留。

## 44. 实拍视频反馈：指引文案与预览布局稳定性

### 44.1 信息分层与动作衔接

针对实拍视频中“中距离”与“远离镜头”同时出现、算法术语不易理解、保存后姿态切换缺少解释的问题，双目自适应采集统一采用：

- 顶部主文案为“操作：将标定板远离/靠近相机，保持当前高度”，明确移动对象是标定板。
- 目标摘要明确标注“目标：中距离/近距离/远距离”，同时标明正面或哪一边靠近镜头；指定补采仍保留模式名称和返回推荐入口。
- 面向使用者的“冷启动：倾角未验证”替换为“建立初始模型，请手动采图”；操作区直接给出板面正对相机或指定边靠近镜头，然后手动点击采集。内部分析与日志仍可保留技术术语。
- 准备、保存、失败状态使用明确的等待或重试提示；与目标档位分开，不把系统等待误解成需要继续移动标定板。

保存结果由真实原图复核与保存状态驱动，不依据预览满点数推断。侧栏固定两行显示最近原图编号、保存中/已保存/失败、左右原图角点及“完整角点可采纳/部分角点可采纳/待复核/拒收”。保存成功但未采纳的图像明确提示未计入有效样本，保留验收详情按钮。已保存提示持续至下一次采图或撤销后重算；顶部确认保存结果并由主操作继续提示下一姿态，不用瞬时消息作为唯一反馈。进入回看、切换数据源或无现存原图时不伪造采图结果。

示意图左侧标题改为“相机视角 · 不镜像”，右侧立体视角为“位置示意 · 相机固定”，不再使用容易被误认为操作指令的“板面放大”。紧凑模式不重复显示顶部动作全文；展开后仍可查看示意与观察选项。动画依旧是非实测的定性示意，不改变采图或标定算法。

### 44.2 防止实时状态挤压预览

新增 `ui/widgets/stable_text.py` 中的 `StableTextLabel`：依据字体预留固定行数，超长内容仅在绘制时省略，底层完整文本继续用于读取、辅助功能和详情。顶部操作/目标、侧栏摘要、示意说明、底部状态和相机预览信息采用有界文本，不随角点、帧龄、错误原因或提示换行重新分配视频空间。字体变化可重算行高；窗口主动缩放、单双路放大切换仍按原有响应式布局工作。

特别移除相机信息条在 `update_info()` 及Resize事件中按 `heightForWidth()` 改变最小高度的逻辑。这两条路径会在同样分辨率下因“本帧/上次检测”、帧龄和较长错误信息改变图像区域高度，是预览内部跳动的直接原因。

底部就绪、白平衡准备和重试保存使用同一个固定高度的Qt堆叠区域，不临时插入额外倒计时/取消行。紧凑采集区保留一行验收摘要与显式详情按钮，压缩行间距；不通过隐藏错误或删除验收入口换取布局稳定。原图保存与验收仍分开，失败提示不会当成成功。

固定占位用于实时RGB双目采集；离线和融合的非准备状态不保留无用空槽，避免挤占已有的融合引导和配置空间。融合准备时仍可显示取消/等待状态，不改变其采图业务逻辑。

实时双目采集期间，图库和日志在所有窗口尺寸下均使用可关闭的非模态工具窗口；展开它们不再挤压视频。离开实时采集后保留原有按窗口高度决定内嵌/独立显示的策略，宽高足够时恢复原位置。复用原有控件、数据和连接，不重新创建视频流。引导详情在独立侧栏滚动，不改变左右预览尺寸。

### 44.3 验收方法

新增 `tests/test_capture_layout_stability.py`，在1000×680、1280×800、1440×1024窗口下，记录左右CameraView与内部图像标签的屏幕位置及宽高，逐项切换准备、完成、保存中、保存失败、长文案、展开引导、加入真实采图记录、打开图库和日志。要求每个状态的几何值与基准完全一致，预览区不出现垂直滚动；不仅比较外框，还比较实际图像区域。另测固定行数省略时保留完整文本、初始模型的明确手动操作及部分角点原图的持久反馈。

既有回归同步检查验收摘要与详情入口可见、宽屏采集也采用独立工具窗口，以及退出采集后正确恢复内嵌。此处验证针对Qt离屏布局及受控真实采集数据结构；Windows安装包、高DPI、多显示器及真实相机连续操作仍需实机复测。

最终完整回归1090项通过、6项跳过；另外启用test3真实图片Qt流程1项通过，完成10对图像的导入、三尺寸布局、回看、交换、标定、标准OpenCV导出及导入失败后保留旧数据，复查原图哈希不变。源码标定自检、变更文件Ruff、编译与差异检查通过。离线状态保留信息列最小宽度，字体/样式变化只重算状态行高、不清除调用方设置的最小宽度；源目录提示在三尺寸下完整显示。尚未执行Windows最终EXE或实拍设备验收。

## 45. 首轮采图与补采分阶段（2026-10-30）

本节取代第43节中“中央阶段按实时距离缺口调整顺序”的调度规则，距离测量、可信度及原图验收规则保持不变。不修改正式标定求解、畸变模型、校正矩阵、导出或设备参数。

### 45.1 有界流程

- `GuidanceRecommendation.phase` 为 `initial`（首轮）、`repair`（距离补采）、`review`（本轮结束待复核）或 `blocked`（无可用构图）。日志和离线回放同时记录阶段、首轮预算、补采次数和真实距离缺口。
- 常规首轮23组：中央7组按中、中、中、近、近、远、远调度，其余8个区域各2组，按同一高度分组移动。边缘优先中距离；不支持某种构图时只使用通过几何可行性检查的候选，首轮预算随可用区域调整。
- 首轮距离与倾斜次序不再因临时内参出现而重排。历史距离仍按现有可信PnP规则重新统计，但缺口只用于首轮结束后的补采，不在第8组插入补采。首轮进度表示有效去重样本推进的建议次数，不证明已按建议完成所有位置、姿态或距离。
- 首轮结束后按实际双路距离缺口调度，优先同档连续移动；最多4次建议、每档最多2次，同档内优先未观测姿态。缺口满足或次数耗尽后进入复核，不重新循环分区或自动无限采图。撤销、重复图及保存失败沿用现有数据来源规则重建进度。
- 复核阶段暂停自动采集，但不禁止手动补拍或指定“补近/补远”；指定补采仍须通过原图复核、距离可信度、重复构图和保存门禁。距离缺口未满足时持续显示，不把达到建议次数上限当作精度或覆盖达标。

### 45.2 一次只展示当前动作

顶部区分首轮/补采进度、当前距离及当前动作。调整距离时明确“板面先正对相机”，标题与示意都使用正面，不同时显示后续“上边近”等倾斜要求；位置平移阶段也暂不倾斜，到位后才呈现目标倾角。可靠PnP可用时，主要状态显示当前板中心深度与双路目标范围交集；冲突或不可信时保留原有阻断提示，不伪造米数。

最近一次原图的编号、保存及验收状态保留在独立回执区，不再用“本组原图已保存”反复覆盖下一步的距离或阶段提示。沿用固定行数状态区，不因阶段变化插入控件、展开面板或改变预览尺寸。详情保留完整建议和每路距离信息。

### 45.3 验证

回归覆盖中央7组固定距离顺序、内参由无到有时顺序稳定、23组前不补采、有界补采后停止、缺口保留、撤销/去重、距离调整与倾斜图文一致、结束后手动补采及三尺寸预览几何稳定。实际图片按历史前缀回放，不使用最终内参提前指导过去帧；该验证不代表Windows设备实拍或标定精度验收。

本轮完整应用回归1094项通过、6项跳过，最终引导/回放/布局专项62项通过，源码标定自检、变更文件Ruff、编译及差异检查通过。`calib_20261029_160743` 的29对原图前缀回放通过阶段断言：前23组首轮、接着4组补采、第28组起复核；回放后源图哈希不变。Qt三尺寸阶段切换中预览外框和内部图像区严格保持原几何；尚未制作Windows新包或验证真实设备连续采集。

## 46. 标定有效性检查与候选择模（2026-10-30）

### 46.1 有效性优先于训练误差

`calibration/health.py` 提供无UI依赖的 `check_intrinsics` / `check_stereo_health`。检查在后台标定完成、结果质量汇总及实际导出时执行；导出重新计算，不信任历史JSON中自报“通过”的字段。低RMS不能绕过有效性失败。旧JSON仍可读取供诊断，但异常参数不能再作为可用标定文件导出；已存在的文件按原子写入规则保留。

- 在全图33×25网格检查默认反算及100次/1e-10精化反算的回投误差，任一点超过0.25 px或出现无效值则拒绝。默认反算也必须通过，因为原生校正参数生成会依赖该路径；增加迭代不能掩盖默认路径异常。
- 在网格反算覆盖的径向区间进行4097点导数检查，要求径向映射保持严格正导数；有理分母同时做多项式求根及最小值检查，视场内零点、近零或非正分母均拒绝。该检查是保守数值验证，不是全局物理正确性的形式化证明。
- 从实际R1/R2/P1/P2反投输出网格，再正投原始镜头，检查正深度、映射Jacobian无折叠、左右共同有效输出比例至少50%，每路原图采样凸包覆盖至少10%。覆盖率约束用于识别“整幅输出只放大原图一小块”；不是只看是否黑边或主点是否位于画面内。部分旧结果缺少全部校正矩阵时，只做内参检查并标记视野未检查；新标定必须具有完整校正矩阵。

默认阈值针对当前全画面双目校正流程；不将严重裁剪当作自动修复手段，不适用于未经单独定义的极端数字变焦输出。RMS等级、采图覆盖和硬件测量精度仍是不同指标。

### 46.2 自动校验择模

标定页“优化策略”提供“自动校验择模：异常拦截与候选比较（推荐）”，自2026-10-31起默认启用；保留原标准/稳健/联合精修策略，显式手动策略同样执行有效性检查，但不会暗中替换模型。工作仍在单组后台worker内运行，全部标定维持串行队列。

`calibration/validated_selection.py` 只比较当前投影类型：针孔候选为标准、固定k3的受约束模型、有理模型；鱼眼为标准及固定k3/k4模型，不自动改变针孔/鱼眼类型。针孔/鱼眼的已有评估流程也增加各折有效性检查，跨类型改变仍走原有明确确认流程。

至少9对共同、唯一编号的观测时，标准模型即使健康且训练RMS不超过0.5 px，也必须完成留出验证；通过后优先保留标准解，不再额外搜索复杂候选。标准解的全量/留出训练模型失效、留出预测失败或误差较高时，再比较候选；自动更换要求至少9对共同观测。最多等距抽取30对，少于15对用3折，否则用5折；每折只使用训练帧初始化和求解，同折候选使用相同跨帧共同角点。验证按留出左图求板姿预测右图，并反向预测，汇总双向投影RMS。每个训练折也必须通过有效性检查，不以完整数据拟合良好代替折间稳定性。留出RMS必须为有限非负值，超过2 px的候选拒绝；标准候选可用且最优改进不足0.03 px时保留标准解。不足9对时仅可保留健康且训练RMS不超过0.5 px的标准解，明确标记未完成留出验证，不自动更换候选；无可用解则失败，不生成“可用”结果。

提供现成内参的工作流不自动重估这些内参，仅执行原计算及健康检查。原图及观测不修改，CV副本不复制原图以减少内存。

### 46.3 参数保真与结果追踪

有理模型先由同一训练集的普通针孔解初始化；内参使用 `CALIB_RATIONAL_MODEL` 和 `CALIB_USE_INTRINSIC_GUESS`。双目求解根据输入畸变向量保留有理、非零薄棱镜及倾斜项标志，避免固定内参时仍丢失高阶项；8项与14项零扩展视为同一畸变函数，不因此把误差指标错误清成NaN。不会截断非零高阶参数。

`calibration_info.health` 保存数值检查版本、限制、各路指标及视野检查；`model_selection` 保存候选、排除原因、每折训练/留出源编号、误差和最终选择。`stereo_flags` 为有效标志，`requested_stereo_flags` 保留原请求。质量评估显示选中候选和留出结果，不将同序列留出称为独立设备精度验收。

JSON与OpenCV保留完整参数；有理向量尾部仅是零扩展时，OpenCV畸变类型标为 `rational_polynomial`。Kalibr radtan无法无损表达非零高阶项时仍明确报错，不为凑齐格式而截断参数；其他已成功导出文件和失败明细沿用原会话manifest。

### 46.4 回归与发行边界

`tests/test_calibration_health.py` 覆盖低RMS坏模型拒绝、伪造历史“通过”字段无效、无黑边但过度裁剪、有理分母奇点、健康解保留、全候选失败、留出隔离、显式UI策略和有理参数保真。真实数据测试用 `MULTICAMERA_HEALTH_SESSION` 指定会话目录，原图及结果哈希前后复查，输出仅写测试临时目录。普通导出测试中的巨大切向畸变样例改为物理可用的小系数，保留原来的Kalibr截断容差和原子写入断言。

`--self-test-calibration` 额外执行已知有理镜头的固定内参双目求解、分母参数保真、健康解通过、折返镜头拒绝和异常变焦拒绝。因此未来Windows源码及最终EXE自检会运行同一检查；本地通过并不代表已构建或验证Windows安装包。

会话的新结果生成入口在创建代次目录、写入JSON和更新latest指针前也执行有效性检查，防止“先存JSON再逐格式导出”绕过门禁。无效代次不会改变已成功结果；原始读取/诊断归档接口不等同可用结果发布。

本轮最终完整应用回归1107项通过、7项跳过，变更文件Ruff、源码编译、差异检查及扩展标定自检通过。另启用真实会话及代次保护测试2项通过：29对图片中原结果被拒绝，自动策略选中有理模型，双目RMS约0.217 px、五折留出投影RMS约0.229 px，校正焦距约487 px。JSON与两种OpenCV文件正常保存并可读回；Kalibr明确记录为不兼容高阶项，原始会话全部文件哈希不变。本轮未提交、推送或生成Windows包，也未验证独立中远距离数据及硬件测量精度。

## 47. 修复小样本放行与校正矩阵一致性门禁（2026-10-30）

本节收紧第46节中的不足9对观测分支：标准解只有在健康检查通过且训练双目RMS不超过0.5 px时才能不经留出比较而保留。超过该值且共同观测不足9对时明确失败，要求补图或核对左右配对；不会把“无法验证其他模型”当作允许接受高误差标准解的理由，也不继续拟合无法验证的替换候选。非有限值或负训练误差同样拒绝。健康低误差的小样本标准解仍可保留，不强制更换模型。

健康检查版本升至2，除原有镜头和视野检查外，还验证：

- `R2 @ R @ R1.T` 与单位阵一致，`R2 @ T` 具有非零基线且对齐水平或垂直视差轴。
- 规范化后的P1以左校正相机为原点，P2平移项与其相机矩阵乘以校正基线一致，不接受零基线、错误尺度或反向平移。
- 在多个离轴位置和基线5/10/20倍的正深度上，使用P1/P2生成左右像素及对应轴的视差，再经Q恢复3D；相对三维误差不得超过1e-6，左右回投误差不得超过1e-4 px，恢复深度必须为正。直接比较3D尺度，避免只检查回投而漏掉整体深度尺度错误。
- Q必须有限且非零；整体乘非零常数仍视为同一齐次映射，不能误判。水平/垂直双目、基线正负及非零主点差均有正向回归。

新标定必须含可验证的Q。历史兼容路径若没有任何校正矩阵和Q，只进行内参检查；有P1/P2而无Q时仍校验投影关系，但标记 `q_status=unavailable`，不代表三维重建已验收。只要提供了Q就必须通过一致性验证，不能靠旧文件中的“通过”标记绕过。质量评估、导出和新会话结果发布共用此门禁；失败不覆盖旧文件或更新最新代次指针。

新增8对错误配对的高RMS失败/成功信号回归、健康小样本保留、P2零/倍增/反向基线、Q全零/深度尺度错误/方向错误，以及8种正常轴向与主点设置。导出契约的正向样例改为真实 `stereoRectify` 生成矩阵，不再使用彼此无关的R/T、投影和单位Q作为“合法”输入。原有数值、字段及文件原子性断言保留；应用自检新增P2基线、全零Q和错误深度尺度拒绝检查，后续Windows最终EXE运行同一入口。

验证：完整应用回归1124项通过、7项跳过；健康检查/导出/自检/校正诊断专项247项通过、1项跳过。真实29对原图专项1项通过，自动选择有理模型、五折验证及会话保存读回保持通过，原始文件哈希不变。最终源码标定自检确认投影基线、全零Q及错误深度尺度被拒绝，Ruff、编译和差异检查通过。未提交、推送或生成Windows新包；本地合成和序列回归不代表独立硬件测量精度验收。

## 48. RGB左目＋IR采图闭环与平面板质量检查（2026-10-30）

本节更新前述融合采集的13步固定建议：当前采用独立的17组基础计划、最多4组有限补采和持续完成状态。只改采集引导及原图验收，不改辅助内参求解或跨模态外参聚合算法，不引入联合BA、RGB/IR自动择模、设备曝光/白平衡写入或点云优化。参考仍为RGB左目与辅助单目，`T_rgb_left_aux`仍将AUX坐标映射到RGB左目；有效RGB双目结果保留。

### 48.1 阶段、进度与动作

- 基础计划17组：中央5组中距离（正面及左右上下倾）、2组近距离、2组远距离，再按中排左右、上排、下排安排8组边缘观测。边缘使用较小占比，给完整外圈留余量；不交替要求支架升降。建议步数不是姿态已经验收的证明。
- 首轮按质量可用的非重复配对数量推进，不在第8组开始插入补采，也不因临时内参生成而重排首轮。之后按可信IR距离缺口，最多4组补采建议；没有可信距离时不把缺口清零，达到上限仍结束本轮，避免无限循环。
- 完成后持续显示“本轮采集完成，可以前往标定评估”，记录日志，保留缺口说明和手动补拍入口。不将采图完成、17/21组或低训练RMS等同于标定精度达标。
- 分开显示已保存、源目录图像、检测有效、质量可用、非重复以及双路姿态可解数量。重复原图保留，仍可能位于质量可用数据中，但不推动非重复进度；原图回执明确提示重复，不静默推进。
- 位置以IR/辅助画面为主，同时要求RGB左目完整共视；倾角以RGB左目光轴为参考。只显示当前动作，先调距离/位置时示意板保持正面，不提前展示后续倾斜。建议几何包络不代表已知两路共视投影，实际完整网格验收仍必需。

### 48.2 距离与位姿可信度

采用IR角点凸包占比区分近、中、远，经验范围分别为18%–60%、6%–18%、1.8%–6%；不直接套用宽视场RGB相机的占比，也不要求不同焦距两路占比同档。候选目标还检查整板外圈、2像素边距和4像素间距；不可能的构图明确报受限，不标记完成。

有尺寸、模型及健康检查兼容的内参时，沿用针孔/鱼眼各自投影求PnP，检查重投影、正深度和局部深度约束。可信IR位姿用于显示板中心深度及目标米数范围，可信RGB位姿用于核对倾角；没有可靠结果时明确显示占比近似或倾角待人工确认。近远按视场覆盖设计，不是适用于所有镜头的固定米数。

至少6组非重复且双方向透视变化充分的平面观测，才在后台尝试临时辅助内参；最多均匀抽24组，要求RMS不超过2.5px及镜头反算健康检查通过。临时内参仅用于引导，不写最终内参、rig或标定导出；PnP局部可信不包含物理板尺寸、内参系统误差和硬件同步误差的独立验收。

### 48.3 原图验收与追溯

使用独立辅助板配置。已确认红外棋盘为7列×5行格子、80mm、6×4内角点（24点）；圆点板按其完整配置点数处理，不使用RGB ChArUco的126角点门槛。

原图检查包括完整网格、物点尺度一致、尺寸/有限值/点在画面内、非退化单元、相邻点最小间距4px，以及外推整板包络距边界至少2px。板内ROI局部亮度对比门槛IR为6/255、RGB为10/255；沿预期格边/圆点边缘检查信号一致率至少75%、边缘宽度P75不超过局部间距40%。不使用全图纹理或单一RGB拉普拉斯门槛代替热板可见性；伪彩亮度检查不等于测温数据验证。

实时稳定性要求至少3次有效检测、跨度至少0.6秒、距最近检测不超过1.5秒，并与窗口首帧及本次原图比较，不能靠连续小位移累计漂移。相同源帧时间戳不重复计数。点击采图时冻结稳定性证据，后台耗时或点击后移动不改变已捕获原图的结论。运动比较和去重可容忍检测器180度枚举变化，但从不修改标定点顺序；原图仍须人工核对物理对应。两路新帧年龄和软件配对时间差继续使用既有策略，报告明确“非硬件同步”。

原图通过后台`AuxCaptureWorker`重检，不将缩放预览角点当作本次采图证据。不通过时原图仍落盘并显示“待复核”及具体原因，不纳入辅助求解或非重复进度。两幅图片与初始待验收的`multimodal/quality_XXXX.json`在同一保存事务内提交，再原子更新正式验收记录，包含板规格、两路质量指标/阈值、原因、稳定性快照判定、软件时间差及对应确认状态。图片/初始验收写入失败整体回滚，不新增记录；正式验收写入失败或中途退出时保留待复核标记及原图，不纳入，重新导入也不能丢失该限制。

离线`rgb_left_XXXX.*`/`aux_XXXX.*`导入在worker中执行相同单图质量检查，保留来源目录，不修改源图。已有旁路验收记录判为待复核的图像，不能仅因重新检测成功自动纳入；没有记录的历史文件允许按原图质量重新检查，但明确无法验证离线时序稳定性。导入数量、步骤浏览、点击跳过均不确认姿态覆盖。

### 48.4 对应确认、线程和显示

两路原图/预览绘制0点和第一个相邻点的行方向箭头，像素原图及检测点顺序不变。融合标定前必须逐组核对并勾选“已核对两路角点对应”，新增质量可用数据或替换导入后清除确认。对称棋盘的跨模态物理点歧义没有被自动解决，低误差或人工勾选也不能替代独立外参验证；禁止盲目倒序或自动翻转掩盖问题。

后台引导只处理板、内参及观测副本，一次保留一个分析任务；上下文包含会话、板规格、相机角色/地址、模型、内参值及数据修订，过期结果丢弃。实时质量提示独立限至约1Hz，适用于预览关闭的通道；原图采集任务纳入忙碌保护。计算线程不操作Qt控件，原有图片保存事务与成功后计数语义保留。

阶段、动作、计数和最近原图各有固定行高占位，长内容可从悬停查看，不自动展开详细区或挤压相机预览。最近原图结果固定在采集按钮上方，可通过详情图标查看完整原因，不依赖滚动侧栏或短时提示。采集侧栏采用紧凑布局，把RGB内参前置提醒移至融合标定页，优先显示完整示意和导航。姿态清单标为“建议”而非已采/已验证；放大引导同步目标、距离大小及完成状态。断流、配对失败、检测过期或旧结果到达时撤销过期就绪提示，保留最近原图回执；新增确认控件只在融合标定页显示。

### 48.5 验证

```bash
.venv/bin/python -m pytest -q tests/test_aux_capture_quality.py tests/test_aux_guidance.py tests/test_aux_capture_ui.py
.venv/bin/python -m pytest -q tests/test_aux_workflow_followup.py tests/test_multimodal_preview.py tests/test_workbench_ui.py
.venv/bin/python -m pytest -q tests
.venv/bin/python -m compileall -q src scripts
.venv/bin/python -m multicamera --self-test-calibration --report /tmp/multicamera_aux_self_test.json
```

覆盖冷热启动、IR/RGB不同视场、首轮不提前补采、补采上限、重复/无效样本、完整网格/模糊/低对比、稳定性、保存/验收写入失败、离线重新导入不晋升、旧后台结果作废、物理点对应确认及三宽度固定占位。Qt实际控件的合成画面检查不能代替真实IR热对比、独立测距、Windows高DPI、软件配对运动误差或融合外参精度验收。本轮尚未生成新的Windows安装包。

本地验证：全量1271项通过、7项跳过；最终界面微调后相关专项202项通过，源码标定自检、编译、变更文件Ruff及差异检查通过。实际Qt导入按钮读取21组生成棋盘配对，完成基础/有限补采分析后仍可补拍，重复原图不增加非重复进度，导入源文件字节不变。三种实际窗口尺寸1280×800、1440×1000、1920×1080的基础/补采/完成状态切换，预览外框几何一致；仅验证布局和软件状态，不证明真实IR设备或硬件标定精度。

## 49. 新建会话反馈与隔离

新建或加载成功后立即刷新工具栏名称和完整目录提示，不再依赖窗口缩放或相机连接事件。新建期间禁用重复入口；同一次运行中新建、配置不变、内存无观测且磁盘只有元数据和空目录的当前会话，重复点击继续使用，不分配新目录。没有会话时点击目录按钮只展开会话菜单，不隐式创建目录。

显式新建时，已有内存采图、导入或结果必须确认清空；磁盘历史和导入源文件不删除。成功后重置引导、累计覆盖、最近原图回执、离线缓存及回看画面，保留实时连接和关闭预览的选择。首次采图自动创建会话只绑定保存目标，保留当前观测；两种入口不混用。后台计算或保存期间继续遵循已有忙碌保护。

初始化元数据写入或UI绑定失败时，仅回收本次创建且未使用、元数据未变、目录内无未知文件的新目录；加载的历史、已保存图像及外部写入一律保留。不会自动清理用户已有的多个历史会话。`results/calib_<时间>_<标识>/`是结果版本目录，不能当成重复会话删除。

回归入口为`tests/test_session_creation_ui.py`，覆盖即时显示、重复/重入、失败回滚、独立新建确认、自动绑定保留观测、加载显示、原图/回看清理及实时连接保留。Windows附件中的`0x8001010d`原生异常尚缺触发时序和原生堆栈，以上修复不宣称解决该异常。

本地全量1281项通过、7项跳过；最后补齐回看清理和实时帧恢复后，相关专项106项通过，编译、变更文件Ruff、差异检查及源码标定自检通过。以上是软件回归，不是Windows原生异常复现或真实设备标定精度验收；本轮未提交或打包。

## 50. RGB双目观测驱动采图调度

### 50.1 候选位置和尺寸

取代固定九宫格中心和单一7%中距离候选。在现有近/中/远档位内搜索参考占比，并在目标区域内部向安全边界移动中心；同时检查已知双路分辨率、印刷板外圈、4%图像边距及至少6px角点间距。倾斜候选按25度近似投影宽高并保留1.30倍包络余量。单路不可行即过滤，不通过强制退远满足固定格中心；无可行候选则明确受限。未获取分辨率时只用1280×720作暂定规划，实时尺寸到达后重新计算，不保证设备共视。

`GuidanceRecommendation`末尾增加可选`target_center`、`target_area`。中心贯通`CaptureTarget`、预览框、侧栏/放大示意、实时与原图验收及离线回放。`target_area`是候选内角点包络参考值，不是印刷板面积，也不是新的精确采图门槛；距离继续共用原有档位范围及可信PnP/占比回退规则。示意尺寸不等于标定后投影，不保证未知畸变、基线或极端倾角下的双目完整可见；原图验收和实际检测始终保留。

### 50.2 实际覆盖、倾角和距离证据

覆盖按每路6×8网格统计实际检测角点，一组观测对同格最多计一次；不按板中心给九宫格打卡，也不填充凸包内部或外推缺失角点。无效、冲突编号、未配对和重复构图仍不增加统计；保存失败数据沿用UI快照排除，撤销按剩余数据重建。

仅当双路PnP均通过既有局部距离可信检查、倾斜方向一致且角度符合实时验收范围时，计入实际倾角覆盖。正面要求两个角度均小于15度；倾斜主方向15–40度、次方向小于20度。每个非正面方向建议至少2组，缺失方向持续显示；请求次数只影响避免反复提示的软成本，从不计为实测完成。

首轮预算保留23组；前7次仍中央中3、近2、远2，采用正面及四方向倾斜建立约束，不因临时内参出现而重启。后续不再强制每区两组，而按真实覆盖、距离分布和倾角缺口推荐。距离调度偏好为近8、中11、远4，已满足的远档继续增加成本；这是可调工程启发式，不是行业标准、硬配额或精度合格线。用于调度的几何距离分布可含占比近似；正式可信PnP距离计数独立保留，低可信结果不能冒充实测距离。

### 50.3 低移动成本与后续安排

候选收益结合实际角点网格补缺、距离分布、未满足倾角、请求重复和接近已观测位姿的代价；左右眼分别统计，额外关注缺口较大的一路。收益距最佳值不超过0.18时，以高度变化、水平平移、参考面积比和转向成本选目标，高度成本高于水平平移。同等收益优先保持高度和前后距离，但不会为少移动永久排除高收益缺口。此实现不是Fisher信息量或全局最优路径求解。

调度只随已纳入的非重复观测、配置或内参快照变化，不随每帧预览噪声不断跳转。后台上下文额外包含相机ID、URL及实时图像尺寸；相应变化使旧结果作废。首轮不提前补采；结束后仍沿用最多4次、每档最多2次的基础距离补采，耗尽后复核，倾角及分布缺口不被清零，不会无限自动补拍。

顶部新增固定一行“后续安排”，无需滚动侧栏查看。基础采集显示“本张验收并保存后，预计中央某距离/倾角”；后续阶段显示预计优先补的距离/倾角缺口和少升降原则，最终目标由实际原图重新评估。复核时给出自主补拍或前往标定的路径。采集紧凑布局将当前操作保留为一行、目标/状态保留两行，并压缩上下留白，避免新提示挤出小窗口滚动条。该提示不提前消费进度，重复、未采纳或保存失败不当作目标完成；长文本省略并保留完整悬停信息，不自动展开或改变预览尺寸。

### 50.4 验证与边界

核心回归为`tests/test_guidance_observation_planner.py`；联合`test_adaptive_guidance.py`、`test_adaptive_distance.py`、`test_adaptive_capture_ui.py`、`test_capture_layout_stability.py`及回放测试验证。覆盖双路宽高比、低分辨率、动态中心贯通、实测/缺失点网格、倾角可信度、低成本同收益选择、连续按目标采图、不偏远/正面、撤销/重复、尺寸及地址过期和后续提示固定占位。

本轮不更换标定模型或求解器，不修改相机曝光/白平衡、原图保存及导出契约，不将该RGB策略直接套用IR。真实历史图片的离线推荐检查与合成闭环只验证软件行为，不证明实际测距精度或全局最优；最终Windows高DPI和实际设备连续采图仍需复测。

本地全量1315项通过、7项跳过；最终后续提示和顶部留白微调后，引导/回放/PnP/Qt验收相关102项通过、1项跳过，三尺寸布局及状态切换相关7项通过。源码标定自检、编译、变更文件Ruff及差异检查通过。当前检出没有`scripts/tests`目录，未宣称该不存在的测试集通过；诊断回放由应用测试中的对应模块覆盖。

24对真实图片复查保留原始字节，使用附件最终内参进行历史推荐诊断，而非因果回放或独立精度验收。远档已有至少4组的12个前缀中，推荐中档9次、近档3次，无远档；缺失左/右/上倾仍显示。25次分析中位约0.183秒、最大0.396秒，后台执行；这个本机小样本耗时不是Windows性能保证。按建议生成连续23组的合成闭环检查不产生远档/正面多数，不能替代真实设备、板平整度、共视、同步和测距验收。未提交、推送或生成Windows新包。

## 51. 引导提示与可行性一致性修复（2026-10-30）

### 51.1 融合预览和原图稳定性

辅助预览不再仅凭检测次数显示“可以采集”，而是使用与原图相同的`stability_reasons()`检查：双路各至少3次检测、跨度至少0.6秒、最近样本不超过1.5秒、图像尺寸一致，以及当前检测相对窗口首帧的位移达标。预览质量结果本身也必须在1.5秒内；断流、配置变更或质量过期时同时清除其检测快照。不足条件时显示“保持静止至少0.6秒”，不提前报就绪。

预览就绪仍不是本次原图的验收承诺；点击采图后保留后台原图重检和点击时稳定性快照。点击后移动或后台耗时不回头改变已捕获照片的稳定性判定，质量失败原图仍按既有待复核规则保存。

### 51.2 辅助板档位内参考占比

辅助候选以面积平方根与投影跨度、角点间距成正比的关系，直接求解当前区域中心下的可行面积区间，与IR近/中/远档位取交集。保留整板外圈、至少4px点间距以及图像边界2px余量；优先默认占比，默认值放不下时缩小，默认值无法分辨时增大，但不越出当前档位。无交集仍显示构图受限，不推进或标记完成。

`AuxTarget.target_area`新增可选参考值；默认占比可行时保持`None`，调整后传入`CaptureTarget`，同步侧栏/放大示意。该值只是绘图参考，不改变IR档位验收范围、原图质量门槛或17组基础/4组有限补采。此近似不包含完整畸变及双目共视投影，也不保证倾斜后的整板可见；实际两路检测和原图验收不能省略。

### 51.3 RGB后续提示

中央基础阶段的实际调度与条件预计共用`_bootstrap_target()`，从已通过双路几何检查的候选选择位置、距离及倾角。不再直接把未过滤的固定计划当作下一步提示；例如低分辨率下仅近档可行时，不预告中/远档。没有任何可行候选时提示调整板型或分辨率，受限距离仍在汇总中列出，不计作已完成。

提示继续使用“本张验收并保存后，预计”；未通过、重复或保存失败不提前消费步骤，后续配置和实际观测变化仍重新分析。不改标定求解器、设备白平衡、原图保存/导出或版本/发布状态。

回归入口：`tests/test_aux_capture_ui.py`、`tests/test_aux_guidance.py`、`tests/test_guidance_observation_planner.py`；包含真实Qt控件及后台检测结果、短时/过期/漂移/尺寸变化、宽棋盘档位内可行解、真正受限构图、低分辨率后续提示与下一步实际推荐一致，以及重复数据不推进。

验证：全量应用回归1332项通过、7项跳过；最后补齐无中央/无可行候选及质量结果过期检查后，三模块专项82项通过。源码标定自检、编译、变更文件Ruff和差异检查通过。以上不等同真实设备精度、Windows高DPI或最终EXE验收；本轮未提交、推送或打包。

## 52. 双目工作台信息层级与检查入口（2026-10-31）

### 52.1 采集页布局

紧凑双目工作台优先保留预览、当前动作和实时验收反馈。右栏先显示双路可见性、清晰度、稳定性及累计板角点ID/视场网格，随后为指定距离补采操作；相机与目标示意可独立折叠，播放、重播和展开控制与折叠入口共用一行，详细分析位于其后。展开只影响右栏滚动内容，不调整左侧预览几何。

最近原图回执移至底部采集按钮上方，保留两行固定高度，区分原图保存状态、左右角点数量与是否可采纳；完整原因继续通过验收详情查看。底部白平衡、验收策略与保存准备仍使用原有控制，图标沿用本地Lucide子集。离线模式隐藏实时采图/白平衡/姿态动画，保留原图源目录、左右交换和前往标定入口。

有效配对达到阈值时改为“达到计算门槛；覆盖与精度待验证”，不再显示“已达建议数量”。保留原有离线导入时调整最少数量的兼容行为；该门槛、自适应首轮预算、观测多样性和标定后质量是不同概念，不能互相替代。

### 52.2 标定对象、设置与执行状态

模型评估对象增加可见标签，并随明确选择的采集组和离线导入对象同步；仍可手动切换评估其他组。没有观测的组禁用模型评估，并提供原因。优化策略与数据筛选放入默认折叠的“标定设置”，入口保持短标签，当前策略及筛选摘要独立换行显示，不改变策略默认值或算法。

有观测时不再用无数据相机组的占位行挤压标定面板；无数据时仍展示空状态。组内标定按钮为次级操作，底部主按钮在单个就绪组时显示“开始标定当前组/重新标定当前组”，多组时明确就绪数量，悬停列出实际执行对象；原有全部就绪组串行调度语义保持不变。

计算期间组状态显示内参、外参及有效性检查正在计算；失败后持续保留“标定未通过”、首项原因与重试入口，完整错误可悬停或在日志中查看，不再还原成“已就绪”。错误绑定当时观测快照，数据或配置变化后移除失效提示；模型健康检查失败仍不得发布可用标定结果。

### 52.3 校正图检查

配对图库和逐帧误差检查选择原图后，同步更新对应组的校正预览，来源文字显示所选帧；只匹配相同组名及左右相机ID，不借用其他组图片。原图尺寸与标定尺寸不一致时清除旧预览并记录原因。

新增独立校正图窗口，支持适应窗口、放大、缩小、原始像素及滚动查看，允许随所选帧更新，不占据主工作台预览空间。参考线仅辅助观察，不等同同ID残差检查或独立精度验证。结果失效、数据重置或预览失败时关闭窗口并释放面板保留的校正图；不修改求解器、模型检查门槛、设备参数、原图或导出格式。

### 52.4 验证入口与边界

新增`tests/test_stereo_workbench_design.py`，联合`test_capture_layout_stability.py`、`test_stereo_capture_ui.py`、`test_offline_ui_state.py`、`test_calibration_workflow.py`和`test_model_selection.py`覆盖三尺寸反馈可达性、展开不移动预览、操作层级、评估对象、数量文案、计算/失败状态、帧选择校正联动、缩放、尺寸错误及结果失效清理。

`MULTICAMERA_DESIGN_QA_DIR`可指定该测试的界面截图目录；布局使用1000×680、1280×800、1440×1024实际Qt控件检查。实图离线导入用于检查对象与失败恢复，合成投影观测用于可重复的成功标定/结果显示验证，不代替真实设备精度、Windows高DPI或最终EXE验证。本轮不自动提交、推送或打包；实际测试结果在修改记录追加。

## 53. 默认自动择模与标准解留出验证（2026-10-31）

新建应用面板默认选择`auto_validated`，内部策略键与下拉选择一致；原始数据、板规格、镜头类型及数据筛选默认值不变。用户显式选择的标准、稳健和联合精修继续按原行为运行，配置/会话重置不偷偷覆盖用户当前选择。针孔/鱼眼类型之间的改变仍需原有确认，不因本次23对数据使用有理模型而将其他设备固定为有理模型。

### 53.1 验证策略与兼容性

`model_selection.policy_version=2`记录本轮规则，`validation_min_views=9`记录留出门槛。报告与所选候选增加`validation_status`：`passed`表示完成同序列留出检查，`insufficient_views`表示小样本仅通过基础检查；不把后者或缺少字段的历史结果显示成已完成留出验证。标准模型低训练误差不能绕过可执行的验证，留出失败后继续评估同投影类型的候选，全部失败仍拒绝发布。

保留0.25px反算、单调性、分母、校正几何/视场门禁及2px留出上限，不放宽数值有效性条件；小样本0.5px上限及候选改进不足0.03px时优先标准的规则不变。样本充足但存在折间不稳定的数据可能比旧版更保守地被拒绝，这是验证加强而非导出失效。留出仅是当前采集序列的内部验证，不是独立硬件测量精度验收。

### 53.2 结果展示

结果页无需展开详情即可查看最终候选、选择依据、同序列留出RMS及折数；手动策略或历史文件未记录时明确说明。显示同ID校正残差RMS、P95，以及最大单点残差及其源帧ID，帮助发现总体误差较低但个别角点偏差较大的样本；最大单点来源按该指标而不是逐帧RMS排序，缺失/非有限值不显示成零。

这些局部指标只提供复核证据，不新增自动删帧、不改变质量等级/接受门槛、不裁切高阶畸变以兼容Kalibr。JSON中保存完整候选和每折来源，现有OpenCV格式及会话生成代次继续沿用；结果失效或清空时同时移除旧验证说明。

### 53.3 验证

回归入口为`tests/test_calibration_health.py`、`tests/test_calibration_workflow.py`及`tests/test_stereo_workbench_design.py`，覆盖9/15对边界、标准低误差仍进行3/5折验证、训练/留出隔离、标准验证失败后选择健康候选、非有限/负值/超限留出拒绝、小样本限制、手动策略不暗换模型、补拍跨过9对门槛后真实执行留出，以及三尺寸结果展示/旧结果清理。

本轮复用`calib_20261031_101251`的23对只读原图，按应用worker与会话保存路径复测自动择模、校正残差、格式保真和读回；测试输出放入独立诊断目录，不覆盖原会话或ZIP。实际数值、测试数量与Windows/发布边界在修改记录追加。

## 54. RGB预览保留原图检测像素（2026-10-31）

RGB ChArUco预览不再将1280px等原图统一缩至960px。`MultiStreamView._detect_for_preview()`将收到的原图数组直接传给既有检测器，返回原图坐标，不进行先缩小再映射。原图ROI内的编码纹理、标记ID、完整性和OpenCV一致性校验均保留；不关闭`checkMarkers`，不丢弃异常编号来拼凑角点，不为了显示126点制造或平滑观测。

改动原因：同批远距离`left_0005.png`在960px检测时出现板外编号768/411和重复17，ChArUco输出0点；原1280px检测为116点，无上述异常。缩小检测的0点不能证明画面内没有板。新的预览输入与采图原图分辨率一致，避免这一额外的信息损失；两次检测的时刻仍可能不同，不能承诺预览数量等于点击采集结果。

每路检测上限继续2fps，UI渲染上限继续15fps；检测缓存仍标注本帧/上次、过期检测及时失效，旧角点不叠加到新图。相机流解码、记录回看、原图采集/保存、标定模型、候选调度和质量门禁均不改变。显示图缩放仅用于窗口适配，不能回流为检测/保存数据源；有已记录检测的图库回看继续复用原检测，不重算替换证据。

本轮仅覆盖RGB双目ChArUco路径，包括实时预览与没有历史检测记录的离线重检。融合辅助平面板仍保留原来的960px宽度策略及坐标映射，参数改名为`PLANAR_DETECT_MAX_WIDTH`以明确适用范围；低分辨率IR输入本来不触发该缩小。辅助检测策略应在其自身样本上另行验证后再调整。

回归入口为`tests/test_native_preview_detection.py`，覆盖1280/3840宽及灰度输入不缩小、像素及坐标保真、与直接原图检测一致、双路2fps节流、辅助平面板路径不变；联合原图回看、离线来源及融合预览回归。`MULTICAMERA_PREVIEW_SESSION`可指定本次23对真实图目录，检查0005/0006/0019原图与新预览结果逐点一致，并复查原文件哈希。

原图检测会增加某些分辨率和设备上的计算量。优先按检测频率、渲染频率控制负载，不能为降低CPU再次默认缩小RGB检测输入；后台调度及ROI属于后续独立优化。本地耗时仅作参考，未声称完成Windows实机CPU、4K连续预览或最终EXE验收。本轮不自动提交、推送或打包；实际验证结果在修改记录追加。

## 55. Windows标定侧栏宽度边界回归（2026-10-31）

Windows构建`41179935437`（`v0.4.2-build.20261031.1`，提交`171eebf`）在源码回归阶段失败：`test_long_result_text_wraps_without_clipping_or_horizontal_scroll[calibration-size0]`在1000×680窗口得到横向滚动范围1，要求为0；其余1364项通过、8项跳过。完整日志已通过已登录浏览器下载复核，未进入PyInstaller或最终EXE检查，不能将本次失败解释为OpenCV打包缺失或算法计算失败。

侧栏原先将“标定设置”、策略和筛选摘要合并进不可换行的`QToolButton`，其最小宽度随平台字形和边距变化，可能超过滚动区可用宽度。本地用16px工具按钮字体、左右9/10px边距复现同样1px溢出。修复保持“标定设置”短按钮，将当前策略和数据筛选移入单独的`_WrappingLabel`，按可用宽度计算高度；展开/收起后摘要始终可见，切换策略同步更新摘要和原有完整提示。不隐藏横向滚动条来掩盖裁切、不扩大主窗口最小宽度、不放宽原断言，不改变任何标定策略或计算门槛。

`tests/test_calibration_workflow.py`将长路径/错误文字用例扩为1000×680、1001×681、1280×800、1440×1024，分别在标定/结果页及默认/扩大工具按钮字形下验证零横向滚动及文本高度。字体压力通过局部Qt样式设置，不依赖本机安装Windows字体、不替换应用默认字体；断言失败时输出视口、内容、最小尺寸与场景，便于区分真正溢出和环境差异。`tests/test_stereo_workbench_design.py`覆盖多种策略/筛选切换、摘要持续可见、按钮最小宽度不变和展开/收起无横向溢出。

本地压力回归不等同Windows原生字体/DPI验收，修复后仍须通过新的Windows工作流完整回归、源码标定自检、打包和最终EXE自检。使用新预发布标签，不移动失败构建的旧标签或覆盖已发布附件；实际验证及提交/构建信息在修改记录追加。

## 56. 双目初始建模操作提示（2026-10-31）

自适应引导的双路模型尚不可用、且仍处于首轮时，顶部明确显示“初始建模 · 建模样本X组”和本张距离/朝向建议，不再用“首轮1/23”混同初始建模进度。下方固定单行持续提示“需手动采图，无需等待绿色提示”；至少样本数直接读取`BOOTSTRAP_MIN_VIEWS`，当前为6。已有6组或更多而模型仍未就绪时，提示继续手动补充左右、上下倾角，明确6组只是尝试建模的最低数量，而非自动切换承诺。

提示的建模样本数来自当前配置和会话对应的后台引导分析，含非重复和几何有效性筛选；不是点击数、已保存照片总数或精度合格数量。完整说明及下一张预期保留在提示中；保存/准备中仍优先显示等待，不引导用户重复点击。双路引导内参可用后，显示“姿态辅助采集 · 引导模型可用（非最终标定）”，当前帧仍须通过已有实时检查，不因为模型可用或达到6组而宣称可以自动采集。回看、离线、固定计划和首轮结束的既有状态优先，不持续显示旧建模提示。

稳定性主操作改为“保持静止，等待连续3次稳定检测”，详细提示说明观测跨度至少0.6秒、每路最多2次/秒检测时通常需要约1秒或更久；未检出有效板点时显示“待角点检测，暂不能判断”，不据此断言相机或标定板正在运动。自动采集复选框提示同步说明自适应初始阶段需手动采图，仍按原2秒周期及本次原图验收执行。

本次仅调整文案与状态展示，不修改角点、清晰度、稳定、距离、姿态、重复判定、自动采图条件或临时内参求解。复用原固定尺寸文本控件，不增加展开区或切换时的布局高度。回归覆盖0/1/5/6/8组未建模提示、已有引导模型但当前帧未就绪、保存等待优先级、撤销/配置变更、离线及三尺寸阶段切换的预览几何稳定；不自动提交、推送或打包。

## 57. 连接、标定数据与引导计数的提示一致性（2026-10-31）

### 57.1 两类就绪状态

底部视频状态只说明连接和软件帧配对，正常配对显示“视频帧已配对（仅连接检查）”，使用中性文字而非采图通过色。完整提示保留软件时间差，并明确不代表角点、清晰度或姿态验收通过；不改变流管理器的状态接口、同步范围或原图验收行为。顶部主操作及最近原图回执分别说明当前应执行的动作与上次采集的结果。

### 57.2 标定数据与引导样本分别计数

ChArUco部分角点满足原图验收时，仍可保留为标定数据；不能因非满点而统称废片，也不能把每张可用照片等同于已推进引导。既有标定验收与引导几何用途不同：标定验收允许满足条件的局部观测，引导几何需要至少6点、物理角点凸包覆盖全板内角点凸包的至少50%，并通过原有图像、单应几何及重复检查。50%是板坐标中的分布范围，不是角点数量比例或图像画幅占比；不在本次调整30%等既有标定验收条件。

`GuidanceAnalysis.sample_decisions`由实际分析筛选过程逐源帧产生，记录`included`、`excluded`或`duplicate`，以及相机对应的排除原因或近似重复源帧；只是当前分析快照，不新增会话或导出契约。最近原图回执仍保持固定两行，显示保存/原图验收与“已计入引导”“未计入引导”“重复构图，未增加引导样本”等独立状态。完整提示说明未采用原因、左右角色及标定数据仍保留；配置、保存或观测修订不一致时只显示“引导复核中/待复核”，不借用旧结论。

实时检测同时区分“未检出板点”和“已检出但引导几何不足”。局部角点过于集中时，提示让更多板面被识别；不再把板面几何不足描述成板没有进入画面。该描述不放宽自动采集门槛，也不将检出板点等同于稳定、清晰或姿态合格。

### 57.3 分析失败与重新分析

自适应分析失败后保持策略选择不变，明确显示“自适应分析失败，自动采集已暂停”。此时既有固定示意只作手动参考，不宣称已切换策略，也不继续声称等待已结束的后台分析。顶部增加带提示的重试图标；用户可重新分析，或显式切换固定计划后手动采图，原图仍须验收。保存、白平衡准备或其他任务忙碌时禁用重试，回看、离线及固定策略不显示该入口；失败和恢复不改变原有自动采集条件。

重试按钮隐藏时保留布局占位，回执与主提示继续使用固定尺寸文字控件。回归入口为`test_adaptive_guidance.py`、`test_adaptive_capture_ui.py`和`test_capture_layout_stability.py`，覆盖各目局部观测、同编号缺失/冲突、尺寸差异、重复来源、实际手动采集与后台保存、过期快照、失败重试和三尺寸预览几何稳定。本地使用受控检测观测与实际Qt采集/保存流程，不访问设备、不代替真实标定精度或Windows最终EXE验收；实际测试结果在修改记录追加。

## 58. 融合采集的统一术语与状态口径（2026-10-31）

RGB左目与辅助单目采集统一使用四种名称：“参考动作”表示浏览的构图与姿态；“原图已保存”表示确认落盘的实时配对；“验收通过”表示通过既有原图质量、稳定等检查的配对；“引导采用”表示当前分析实际采用的检测有效、尺寸一致、非重复配对。离线图像单列“源目录”，不显示为复制到会话或实时保存。检测有效数和姿态可解数移入完整提示，不再在主统计里混用“质量可用”“已记录”和“非重复进度”。

### 58.1 参考动作不等同采集进度

融合参考控件启用`recommendations_only`时，标题、清单、展开、上一项/下一项及区域提示统一称“参考动作”。浏览第2项显示“浏览参考动作2/17，非采集进度”，不再同时显示无关的“已记录0/17”。手动浏览不增加原图或验收记录，也不意味着手动拍照不计数；当前操作明确说明手动采图仍需原图验收，采图后恢复推荐。

统计区移到示意图之前，保持固定两行，便于同时看到参考动作和真实数量。引导进度由`AuxCaptureGuide._update_counts()`按当前分析快照统一更新，面板刷新不再以验收通过数暂时覆盖引导采用数。分析未完成显示“待更新”，失败显示“分析失败”，不伪造零数量、不把失败描述成仍在运行。`AuxGuidanceAnalysis.sample_decisions`是实际筛选过程的源帧采用结果，仅用于当前回执，不改变会话或导出结构。

### 58.2 连接、验收与保存分别说明

原“配对采集已就绪”改为中性“视频配对：视频已配对/视频未配对”，完整提示说明只检查连接和软件时间配对，不等同原图质量或推荐姿态通过。距离和姿态文案根据实际辅助模态显示“红外”或“声像仪RGB”；位置以辅助画面为准，板边倾斜以RGB左目为参考，保持现有计算方向与单位。

最近原图回执固定两行：第一行是源帧编号、保存状态与验收状态；第二行显示角点/引导采用结果，或优先显示未采用的关键原因。完整详情保留两路角点数、所有验收原因及物理对应/精度边界，详情按钮读取完整提示而非省略后的摘要。重复图可以“验收通过”但“引导未采用（重复构图）”；保存、检测或配置过期失败的回执持续保留，不被周期刷新中的旧成功状态覆盖。验收记录写入失败仍保留原图并明确不纳入验收通过或引导采用。

本轮没有增加后台保存队列或同一组原图的保存重试；原图落盘失败时准确提示“请重新采图”，不提供尚不存在的重试保存承诺。不修改原目录、双图暂存提交和待复核记录保护机制。稳定提示统一为“等待连续3次稳定检测”，完整说明仍须跨越至少0.6秒，实际等待取决于检测更新，不承诺静止0.6秒即通过。内部存档继续使用既有`可采纳`/`待复核`状态，界面将其解释为“验收通过”/“验收待复核”，不迁移历史文件。

### 58.3 结束状态与布局验证

“本轮采集完成”改为“本轮引导结束”，保留手动补拍；只有既有前置条件满足时才提示可计算，否则直接显示缺少RGB内参、有效数据或物理角点对应确认等具体原因。未建立可信姿态时允许按提示手动采图，但仍需本次原图验收，不跳过完整网格、稳定或质量条件，不将部分ChArUco规则套用于红外棋盘。

回执高度不随内容变化；短窗口依据侧栏可用高度约束参考示意尺寸，支持滚动和展开查看，不通过隐藏滚动条或放宽几何断言掩盖裁切。回归覆盖原图保存1组而验收/引导为0时浏览第2项、重复保留但引导不增加、保存失败后不回显旧成功、分析失败与数据变化恢复、过期快照、离线来源、声像仪模态称谓、完整详情及1000×680/1280×800/1440×1024预览几何稳定。测试入口为`test_aux_capture_ui.py`、`test_aux_guidance.py`、`test_aux_workflow_followup.py`、`test_workbench_ui.py`及`test_interaction_safety.py`；实际结果在修改记录追加，不等同Windows最终EXE或真实设备精度验收。

配对图库和“最近采图”入口同步使用“验收通过/验收待复核”；`PreviewPair.display_quality_state`仅对平面板观测映射显示名称，`quality_state`仍保留原存档值。RGB双目ChArUco原有显示和内部验收状态不因该映射改变。

## 59. 融合回看隔离与异步浏览保持（2026-11-01）

### 59.1 回看与实时采集隔离

进入RGB左目/辅助原图回看后，`AuxCaptureGuide`通过`review_changed`立即显示原图编号和“实时指引已暂停；请先结束回看，再采集实时图像”，不继续要求用户调整实时板面。回看期间采图按钮禁用，`_on_capture_aux()`也检查回看状态，直接调用不会创建检测/保存任务或隐式退出回看。操作路径改为先点击既有“结束回看”，再从实时画面采图；不把回看文件重新当成实时采图源。

进入与退出回看均清除实时质量、稳定历史和检测缓存，并更新预览失效时间。回看前启动、在结束回看后才返回的旧检测结果不能重新恢复“可以采集”；后续采集需重新满足既有连续检测与原图验收条件。回看过程不清空采集数据、验收记录或已保存回执，后台引导分析仍可更新统计，但不能覆盖回看提示或切换当前参考动作。

主窗口回看信息统一使用`PreviewPair.display_quality_state`，因此图库、回执和回看画面均显示“验收通过/验收待复核”；内部`quality_state`和历史文件仍保留原值。RGB双目ChArUco显示不因平面板的名称映射改变。

### 59.2 手动浏览不被后台分析打断

手动选择参考动作后，当前序号、构图及浏览说明保持不变；分析进行中或完成时只更新当前数据快照和计数。`_analyzed()`不再无条件关闭手动浏览，`render()`优先处理回看和浏览模式，不在这些模式下改写参考动作序列。实际采图完成、数据更新或明确重置时才恢复推荐；过期分析回调仍按上下文忽略，后续最新回调也不能覆盖用户在等待期间重新选择的动作。

结束回看后，如果原先在手动浏览，则回到同一参考动作；否则在当前分析可用时恢复实时建议。回看状态优先于浏览提示，避免在历史图像旁继续显示实时移动要求。本轮不新增“返回推荐”按钮，不改变已有采集、标定算法、质量门槛、文件目录或变换方向。

### 59.3 验证入口

`tests/test_aux_capture_ui.py`覆盖真实后台worker分析中浏览、连续修订与过期结果、采图后恢复推荐；`tests/test_workbench_ui.py`覆盖图库进入回看、按钮及直接采集调用均不落盘、结束回看后保存新实时帧、回看期间分析完成仍保留状态、旧预览回调失效及两种验收显示名称。验证同时检查内部存档状态不迁移。

实际Qt在1000×680、1280×800和1440×1024检查分析后仍保留第3个参考动作，及1280×800的回看暂停/恢复提示；源码标定自检和回归结果在修改记录追加。这些是软件流程和受控图像检查，不等同真实设备精度或Windows最终EXE验收，不自动提交、推送或打包。

## 60. 双目引导动画区布局整理（2026-11-01）

本轮仅实现动画填满右栏、实时状态单行化、常驻动作图例三项布局调整。保留默认“板面＋立体视角”、现有展开/收起和暂停/重播行为，不增加独立窗口或单视图切换，不修改采图、检测、姿态判断、标定或保存算法。

### 60.1 宽度与信息层级

姿态详情列由有布局归属的完整容器承载，紧凑模式隐藏整个容器，而不是只隐藏其中的文字控件。由此清除原先仍占用约50至58px的空列，动画画布使用右栏内容区全部宽度；本机三尺寸实测由280px增加至330至338px，具体像素随平台字体、边距与侧栏宽度变化。

实时状态只显示一行“板点/清晰/稳定”，保持既有判定来源；姿态数值、几何不足原因和稳定规则保留在完整提示及顶部当前操作中，不因缩短状态文字放宽门槛。“数据集累计：板角点ID/视场网格”、各目累计值及有效配对数移入“数据分析”展开区，使用原有数据修订缓存计算。离线页继续显示当前预览角点与检测来源，累计统计同样放在数据分析中，避免混淆本张与整个数据集。

### 60.2 常驻图例

`StereoSchematic.legend`位于画布正下方，白色实线表示固定边、黄色实线表示移动边、蓝色虚线表示目标。图例在紧凑和展开状态均显示，不再依赖展开后的说明；完整提示明确它只解释动作示意，不表示实测姿态、检测结果或精度。保留现有双视角画法、几何方向、暂停状态和绘制缓存。

紧凑组件按内容高度排列，图例保持自然固定高度，避免大窗口中被弹性空白推远。状态、画布和图例不随检测文案变化改变相邻预览布局；短窗口仍允许独立滚动右栏，不扩大主窗口最小尺寸、不隐藏滚动条掩盖内容。

### 60.3 验证与预览

`tests/test_stereo_capture_ui.py`新增三尺寸的画布/内容区等宽、单行实时状态、图例可见与间距、数据分析归属、滚动可达及预览几何不变检查；`test_stereo_schematic.py`覆盖紧凑/展开常驻图例；`test_offline_ui_state.py`对应验证当前帧计数与累计统计分离。联合自适应、工作台、回看、布局和融合回归验证兼容性。

实际Qt预览保存在忽略跟踪的`data/diagnostics/stereo_animation_layout_20261101/`，使用`test3`的0008左右原图作静态素材回放，包含1000×680、1280×800、1440×1024和数据分析展开截图；输入图像哈希保持不变。这些截图不表示正在连接相机，也不构成真实设备精度或Windows高DPI验收；本轮验证结果在修改记录追加。

## 61. 采集工作台及关联窗口重构（2026-11-01）

本节在第60节布局调整基础上，按确认后的设计稿重组界面。实现仍为PySide6与QPainter，不把生成设计稿作为运行时图片，不引入浏览器、网页界面或新的依赖。现有未提交的引导、原图验收与融合回看修复继续保留；本轮不调整检测、调度、模型选择、标定求解、变换方向或磁盘写入契约。

### 61.1 主窗口与固定布局

- 顶部按品牌、当前会话、连接状态及设备操作排列。会话沿用新建/打开/目录入口；相机设置与RGB参数保留原有安全检查，不因重排而自动访问设备或改写参数。
- 流程栏维持采集、标定、结果三个阶段，采集阶段在右端提供当前任务的导入入口。入口代理现有双目/辅助导入按钮，定时同步文本和启用状态，点击时重新检查当前任务及原按钮状态，不绕过后台忙碌检查。
- 双目采集主区使用左侧预览/图库/回执与右侧独立滚动引导。原来的全宽提示条移入右栏，主预览不会因为当前动作、下一步文案、分析展开或保存状态而重新分配高度。
- 最近原图回执位于左侧图库下方，独立于当前帧的实时状态。查看详情回看最近采集原图；回看不代表重新验收，也不自动重拍。离线来源明确为源目录原图，不显示为本次实时保存。
- 底部使用全宽固定操作区，主采集按钮保持有限宽度，自动采集、撤销、跳过与前往标定各有独立入口。导入主入口移到流程栏；原图保存、验收策略、失败重试、白平衡准备与取消能力保留，不因精简外观而删除。

图库在高度至少900px的窗口中内嵌展示，较矮窗口默认收起，用户点击后使用既有独立工具窗口。单纯缩小窗口不自动弹出图库；展开或关闭图库、操作日志不挤压采集预览。单页缩略图按宽度显示1至6组，缓存只保留当前页，不复制原图缓冲。该高度阈值仅用于展示密度，不是采图或标定质量参数。

### 61.2 当前动作与下一步安排

右栏固定顺序为当前采集任务、当前动作/目标、下一步安排、姿态示意、实时检查、数据分析及采集设置。固定边、移动边与目标图例持续保留。原有单行实时摘要不再与详细实时检查重复展示；离线预览仍独立显示当前角点来源和数量。累计板角点ID、视场网格与有效配对统计仅在数据分析中展开。

下一步安排是持久可见区域，而非短暂提示。它依赖现有状态，按以下优先级选择文案：

1. 回看时提示先结束回看；离线时提示检查导入数据后前往标定。
2. 保存失败时提示重试保留原图，不要求重复采图；白平衡准备时提示等待本次准备完成。
3. 保存或后台分析尚未完成时提示等待确认，不显示已经推进或已经采集完成。
4. 固定计划根据实际序列给出下一参考动作；末项提示进入标定评估或自主补拍，不宣称精度通过。
5. 自适应引导继续采用当前分析的后续安排。初始建模需手动采图、姿态模型非最终标定、复核/补采等原有语义保持不变；未分析完成时不承诺具体下一距离与角度。

实时检查分别展示双路板点、清晰度、稳定性及视频配对。软件视频配对保持中性色，只表示帧时效及时间差条件；绿色“预览条件满足”还必须通过既有完整自动采集就绪检查，并明确本次原图仍需单独验收。准备、保存、分析失败、离线和回看不会被显示为实时可采图。

### 61.3 标定、结果与融合页面

标定页将数据与模型、设置摘要、相机组就绪状态、校正检查和误差诊断按分隔线组织。结果页独立展示质量结论、关键指标与验证、风险、结果文件及导出。新增返回采集/标定入口与状态驱动的下一步提示，保持原按钮信号与异步计算流程；数据就绪、计算成功、质量结论和文件保存成功仍是四种不同状态。

RGB+IR/辅助工作台沿用相同的当前动作、目标、下一步、示意、实时检查和数据概况层级。板规格、两路检测、稳定性、软件配对、保存/验收/引导采用分开表达；17项参考动作是浏览序号，不替代实际采用计数。回看、手动浏览、后台检查、失败回执与完成阶段保留原有优先级。融合标定仍要求RGB左目内参及两路角点对应核对。

### 61.4 配置与导出窗口

- 相机/标定板配置使用可滚动内容及固定应用/取消栏，支持矮窗口。下一步入口前往标定板页；应用仍执行原有配置验证与主窗口影响确认，取消不提交本次配置。
- RGB参数窗口将目标与同步、可调参数分组，长设备名和目标地址换行，不挤出底部操作。明确关闭窗口不会撤销已经应用到设备的参数；未改变左右同步、刷新或应用行为。
- 导出窗口固定对象/格式选择、内容清单及底部保存位置入口，长错误按纯文本呈现。保持原格式兼容性验证、文件协议、失败说明及保存位置选择流程。

### 61.5 验证范围

新增`test_capture_workbench_redesign.py`、`test_calibration_workbench_redesign.py`、`test_aux_workbench_redesign.py`及`test_settings_redesign.py`，覆盖三尺寸布局、当前/下一步优先级、入口路由、回看不改原图、失败与过期状态、图库按需展开、配置应用/取消、导出与放大字体/长文本。既有布局回归对应更新新设计的控件归属及有界缩略图数量，不放宽验收、标定或保存条件。

实际Qt回放截图位于忽略跟踪的`data/diagnostics/workbench_redesign_20261101/`。`test3`原图只读，数据回放和受控状态仅验证软件呈现与交互，不代表真实设备连接或标定精度。最终实际测试结果在修改记录中追加；本机离屏检查不能替代Windows原生字体/DPI及最终EXE验收。本轮不自动提交、推送或触发打包。

## 62. 重构后的设计与流程复核（2026-11-01）

本轮把确认的设计稿与新捕获的Qt实际界面并列对照，检查设计还原、可读性、空状态、缩放、按钮含义及关键流程。结论是信息结构基本还原，但不作逐像素一致性承诺：原有验收策略、白平衡准备与取消、失败重试、撤销和导入角色交换仍保留，因此实际操作区与设计稿有必要差异。姿态线稿继续使用已验证的QPainter几何，不为复制生成图而改变坐标轴、固定边、移动边或距离定义。

### 62.1 发现并修复的问题

- 多组回执的证据关联错误：选择全部相机组时，原回执取第一组记录，回看入口却从拼接列表末尾找原图，可能把保存失败记录关联到另一组已保存图片。新增仅用于显示的`StereoCaptureWorkspace.receipt_key`，回执显示完整组名及编号，回看按同一key精确解析；记录不存在时禁用入口，不回退到其他组。
- 离线状态的显隐时序：导入结束后，原图回看按钮与来源提示曾依赖250ms定时同步。数据来源、采集目标、配对记录和回看状态变化现在立即刷新导航，定时器仅作兜底；测试停止定时器后仍验证导入/返回实时的即时更新。
- 保存回执与动作提示可能短暂矛盾：照片已经落盘，右栏仍显示正在保存。配对记录变化现在同时刷新动作、下一步、检查与回执；固定计划在保存、失败或准备期间显示等待事项并停止动作演示，保留所选参考目标，不改变后台序列推进规则。等待期间目标标为参考目标，不催促用户提前移动。
- 保存失败与验收待复核并存时，回执完整提示优先指向重试保存，不先要求重拍；原图保存确认前不宣称完成。
- 动画紧凑模式改为等宽板面/立体视角，提高右侧线稿可辨识度；下一步正文提高可读性，固定计划明确分行显示后续距离与姿态。预览条件结论增加边框强调，颜色和明确文字共同表达状态。
- 图库补充空状态，短窗口不撑开占位；“已采/导入图像”统一为中性的“配对图像”。回执按钮明确命名“回看原图”，避免把图像回看误解为完整验收详情。

### 62.2 保持不变的流程契约

双目回看暂停的是实时指引，不改变原有手动采集路径：点击采集仍获取实时新帧，并在真正开始采集时退出回看，不保存当前回看图像。文案对应改为“实时指引已暂停”，不新增双目回看禁采规则。RGB+IR/辅助回看仍按原有门禁拒绝采图，须结束回看后恢复。

不更改角点检测尺度、时间配对、清晰度/稳定性、自动采图门槛、部分角点验收、模型选择、标定、坐标变换、导出协议、白平衡控制或原图目录。原有保存队列、失败重试不重复检测/计数、撤销与过期结果拒绝等流程继续回归。显示状态刷新可以提前到事件到达时，但引导分析仍使用既有上下文、worker及去重机制。

### 62.3 实际流程证据与验证边界

在离屏实际Qt应用中回放`test3`原图，未连接真实相机、未发送设备参数请求，截图和受控会话位于忽略跟踪的`data/diagnostics/workbench_review_20261101/`：

1. `01-empty.png`及`02-replay-before-capture.png`：检查空状态、当前动作、后续安排、等宽示意与固定操作区。
2. `03-original-saved.png`：实际点击采图并经后台队列落盘，读取输出PNG逐像素确认等于输入原图。
3. `04-save-failure.png`、`05-review-failed-original.png`、`06-retry-saved-without-recapture.png`：注入写盘失败，回看对应失败原图，再经既有重试入口保存；确认没有重复采图或增加标定帧数。故障是测试注入，不是设备故障。
4. `07-offline-import.png`、`08-calibration-ready.png`及`09-no-result-no-export.png`：经真实导入入口读取图片目录，检查离线即时显隐、标定页导航，以及尚未计算时不允许导出。对原始数据文件校验哈希，确认未修改。

`tests/test_workbench_review.py`补齐多组失败回执、记录撤销后的无回退、无定时器显隐/保存回调、等待事项、空图库、等宽示意和双目回看仍采实时新帧的回归。继续运行已有应用、交互安全、融合、标定及导出回归与源码标定自检，实际结果在修改记录追加。截图和本机测试不证明真实硬件同步/标定精度、Windows原生字体/DPI、屏幕阅读器完整支持或最终EXE兼容性；本轮不提交或打包。

## 63. 标定检查与原图验收详情窗口统一（2026-11-02）

本轮补齐采集、标定、结果主流程中常用的三个检查窗口，不把公共配色继承视为所有界面已重构。继续使用PySide6和已有Lucide图标，不引入新依赖，不改标定算法、误差口径、原图验收、设备参数或保存流程。点云窗口不在本轮范围；增强预览指示框仍仅为设计预览，未接入运行界面。

### 63.1 逐帧误差检查

`CalibrationDiagnosticsDialog`采用紧凑标题和组选择、常驻指标口径、可展开说明、优先占用空间的数据表和底部操作栏。详细说明使用有界独立滚动区，旧JSON缺少ID或误差指标定义等信息不因收起而丢失；常驻声明训练集误差不是独立验证、缺失值不代表零误差。长组名保留完整数据和提示，不撑大窗口。

无结果或当前组无逐帧记录时显示明确空状态，不伪造数据行。底栏分别显示未选择帧、无匹配原图、原图可用；只有选中行存在可靠匹配时才能回看。取消选中后不沿用上次原图。数字排序、源帧ID映射、左右/双目参与记录、校正残差及`preview_requested`保持原有契约；窄窗口通过表格自身横向滚动访问全部列，不隐藏误差或参与状态列。

### 63.2 校正图检查

`RectificationViewer`统一标题、来源区、图标工具栏、独立图像画布和底部图像信息。长来源信息可选中、可滚动，并保留完整提示；图像区不被长路径或状态说明挤出。适应窗口与原始像素模式显示选中状态，缩放比例及校正拼接图像素尺寸独立展示。

没有校正图时显示空状态并禁用图像工具；清除图像同步清除来源、比例和旧像素内容。首次显示及窗口缩放后，适应模式等待画布尺寸更新再计算比例；原始像素和手动缩放模式不被窗口变化强制重置。所有操作仅缩放已有显示图，不重算校正矩阵、不写回原图；“左目在左，右目在右；参考线不代表匹配已通过”的语义继续保留。

### 63.3 双目与融合验收详情

两种采图验收详情由普通消息框改为复用的`CaptureDetailsDialog`，包含纯文本只读正文、完整复制入口与固定关闭栏。正文按窗口宽度换行，长记录可垂直滚动，保留路径、角点数量及全部未采用原因，避免依赖短回执或悬停提示才能定位问题。

详情是点击时的回执快照，窗口底部明确标为“验收记录快照”。后续采图不会悄悄替换正在阅读的内容；再次点击对应入口读取最新回执，双目/融合切换复用同一窗口并同步标题与正文。窗口非模态、不改变当前工作流、不阻断已有保存和采集；复制仅访问剪贴板，不触发重检、重拍、保存或重新标定。空内容明确为暂无原图验收记录并禁用复制。

### 63.4 验证与边界

`test_diagnostics_redesign.py`及`test_inspection_dialogs_redesign.py`覆盖360×300、800×560及校正/验收窗口1100×720，长组名、长来源、长验收记录、展开滚动、固定底栏、数值排序和准确源图路由、取消选中、缩放模式、清空旧图、复制完整内容和复用快照；既有双目/融合采图测试继续验证完整验收原因。

实际Qt截图位于忽略跟踪的`data/diagnostics/inspection_redesign_20261102/`。逐帧表格及验收记录为受控界面样例，校正图沿用已有离线实验产物；没有连接相机或发送设备参数，不作为新增标定精度证据。全量回归和源码标定自检结果在修改记录中追加，Windows原生字体/DPI与最终EXE仍需后续构建验收。本轮未提交、推送或打包。

## 64. 采集操作区与右侧引导布局细化（2026-11-02）

### 64.1 底部操作与设置分行

宽度至少900px的既有宽版底栏中，第一行集中采集、自动采集、撤销及跳过，“前往标定”保持右对齐。原图状态、验收策略和白平衡设置位于下方带分隔线的设置区，同一行垂直居中；连接/准备/保存失败重试与验收回执占用下一行的稳定预留空间。

取消宽版空导航行和导入行的额外间距，限制采集按钮最大宽度240px、验收下拉框340px及白平衡下拉框280px；白平衡复选框与选项紧邻，剩余空间留在选项末端，不摊到两者之间。控件高度不低于原生最小尺寸，准备与重试使用更紧凑的内边距而非强行压缩文字。原图状态宽版保留一行，验收回执跨越设置两列；完整路径及验收原因仍在提示和详情中可查。较窄的原侧栏保持纵向排列，不改变原有动作与显隐规则。

本机三种主窗口尺寸下底栏仍为137px，重点是对齐和信息分组，不以减少必要的失败/准备状态空间换取表面紧凑。白平衡开关、验收切换、保存/失败提示不改变主预览几何。离线来源、重试保留原图、取消准备和前往标定仍沿用原有入口及信号。

### 64.2 引导区减少重复与分散

右栏将重复的“当前采集任务”和“当前动作”合为一个“当前动作”标题，重新分析按钮移到该标题同行，保留隐藏时的占位，避免按钮出现挤动内容。不再让该图标撑高“下一步安排”的标题行。动作字号调整为16px并保留两行；目标/原因使用12px两行，下一步仍保留13px三行空间，长内容不因实时变化撑高界面。

相机与目标示意的展开入口改为无背景工具标题，与暂停、重播和展开图标组成同一行；保留等宽板面/立体视角和常驻固定边、移动边、目标图例。实时检查由四个单独竖排项目改为两列两行，分别为双路板点/清晰度、稳定性/视频配对，中间预留分组间隔。各项仍显示完整文字，颜色不替代含义；采集条件结论独立保留，视频配对正常仍不宣称原图验收通过。

没有修改动作调度、距离估计、实时门槛、原图验收或任何模型与设备参数。`StableTextLabel`的行数按区域固定，不使用随状态内容自动展开的高度；累计统计继续归属数据分析，短窗口独立滚动侧栏。

### 64.3 验证

`test_capture_workbench_redesign.py`新增1000×680、1280×800、1440×1024的设置对齐、最大宽度、白平衡选项邻接、原生最小尺寸、两列实时检查、标题与重试占位、状态变化不移动主预览断言，联合已有自适应、采集、离线、融合及布局回归。

`data/diagnostics/capture_layout_refinement_20261102/`保存当前代码生成的三尺寸实际Qt采集页、底栏及引导截图，使用`test3`图片只读回放，源文件哈希不变。截图不表示已连接真实设备，亦不代替Windows原生字体/DPI或EXE验证；测试结果追加到修改记录，本轮不提交或打包。

## 65. Windows构建28配置窗口溢出修复（2026-11-02）

构建`41394423763`（`v0.4.3-build.20261102.1`，提交`24a2a33`）在Windows源码回归中出现两项布局失败：`test_larger_font_configuration_fits_narrow_window`相机连接页横向滚动范围为4；`test_large_font_and_long_names_fit_short_windows[config]`融合相机页为22。其余1498项通过、8项跳过，尚未执行PyInstaller、最终EXE校验或发布。不是OpenCV缺失、求解失败或安装器异常；标签页下只有自动生成的源码归档，不能当作安装包已发布。

### 65.1 原因与修复

两处配置组合行此前要求所有标签、下拉框和按钮始终横排。下拉框按最小字符数预留宽度，叠加Windows字体/样式的标签及按钮最小尺寸后超过内容区，造成窄窗口的大字体横向溢出。本机原生字体较窄，普通测试没有重现这两个具体像素值，不能据此断言Windows也适配。

新增仅在配置窗口使用的`_ResponsiveRow`，按子控件实际最小宽度及间距决定横排或纵排。相机连接页将类型、左目、右目分别作为完整的标签/选择器组合；融合页将辅助流选择器、使用发现服务、发现/刷新作为独立项，空间不足时整体分行，变宽后恢复横排。行布局在显示、尺寸、字体及内部布局变化时重新计算，不改配置数据、发现筛选、自动配对或应用/取消信号。

没有隐藏水平滚动条、放宽零溢出断言、减少下拉框最小字符数或提高窗口最小宽度。窄窗口允许正常纵向滚动，操作底栏仍固定在页面滚动区域之外；不同时修改白平衡或设备端逻辑。

### 65.2 复现与验证

`test_configuration_responsive_rows.py`使用可用的Qt Windows/Fusion控件样式、大字体和较宽选择器最小字符数，在900×480、901×641及1800×900检查分行、恢复横排、全部控件边界、原生最小尺寸、零横向溢出及固定应用按钮；另验证窗口已打开后改变字体能正确往返重排。改动前后的配置值必须一致，保留原有两项失败测试原断言。

忽略跟踪的`data/diagnostics/windows_config_overflow_20261102/compare.py`同时加载失败标签源码和当前源码进行受控对照：本机加宽控件条件下旧相机/融合页分别横向溢出89px和49px，修复后均为0；普通条件下也均为0。该实验不伪称重现Windows的4px/22px原始数值。目录包含比较数据及实际Qt截图，未连接相机或修改源图。

配置及布局专项、全量应用回归、Ruff、编译和源码标定自检结果追加到修改记录。Qt Windows样式模拟不等于在Windows系统运行；仍以新标签构建中的Windows源码回归、源码/EXE自检、打包和附件验证为准，不能将本地通过解释为新安装包已可下载。

## 66. Windows构建29字体切换测试的跨平台修正（2026-11-02）

构建`41399773867`（`v0.4.3-build.20261102.2`，提交`2fcf1dd`）仍在源码回归阶段停止，结果为1项失败、1504项通过、8项跳过。第65节针对实际横向溢出的原有两项测试已通过；剩余失败为新增的`test_font_change_reflows_existing_open_configuration`：将选择器最小字符数增加到22后，测试仍要求900px窗口在初始及恢复字体时必为横排。Windows字体较宽，合法纵排也会触发这一错误断言，不能据此判断配置窗口再次溢出。

本轮只调整测试及文档，不修改应用运行逻辑。测试先在足够宽的窗口分别测量两种字体下各`QLayoutItem.minimumSize()`的横排需求，结合对应字体下表单标签和外层布局占宽，选择介于两种窗口宽度需求之间的尺寸，固定窗口验证初始横排、放大字体后纵排、恢复字体后再次横排。每个阶段先确认实测宽度位于对应阈值一侧，不假定字体变化前后的字段可用宽度相同，也不再把固定900px作为跨平台字体边界。

用例扩展为相机连接/融合相机两页、Qt Windows/Fusion样式、默认/较大两种初始字体的组合。保留固定900×480、901×641窄窗口和1800×900宽窗口往返检查；字体变化阶段保留原生控件最小尺寸、控件边界、当前可见页零横向溢出、固定底部按钮、窗口尺寸不变及配置值不变断言。隐藏标签页的布局由Qt延迟更新，检查在相应页实际显示时执行，两页均有独立覆盖；不删除失败用例、不跳过Windows、不提高生产窗口最小尺寸或隐藏滚动条。

本机配置专项49项通过。完整回归及新预览标签构建结果分别在修改记录追加；本地控件样式仍不能替代真实Windows运行和最终EXE验收。修复保留0.4.3产品版本，重新打包必须使用新的预览标签，不移动或重用构建28/29的标签。
