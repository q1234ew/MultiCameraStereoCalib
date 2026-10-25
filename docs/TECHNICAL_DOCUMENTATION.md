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
  results/rig.json
  results/rig_opencv.yml
  results/rig_opencv_standard.yml
  results/rig_kalibr.yaml
```

双目图片保存由后台队列串行执行，避免多个线程同时写 `session.json`。保存失败不会推进 metadata，并会在 UI 日志中提示。辅助配对图片目前仍在采集槽中同步保存。

标定完成时，主窗口自动保存结果；如果没有当前会话，会先自动创建 `calib_<时间戳>` 会话。单组、全部标定和多模态标定均走这条保存路径，不需要再执行联合优化或生成点云。状态栏和操作日志显示实际结果目录，文件日志逐条记录成功写入的绝对路径。

- `rig.json`：完整 rig，可重新读取。
- `rig_opencv.yml`：OpenCV 完整 YAML。
- `rig_opencv_standard.yml`：单组双目的 OpenCV 标准文件。
- `rig_kalibr.yaml`：所有相机之间存在所需变换时，保存完整 Kalibr 链。
- 多组双目分别生成 `rig_pair_01_opencv_standard.yml`、`rig_pair_01_kalibr.yaml` 等文件，序号按 rig 中相机组顺序排列；OpenCV 文件内记录 `pair_name` 和左右相机 ID。缺少组间外参导致完整 Kalibr 链失败时，各组文件仍独立生成，并在操作日志提示具体失败原因。

每个文件先在结果目录内的临时目录生成，完成后替换同名文件。JSON 保存失败会弹窗提醒，并保留内存结果供手动导出；某个 YAML 格式失败不阻断其他格式。失败文件若曾存在，将保留上一版内容，此时不能将它视为最新结果，应以本次成功文件清单和失败日志为准。

自动创建会话只保证标定结果落盘，不会补存此前在无会话状态下采集的原始图片。需要保存完整原始数据时，仍应在第一次采集前新建会话。

### 3.1 配置变更与已有工作保护

配置对话框先构建候选配置，完成 ID、相机组名和地址校验，再由主窗口计算影响；确认之前不会停止原视频流或修改相机配置。取消确认不会改变连接、缓存或结果。自动采集、图片导入、后台保存或标定尚未结束时，暂不允许打开配置修改，避免工作线程使用变更中的数据。

- 无实质修改：保留采图、标定结果、预览及视频连接，不重置、不重连。相机角色/分组的内部规范化不算来源改变。
- 仅修改辅助 ID、URL、模态或板参数：清理辅助配对缓存、辅助内参和跨模态结果，保留 RGB 内参、双目结果及双目采图。
- 修改某组左右目来源、类型或删除该组：清理该组的检测缓存和结果，保留其他独立组；共享相机的组按依赖关联失效。若涉及 RGB 参考相机，还会清理依赖它的辅助数据和结果。
- 修改全局 ChArUco 板或镜头模型：按全局几何设置变更处理，清理全部相关采图缓存和结果，需重新采集或导入后计算。

只在受影响范围内确有缓存/标定结果时弹出确认，逐项列明失效对象；默认不应用。所有上述清理均针对内存，不删除磁盘历史图像或文件。后续标定成功仍会按原规则更新会话 `results/` 的同名文件；需保留每次结果时应另行导出。

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
- 使用 Windows runner 预装的 GitHub CLI，将 `MultiCamera_portable_x64.zip`、可选的 `MultiCamera_Setup_*_x64.exe` 和 `SHA256SUMS.txt` 上传到 GitHub Release 附件，不再调用 `actions/upload-artifact`，不占用 Actions Artifact 存储额度。
- 手动勾选 `pointcloud` 时才把 Open3D 纳入 CI 包。
- PyInstaller 打包前安装测试依赖并以离屏 Qt 执行 `python -m pytest -q tests`；测试失败会阻止构建和发布。
- 项目依赖限制 OpenCV 为 `>=4.9,<5`，Windows CI 和本地构建脚本另使用 `packaging/windows/constraints.txt` 固定为已验证的 `opencv-contrib-python==4.13.0.92`。升级该约束时必须重新通过源码和 EXE 自检；不要混装四种共享 `cv2` 命名空间的 OpenCV 发行包。
- 构建前生成 `calibration_source_test.json`，PyInstaller 完成后执行真实 `MultiCamera.exe --self-test-calibration`，生成 `calibration_self_test.json`。后者使用临时工作目录、不依赖相机，不创建 Qt 界面或加载 Open3D。
- 两份报告必须均通过，且 OpenCV 版本、原生库 SHA256、API 清单及 contrib 发行版本一致；EXE 还必须从 `_MEIPASS` 包内加载 OpenCV。缺报告、非零退出、120 秒超时、接口或数值检查失败均阻止压缩和发布。
- 自检报告随 Release 附件发布并纳入 SHA256 清单；失败时尽可能写入报告和 Actions job summary，不依赖 windowed EXE 的标准输出。若进程在 Python 入口前因 DLL 缺失而退出，则无 JSON，CI 仍按退出码/缺报告判失败。
- 手动构建可填写 `audit_release`（例如 `v0.1.0-build.20261020.2`），下载旧 ZIP、核对其校验清单，并在独立 Windows Python 进程加载旧包中的 `cv2.pyd`，输出 `previous_opencv_audit.json`。此报告检查旧原生绑定，不代表直接运行了旧 GUI EXE；探针环境的 Python 包 metadata 与归档二进制版本分开记录。
- 发布 job 使用仓库自带的 `GITHUB_TOKEN` 和 `contents: write` 权限，不需要新增 PAT。相同 ref 的构建串行运行，避免同时修改同一个 Release。
- tag 构建以当前标签创建 Release；手动构建自动使用 `windows-build-<run_id>` 标签，指向本次构建的精确 commit。带 `-` 的标签和手动构建标记为预发布，不覆盖正式 Latest。
- 新 Release 先创建草稿，附件上传成功后再发布。重跑复用原标签并更新同名附件；启用不可变 Release 的仓库应使用新标签重新发布。
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
- 多模态步骤有独立采图引导，覆盖视场九宫格和中心倾角；“采集多模态一帧”会取 `RGB_L` 与 AUX 最新帧、检测平面板并写入标定缓存。
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
