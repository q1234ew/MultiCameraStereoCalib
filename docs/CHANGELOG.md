# MultiCamera 修改记录

本文维护项目级功能变更、兼容性影响、验证情况和后续注意事项。凡是涉及用户可见流程、标定结果、导出格式、日志、打包、数据模型、会话结构或性能策略的提交，都应同步更新本文件。

本文件采用追加式维护：新增变更只能追加新条目或在最近一条未提交记录中补充细节，不得用新内容覆盖、删除或重写已有历史记录。

## 2026-10-21 - Windows 最终 EXE 自检与旧包缺接口原因实测

### 验证结果

- 源码提交 `a51f264` 的 Windows 构建 #12 成功，用时 4 分 42 秒；最终 EXE 而非仅源码环境完成 ChArUco 检测、内参、双目、校正和导出回读。
- 新 EXE 报告 `frozen=true`、OpenCV `4.13.0` / contrib `4.13.0.92`；源码与 EXE 的原生库哈希和接口清单一致。仿真双目 RMS 约 `0.00014035` 像素，恢复基线 `0.11999995` 米。
- 已对旧 Release `v0.1.0-build.20261020.2` 的 ZIP 校验并加载其原生 `cv2.pyd`：版本为 `5.0.0`，旧 `aruco.calibrateCameraCharuco` 不存在，通用标定接口存在。
- 旧原生库 SHA256 `ae833b9b5771f5932200bd3b8ad0f483f4fbec92cbdbd26d21d38fe524c60e66` 与官方 PyPI `5.0.0.93` Windows wheel 完全一致，确认是依赖跨主版本后的 API 差异，不是 OpenCV 漏打包或 PyInstaller 裁剪接口。
- 新 Release 包含约 93 MB 安装器、143 MB 便携 ZIP、源码与 EXE 自检报告、旧包审计报告及 SHA256 清单。此验证不包含用户原始采图、鱼眼数值流程或安装器实机安装。
- 构建：https://github.com/q1234ew/MultiCameraStereoCalib/actions/runs/39475861438
- 下载：https://github.com/q1234ew/MultiCameraStereoCalib/releases/tag/windows-build-39475861438

## 2026-10-21 - OpenCV 构建版本约束与最终 EXE 标定自检

### 修复与防回归

- 旧构建 #10 的依赖安装日志确认使用 `opencv-contrib-python 5.0.0.93`，与本地 `4.13.0.92` 不一致；原依赖仅有下限，允许跨主版本升级。项目限制 OpenCV `<5`，Windows 构建通过 constraints 固定已验证的 `4.13.0.92`。
- 新增无界面的 `--self-test-calibration --report <path>`：真实执行 ChArUco 检测、24 帧部分/乱序观测内参、双目、校正和 JSON/OpenCV/Kalibr 导出；检查已知内参、基线、误差及文件回读，不修改用户会话。
- Windows CI 和本地构建脚本在打包后运行最终 EXE，以超时、退出码、报告状态、包内加载路径、源码/EXE 原生库哈希及接口一致性作为发布门槛。Release 附带两份自检报告，失败诊断写入 Actions summary。
- 新增可选 `audit_release` 检查旧 ZIP 的实际 `cv2.pyd`，区分归档二进制与探针环境 metadata；应用启动日志补充 OpenCV 版本、路径和接口清单。

### 兼容性与验证边界

- GUI 正常启动入口与标定结果格式不变；旧 `calibrateCameraCharuco` 接口缺失仍允许自检通过，生产标定继续使用通用接口。
- 自检覆盖针孔主流程，不代表真实图片、鱼眼数值、网络流、点云或安装器已验证；Windows 最终产物结果以对应 CI 报告为准。
- 已通过 21 项本地测试，以及 compileall、Ruff、actionlint、`git diff --check`；包含缺接口、导入失败、发行包冲突、求解异常、包外加载、无标准输出和无 GUI 入口回归。Windows 最终 EXE 和旧包原生接口尚待 CI 实测。

## 2026-10-21 - ChArUco 兼容性修复版发布验证

### 验证结果

- 提交 `4797222`、标签 `v0.1.0-build.20261021.1` 的 Windows 构建 #11 已成功，新增的离屏标定回归步骤通过后才执行打包。
- Release 已包含轻量版 EXE 安装包（约 97.9 MB）、便携 ZIP（约 150 MB）和 SHA256 校验文件。
- 构建记录：https://github.com/q1234ew/MultiCameraStereoCalib/actions/runs/39468498306
- 下载入口：https://github.com/q1234ew/MultiCameraStereoCalib/releases/tag/v0.1.0-build.20261021.1
- Windows CI 验证使用仿真观测；用户实际的 24 对图片尚未在本机或 CI 中实测。

## 2026-10-21 - 修复 Windows ChArUco 内参标定接口兼容性

提交：随 `Use generic OpenCV calibration for ChArUco intrinsics` 修复提交维护。

### 修复

- 针孔内参改用 `cv2.calibrateCamera`，按每帧 ChArUco 角点 ID 从标定板取对应三维点，取消对部分 OpenCV 构建缺失的 `cv2.aruco.calibrateCameraCharuco` 接口的依赖。
- 保留部分角点检测、非连续/乱序角点 ID、标准/稳健标定 flags、逐帧误差和位姿输出；原图和板尺寸设置可直接复用。
- 内参求解日志新增 OpenCV 版本、计算接口、样本数和图像尺寸，方便定位平台差异。
- Windows CI 在 PyInstaller 打包前执行标定回归，覆盖缺少旧接口时的内参求解、重复标定与结果保存。

### 兼容性

- 不修改标定板尺寸单位、相机会话结构或 JSON/OpenCV/Kalibr 导出字段；鱼眼标定流程不变。
- 已发布的旧 EXE 不会自动更新，需要安装或解压包含本次修复的新 Release，再导入原图重新标定。

### 验证

- 已通过 `.venv/bin/python -m pytest -q tests`：12 项测试通过，含 24 帧部分/乱序角点仿真、旧接口缺失回归和旧接口数值对照。
- 已通过 compileall、Ruff、actionlint 和 `git diff --check`。实际用户的 24 对原图不在本机，尚未用该批图像实测。

## 2026-10-20 - Windows Release 分发上线验证

### 验证结果

- 新流程提交 `e152106`、标签 `v0.1.0-build.20261020.2` 的 Windows 构建 #10 已成功完成，Release 发布步骤通过。
- 已在 Release 页面确认轻量版 `MultiCamera_Setup_0.1.0_x64.exe`（约 94.7 MB）、`MultiCamera_portable_x64.zip`（约 147 MB）和 `SHA256SUMS.txt` 三个附件。
- Release 为预发布版本，包含标定结果自动落盘和补拍后重新标定修复；本轮验证了 Windows 构建、附件上传及发布，未在 Windows 实机运行安装包。
- 构建记录：https://github.com/q1234ew/MultiCameraStereoCalib/actions/runs/39396166218
- 下载入口：https://github.com/q1234ew/MultiCameraStereoCalib/releases/tag/v0.1.0-build.20261020.2

## 2026-10-20 - Windows 安装包改为 Release 分发

提交：随 `Distribute Windows builds through GitHub Releases` 打包流程提交维护。

### 调整

- Windows 工作流改用 GitHub CLI 将安装程序和便携 ZIP 上传到 Release，移除 Actions Artifact 上传，避免 Artifact 存储额度耗尽导致分发失败。
- 附带 SHA256 校验文件，发布说明记录源码 commit、轻量/点云变体和构建链接；新 Release 在附件上传成功后才发布。
- 版本标签构建沿用原标签，手动构建使用独立 `windows-build-<run_id>` 标签；预发布构建不设置为正式 Latest。
- 发布 job 使用内置 token 的 `contents: write` 权限，同 ref 串行发布，重跑可更新原 Release 的同名附件。

### 兼容性

- 保留轻量包默认值、可选 Open3D 和可选 Inno 安装包；应用代码和标定文件格式不变。
- 下载入口改为仓库 Releases 页面，而非 Actions 页面的 Artifacts 区域。
- 旧运行的重跑不会采用新工作流，需要从新提交触发构建。不可变 Release 发布后不可覆盖附件，应使用新标签。

### 验证

- 已通过 actionlint、YAML 解析、Release 上传配置检查和 `git diff --check`。
- 实际 Windows 构建与 Release 发布需由包含本次改动的新提交/新标签验证，不能重跑旧标签来验证新流程。

## 2026-10-20 - 标定结果自动落盘与补拍后重新标定

提交：随 `Fix calibration autosave and recalibration after capture` 修复提交维护。

### 修复

- 标定完成后若没有当前会话，自动创建会话并保存结果，避免只在内存生成结果而磁盘没有标定文件。
- 自动保存 `results/rig.json`、OpenCV 完整/标准 YAML 和可求解的 Kalibr 链；多组双目分别输出每组的标准 OpenCV 和 Kalibr 文件。
- 已完成的组可重新标定，“一键标定全部”不再跳过已有结果的就绪组；补拍或撤销后清除相关旧结果，保留样本并重新计算内参和外参。
- 质量评估显示、校正预览或某种导出格式失败，不再阻断其余结果的保存；保存失败显示具体原因，保留手动导出入口。
- 文件先写入同目录临时文件再替换，避免写入失败截断上次结果；状态栏和日志显示成功保存的结果目录和文件清单。

### 兼容性

- 保留原有 `results/rig.json` 路径和 JSON 结构；重新标定成功会更新当前会话的同名结果文件。
- 自动创建会话不会补存此前未保存的原始图片；完整采图数据仍需在采集前新建会话。
- 缺少组间外参时完整 Kalibr 链会明确报错，各组独立链仍可保存；失败格式的既有文件保留上一版内容，日志会标明未更新。
- 更新技术文档，说明输出文件、重复标定流程、失败行为与离屏回归步骤。

### 验证

- 新增 `tests/test_calibration_workflow.py`，覆盖真实 worker 的补拍重算、已有样本重跑、自动保存和异常分支。
- 已通过 `.venv/bin/python -m pytest -q`：9 项回归测试全部通过。
- 已通过 `.venv/bin/python -m compileall -q src scripts tests`、`.venv/bin/python -m ruff check src scripts tests` 和 `git diff --check`。
- 使用离屏 Qt 和仿真观测完成验证，未在 Windows 实机和在线相机上复测；本轮不涉及点云性能或相机硬件同步。

## 2026-07-07 - RGB 参数同步应用到左右路

提交：随 `Sync RGB settings across stereo cameras` 功能提交维护。

### 新增

- RGB 参数窗口会从当前配置中收集所有 RGB 左/右相机 URL，默认勾选“同步应用到已配置 RGB 左/右”。
- 点击应用时，同一组 RGB 参数会按目标 URL 逐路下发到 `/api/rgb/config`，用于保持左右路图像参数一致。
- 应用成功后，主窗口会按所有成功下发的 URL 清理实时帧缓存和预览缓存，避免后续采集保存混入参数变更前旧帧。

### 修复

- 加固 RGB 参数 URL 归一化，支持从 `/stream`、`/snapshot`、`/api/rgb/config` 以及带 query 的流地址回到同端口 API 根路径。

### 兼容性

- 不改变相机配置、采集图片、标定结果和导出文件结构。
- 未配置多路 RGB 相机时仍保持单目标应用行为。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 RGB 参数 URL 归一化最小检查。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-07 - RGB 参数应用后刷新标定预览

提交：随 `Refresh preview after RGB settings apply` 修复提交维护。

### 修复

- 修复 RGB 参数应用成功后，采集缓存已清理但标定预览 tile 仍显示旧 pixmap，导致界面看起来没有刷新的问题。
- RGB 参数应用成功并匹配到实时相机后，立即清空对应相机预览缓存、FPS、检测缓存和渲染节流时间戳，预览区显示“等待新画面”。

### 兼容性

- 不改变相机连接、采集保存、标定和导出数据结构。
- 仅影响 RGB 参数应用后的预览显示刷新。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-07 - 修复 RGB Settings 接口读取超时

提交：随 `Use aiohttp for RGB settings requests` 修复提交维护。

### 修复

- 修复 `http://192.168.31.55:8080/` 在浏览器和 `curl` 可访问，但应用内 RGB Settings 显示接口不可用的问题。
- RGB Settings HTTP 客户端由 `urllib` 切换为项目已有的 `aiohttp`，与 MJPEG 流连接栈保持一致，避免相机服务与 Python `http.client` 兼容性导致读取状态行超时。

### 兼容性

- 不新增依赖，`aiohttp` 已是项目运行依赖。
- 不改变 RGB 参数接口、采集、标定和导出数据结构。

### 验证

- 已通过目标相机 `GET /api/rgb/config` 读取检查。
- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-07 - RGB 参数应用后采集使用新帧

提交：随 `Discard stale frames after RGB settings apply` 修复提交维护。

### 修复

- 修复应用 RGB 参数后立即采集时，可能从 `StreamManager` 最新帧缓存中取到参数变更前旧图像的问题。
- RGB 参数 POST 成功后，对匹配 URL 的实时相机清空旧最新帧，并在约 `0.8s` 过渡期内忽略新到达的过渡帧。
- 状态栏和操作日志提示“已丢弃旧帧，等待新画面后再采集”。

### 兼容性

- 不改变相机连接、采集保存、标定和导出数据结构。
- 仅影响 RGB 参数应用后的短时间采集行为；正常实时预览和离线标定流程不受影响。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。
- 已通过 RGB Settings URL 匹配和 `discard_frames_for_url()` 最小检查。

## 2026-07-07 - 增加 RGB 参数设置功能

提交：随 `Add RGB settings dialog` 功能提交维护。

### 新增

- 新增“RGB 参数 / RGB Settings”对话框，通过 `GET/POST /api/rgb/config` 读取和应用 FusionSensor RGB 相机参数。
- 默认目标相机为 `http://192.168.31.55:8080/`；如果当前项目已配置 RGB 双目相机，则优先使用 RGB 左目 URL。
- 支持曝光模式、曝光时间、亮度、增益、白平衡、防闪烁、对比度、饱和度、伽马、锐度、色调、背光和动态帧率等接口返回的控制项。
- 菜单 `工具 -> RGB 参数` 和主工具栏增加入口。
- 参数读取/应用在后台线程中执行，避免阻塞 UI。

### 兼容性

- 不改变相机连接、采集、标定和导出数据结构。
- 仅在目标相机服务支持 `/api/rgb/config` 时可用；不支持时会提示接口不可用。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。
- 已通过 RGB Settings URL 规范化导入检查。

## 2026-07-01 - 修复连接相机时多余空状态界面

提交：随 `Avoid empty preview flash on connect` 修复提交维护。

### 修复

- 修复点击“连接相机”时，预览区先显示“尚未连接相机”空状态、再切换到“连接中”相机卡片的多余界面闪现问题。
- 连接流程改为直接刷新为已配置相机的“连接中”卡片，同时清空旧 FPS、检测、渲染和覆盖率缓存。

### 兼容性

- 不改变相机连接、采集、标定和导出数据结构。
- 停止相机或确实没有相机卡片时仍会显示“尚未连接相机”空状态。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-01 - 降低 Windows 视频预览流延迟

提交：随 `Reduce live preview latency` 性能优化提交维护。

### 优化

- MJPEG 读取端改为“最新帧优先”：当网络 buffer 中已经累积多张完整 JPEG 时，只解码最新一张，丢弃更旧完整帧，避免 Windows 端解码或 UI 短时卡顿后继续追旧帧。
- 增加 MJPEG buffer 上限保护，异常堆积时保留最近 SOI 后的数据，防止旧数据持续扩大延迟和内存占用。
- UI 预览端丢弃超过 `350ms` 的过期帧，避免 Qt 事件队列积压时把旧画面慢慢播放出来。

### 兼容性

- 不改变采集和标定使用的最新帧缓存逻辑。
- 不改变相机配置、导出格式和会话数据结构。
- 预览更偏向低延迟，极端卡顿时可能跳帧，但不会按旧帧顺序补播放。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-01 - 增强相机连接状态与失败提醒

提交：随 `Improve camera connection failure prompts` 功能提交维护。

### 新增

- 点击“连接相机”后，预览区会立即为所有已配置实时相机创建“连接中”占位卡片，不再等首帧到达才显示。
- 相机连接失败时，在预览卡片、状态栏和操作日志中展示相机 ID、视频流 URL 和失败原因。
- 每次连接尝试中，每路相机首次失败会弹出一次明确提醒，包含常见排查建议和日志文件路径。
- 操作日志记录开始连接的相机列表以及每路相机成功连接信息。

### 修复

- 修复相机从未成功连接时，预览区没有对应卡片、用户难以判断是哪一路失败的问题。

### 兼容性

- 不改变相机配置、采集、标定和导出格式。
- 失败弹窗和 UI 操作日志按“每次连接尝试每路相机一次”限频，避免 MJPEG 自动重连时刷屏。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-01 - 优化低分辨率传感器预览叠加层

提交：随 `Compact overlays for low-resolution previews` 修复提交维护。

### 修复

- 修复红外等低分辨率视频流在预览区被采图引导文字、右下示意窗和 OSD 大面积遮挡的问题。
- 当预览帧小于 `480x320` 时启用紧凑叠加层，只保留检测边框、目标区域和倾角轮廓，不再向原始小图写入大段中文标签。
- 低分辨率帧仍保留下方信息栏中的分辨率和 FPS，不影响连接、采集、检测和标定流程。

### 兼容性

- 不改变相机流、采集图片、标定输入和导出格式。
- RGB 高清预览仍使用完整采图引导文字和 OSD。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-07-01 - 清理历史相机残留预览画面

提交：随 `Clear stale camera previews` 修复提交维护。

### 修复

- 修复停止相机、重新配置相机或导入较少离线图片后，历史相机 tile 仍显示最后一帧的问题。
- 相机断开时清空 `CameraView` 的帧缓存、缩放 pixmap 缓存和预览 QLabel，避免残留旧画面。
- 启动实时流前、停止实时流后、配置变更完成后主动清空预览网格。
- 离线静态图片刷新时移除本次导入数据中不存在的旧相机 tile。
- `StreamManager` 停止、移除或重启相机时发出断开事件，并忽略停止后迟到的旧帧。

### 兼容性

- 不改变相机配置、采集、标定和导出数据结构。
- 断开或重新配置后预览区会回到空状态，不再保留旧相机最后一帧。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过 `QT_QPA_PLATFORM=offscreen .venv/bin/python -c "... MainWindow ..."` 启动检查。

## 2026-05-24 - 增加 Changelog 追加式维护约束

提交：随 `Require append-only changelog updates` 文档提交维护。

### 新增

- 明确 `docs/CHANGELOG.md` 采用追加式维护，历史记录不得被覆盖、删除或重写。
- 技术文档同步补充 Changelog 维护约束，要求提交前检查是否只追加了新记录。

### 兼容性

- 仅文档约束变更，不影响应用运行、标定结果和导出格式。

### 验证

- 文档变更，未运行编译或 UI 测试。

## 2026-05-24 - 标定导出、离线流程和质量策略增强

提交：`4e3dbb9 Improve calibration export and quality workflows`

### 新增

- 新增 OpenCV 标准 Stereo YAML 导出选项，输出 `M1/D1/M2/D2/R/T/E/F/R1/R2/P1/P2/Q`、`K1/K2`、`cameraMatrix*`、`distCoeffs*` 等第三方工具常用字段。
- 新增标定质量评估显示，包含单目标定 RMS、双目标定 RMS、质量等级、综合分数、关键指标和优化建议。
- 新增标定优化策略选项：标准标定、稳健标定、联合精修。
- 新增数据筛选策略选项：全部使用、自动质量筛选、角点丰富优先。
- 新增实时相机配置中的左右图像源交换功能。
- 新增离线导入后的左右图交换功能，支持修正离线左右图反向问题。
- 离线图片导入支持递归扫描 `left/right`、`*_left/*_right`、`left_*/right_*`、`*_L/*_R` 等目录命名。

### 修复

- 修复单组/离线标定完成后没有构建 `MultiCameraRig`，导致导出按钮不可用的问题。
- 修复配置窗口中相机模型字符串被重复包装导致的保存异常。
- 修复重复导入离线图片时旧虚拟离线相机组残留、与新数据混用的问题。
- 修复重置/重新导入数据后导出按钮仍可能保留旧结果状态的问题。

### 日志

- 导出流程记录目标路径、格式和失败原因。
- 标定 worker 记录策略、内参/双目标定开关、样本数量、RMS 和异常。
- 离线导入记录目录、匹配规则和导入结果。
- 左右图像源交换记录 pair 名称和左右 URL 变化。

### 文档

- 更新 `docs/TECHNICAL_DOCUMENTATION.md`，补充离线导入目录规范、左右图交换、OpenCV 标准导出、质量评估和数据筛选策略。

### 验证

- 已通过 `.venv/bin/python -m compileall -q src scripts`。
- 已通过 `.venv/bin/python -m ruff check src scripts`。
- 已通过离屏方式创建 `MainWindow` 的启动检查。
- 已用合成 rig 导出 OpenCV 标准 YAML，并通过 `cv2.FileStorage` 读取核心矩阵。

### 注意事项

- 如果应用已经打开，需要重启后才能看到新增按钮和导出格式。
- 已交换左右图后，应重新执行标定并重新导出文件。
- 数据筛选策略只影响本次标定计算，不会删除原始采集图片。

## 维护规则

- 每次实现用户可见功能、修复影响标定结果的问题、修改导出格式、改变会话数据结构、调整打包流程或新增日志能力时，都需要新增一条记录。
- 修改记录必须追加式维护。新增内容应插入为新的日期条目，或在尚未提交的最新条目中补充，不允许覆盖、删除、合并或重写已有历史条目。
- 每条记录至少包含变更摘要、兼容性影响、验证命令或未验证原因。
- 修改记录应与代码提交保持一致；如果一次提交包含多个主题，记录中应拆分为清晰的小节。
- 纯格式化、注释微调或不影响行为的内部整理可以不新增记录，但如果同时更新技术文档，应在对应记录中说明。
