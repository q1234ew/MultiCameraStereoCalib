# Jetson 三设备服务框架（候选，已部署至 Orin）

## 当前能力（2026-11-05）

统一入口 `http://192.168.31.10:8092/` 已部署原始预览、双目校正、原版 FoundationStereo
CUDA 米制深度、各设备独立点云、分辨率/迭代档位和相机设置。三设备×四分辨率×三迭代档位
共36组实拍推理通过，本地和Orin各52项相关回归通过。没有跨设备外参或多点云融合。
下面的“未实现”描述属于早期框架记录；最新架构、模型身份、限额和测量边界以
[JETSON_ORIN_DEPLOYMENT.md](JETSON_ORIN_DEPLOYMENT.md) 文末为准。8091的原离线状态接口
保持兼容，8092新增实时按需计算与显式相机参数写入。

## 初始本地阶段记录（部署前）

这是一个受限首版：提供独立配置、CPU 校正缓存、原始帧配对原语、离线深度点云 worker、
只读状态接口及 systemd 模板。没有部署 Jetson，没有修改 RV1126B 的原始流或现有服务。
不声称已完成现场三机接入、物理精度验收、硬件同步、立体深度估计或跨设备融合。

2026-11-05 从本机执行了：

```sh
ssh -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=5 Nv '<read-only checks>'
```

结果为连接 `192.168.31.148:22` 超时，退出码 255，尚未进入身份认证。
这不是密码错误或业务服务故障；免密登录是否可用尚未知。没有猜测密码。
型号、JetPack/L4T、CUDA、OpenCV、Python、磁盘、运行服务、端口监听及三台相机端点
均未取得远端证据。SSH 配置的 RemoteForward 未启用。

本机实际读取了三个 `20261105_2/bundles/device_N` 的 manifest 和 calibration_mm.yml：
RGB 左右均为 1280×720，IR 为 256×192；IR 使用 fisheye，不能按普通针孔畸变处理。
配置内 SHA256 是这批候选文件的实际哈希，不代表现场已加载或验证。

## 身份与边界

- device_1：192.168.31.159，86:6b:17:d4:ec:32，calib_20261031_172015。
- device_2：192.168.31.244，aa:ce:a2:64:d6:41，calib_20261104_174237。
- device_3：192.168.31.55，92:6f:fb:29:99:c4，calib_20261102_141028。

原始 URL 保持 `/stream`：device_1/3 的候选 left=8080、right=8082，8081 为 IR。
device_2 特别沿用历史 left=8082、right=8080，**必须核对，不是物理验证结论**；
其他设备也默认 `physical_binding_verified=false`。IP/MAC 是期望绑定，不是探测到的身份。
仅手动把此布尔值改成 true 不会启用任何现场采集功能。

候选路径为各相机上的
`/opt/fusion-sensor/calibration-candidates/device_N/0.6.1-compat.20261105.2/calibration_mm.yml`。
此路径不是 Jetson 已存在文件的证据；本地/Jetson 读取复制的 bundles 时传 `--bundle-root`。
严禁把设备间的同名 camera_id 当成全局唯一标识，所有状态及缓存都隶属于 device_N。

RV1126B 8090 由另一项工作实现：`GET /health`、`GET /api/calibration`、
`GET /api/rectified.jpg?camera=left|right`。本框架不部署、不修改、不探测该服务，
也不假设两个独立 JPEG 请求是同步对。配置中的 8090 仅为预期服务地址。

## 所有权与生命周期

一个 Jetson 进程管理三份独立设备状态，不需要数据库、Qt、Open3D 或新的重依赖。
依赖仅 Python 3.10+、NumPy、OpenCV 和 aiohttp；不要为此在 Jetson 安装桌面应用整包。
不承诺 CUDA、GStreamer、JetPack ABI 或 PyPI OpenCV wheel 可用，后续先盘点再选环境。

1. 启动先验证配置，再按设备读取候选文件；读取同一份字节校验哈希并创建映射。
   检查 device_id、mm、C_L、左右/IR 变换方向、三路分辨率和相机模型。
   一台设备参数缺失/损坏不阻止其他设备初始化；状态记录该设备错误。
2. `Pairer` 每设备仅两个最新帧槽，按左右角色接收原始帧，不改输入图像。
   上游必须提供同一时钟域的纳秒时间戳；拒绝跨设备、尺寸变化、重复/倒序、未来及过期帧。
   默认容许 40 ms 左右偏差、500 ms 帧龄，超差丢旧帧；不等同于硬件曝光同步。
3. `Rectifier` 每设备创建一次左右模型感知映射并复用，缓存生命周期等于进程。
   参数更新必须重新初始化，不做文件热替换；禁止静默缩放图像来适配标定。
4. 离线 `DeviceWorker` 使用单线程 executor 处理一对帧，先校正，再可选生成点云。
   当前 CLI 一次一个回放任务，没有网络提交队列；未来实时任务每设备只允许一个在途任务，
   使用 latest-only mailbox，禁止向无界 executor 队列持续提交采集帧。
5. 深度仅接收与同一对 rectified-left 图像注册的真实轴向 Z 图，必须声明 m 或 mm、
   左帧时间戳、设备及参数哈希。无效/非正深度过滤，输出米，坐标系为 device_N:C_L_rect。
   不默认为 Z=1；显式深度平面可作为未来独立模式，但当前没有平面模式或估深器。
6. 各设备校正左坐标系互不相同；缺少实测跨设备外参时不合并点云。
   `T_rgb_left_aux` 为 IR/aux → 原始 RGB-left；从校正左系转回原始左系还需 R1 的逆旋转，
   不得直接拿 rectified 点套用 IR 外参。当前没有 IR 叠加投影路径。

## 运行与离线验收

从仓库根运行，使用已有且满足依赖的 Python 环境，不自动安装任何软件：

```sh
python -m scripts.jetson.service --check --bundle-root /path/to/bundles
python -m scripts.jetson.service --bundle-root /path/to/bundles
```

`--check` 所有设备候选文件可读取时退出 0，有任何失败时退出 1。
文件检查通过不代表可上线。服务默认只监听 `127.0.0.1:8091`，无写入或激活接口：

- `GET /health`：200 表示状态进程存活，明确 `framework_only` 和 `ready=false`。
- `GET /ready`：当前始终 503，因为现场 transport 尚未实现。
- `GET /api/devices/device_N`：单设备状态，包含期望身份、候选状态及验证警告。

不应将存活检查配置成现场就绪检查。状态 API 没有身份认证，因此不允许直接绑定外网；
通过 SSH 隧道访问，未来暴露网络必须先设计认证和访问控制。

回放格式是受信任本地 `.npz`（禁止 pickle），每文件一对原始图像：

- `device_id`、`calibration_sha256`：标量字符串，与设备配置完全匹配。
- `left`、`right`：原始分辨率 NumPy 图像数组，不能提前校正。
- `left_timestamp_ns`、`right_timestamp_ns`：同一时钟域整数纳秒标量。
- 可选 `depth`：与校正左图同尺寸的轴向 Z 数组；同时必须给出 `depth_unit=m|mm`、
  `depth_frame=rectified_left_axial_z`、`depth_timestamp_ns`（必须等于左帧时间戳）。

```sh
python -m scripts.jetson.service --bundle-root /path/to/bundles \
  --device device_2 --replay /path/to/trusted-pair.npz
python -m pytest -q tests/test_jetson_service_config.py
```

回放输出配对偏差、校正尺寸、候选哈希、深度状态及点数，不自动保存图像或点云。
无 depth 时只校正，不生成假深度。声明的深度注册关系由录制方负责，不能靠元数据证明精度。
单文件回放支持原语验收，不是持续流吞吐测试或硬件精度结论。

本轮本机验证：15 项针对性测试通过；Ruff 与 compileall 通过；上述三个真实候选 bundle
的 `--check` 退出 0。分别以每设备候选参数运行合成零图、1 ms 左右时间差、显式 1500 mm
深度图的回放，每台输出 57600 个抽样点（默认步长 4），单位 m，独立校正左坐标系。
合成输入仅验证软件链路，不是实际采集到的深度图或现场精度证据。

## 部署模板与下一阶段

`packaging/jetson/multicamera-jetson.service` 是待审批模板，**未复制、安装、启用或启动**。
假定未来手工放置代码与 venv 到 `/opt/multicamera-jetson`，bundles 放到其 `bundles` 目录，
运行用户 nvidia；部署前核对实际用户、路径、空闲端口、库兼容性及资源限额。
模板启用只读文件系统限制，不向 `/opt/fusion-sensor` 写入，不替换任何现有服务。

下一阶段顺序：恢复 Jetson SSH 连通性 → 只读完成环境/网络盘点 → 验证 MAC 和每个物理镜头
对应端点/分辨率 → 对接带严格超时、大小上限、重连退避的 MJPEG transport → 接收时间戳配对
（或真实采集时间戳）→ 小规模回放及负载测试 → 单机只读影子运行 → 经授权部署。
IR 独立状态与时间戳配对、8090 参数一致性检查、显式平面投影、StereoSGBM/其他真实深度来源、
跨设备外参、点云输出、加速与长期录制均属于后续工作，不隐藏为已完成能力。


## 2026-11-05 Orin 部署与实机验证

SSH 地址已更新为 192.168.31.10，主机 scs-orin-nx，aarch64，L4T R35.4.1。
框架部署于 `/home/nvidia/multicamera-jetson`，使用独立 Python 3.10.22 venv，
未安装桌面应用整包。用户级 `multicamera-jetson.service` 已启用并运行；nvidia 的
Linger=yes 已存在。系统级模板仍未安装。用户级模板见
`packaging/jetson/multicamera-jetson-user.service`。

Orin 端 15 项回归、三个候选参数检查、三设备合成回放、状态接口与受控重启恢复均通过。
从 Orin 获取六路 1280×720 RGB 实拍快照并完成 CPU 校正；单帧校正约 5.5–9.8 ms，
这是一次快照样本耗时，不代表持续吞吐。三台 8090 参数原始字节 SHA256 与候选文件一致，
邻居表 MAC 均与配置一致。修正 device_1/3 为 left=8080、right=8082，device_2
仍为 left=8082、right=8080，与现场 8090 source roles 一致，物理镜头身份仍未核实。

三台 IR 快照分别为 512×384、768×576、512×384，与标定 256×192 不匹配；未缩放或
进入几何校正。没有更改相机显示配置。框架进程 `/health`=200，`/ready`=503（预期），
仍没有实时 MJPEG transport、估深器或跨机点云融合。合成深度每台 57600 点只验证
数值链路，不属于实测深度或精度验收。实拍快照请求独立，不证明同步。

进程重启后 RSS 约 104 MiB；用户 cgroup 未提供 MemoryCurrent/TasksCurrent，模板资源
限额的实际执行未验证。现有 fusion-orin-rtsp/usb 服务保持 active。未重启整机；开机启用
配置已验证，断电重启恢复未测。

完整结果与操作说明见 [JETSON_ORIN_DEPLOYMENT.md](JETSON_ORIN_DEPLOYMENT.md)。
以上记录更新了早期“未部署”结论，早期记录保留用于追溯。


## 统一预览扩展

2026-11-05 新增独立只读局域网预览进程，地址 `http://192.168.31.10:8092/`。
九路原始快照通过固定上游代理显示；不改变 8091 的 framework_only/ready=false 语义，
不进行同步、估深或融合。具体部署和验证见 JETSON_ORIN_DEPLOYMENT.md 的统一预览章节。
