# 相机采集守护（SysV / RK1126B）

`S90fusion-sensor` 为相机开机入口，`fusion-supervisor.sh` 持续监管原来的
`/opt/fusion-sensor/start_fusion.sh`。不改相机编号、左右绑定、标定参数或采集分辨率。

- 启动入口在后台等待 `eth0` 链路 `LOWER_UP` 及非链路本地的全局IPv4，不阻塞其他开机服务。
- 每2秒检查8080/8081/8082的 `/snapshot`，请求有2秒超时和3秒外部期限。
- 启动宽限25秒后，连续3次失败或启动器退出即清理整个采集进程组并重试。
- 重试间隔2、4、8、16、30秒封顶，持续健康60秒后回到2秒。
- `flock` 防止重复启动；PID核对脚本路径后才发停止信号，不按名称全局杀进程。
- 主动 `stop` 不会自动拉起；再次 `start` 才恢复监管。守护器自身被SIGKILL或设备断电后，
  需依赖重新执行启动入口或下次正常开机；它不是操作系统级硬件看门狗。

部署前备份旧启动入口、原始启动器和日志。**先用旧入口停止旧进程，再原子替换新入口**，
不能在旧采集仍运行时直接让新守护器接管。

安装位置及权限：

```text
packaging/jetson/camera/S90fusion-sensor -> /etc/init.d/S90fusion-sensor (0755)
packaging/jetson/camera/fusion-supervisor.sh -> /opt/fusion-sensor/fusion-supervisor.sh (0755)
```

使用 `/etc/init.d/S90fusion-sensor start|stop|restart|status`。
`status` 显示 `waiting_network / starting / healthy / degraded / backoff`；仅 `healthy`
返回0。日志追加到 `/var/log/fusion-sensor.log`，PID、锁、状态在 `/var/run/fusion-sensor.*`。
网络重试间隔默认5秒。可通过环境变量 `FUSION_NETWORK_INTERFACE` 选择网卡，
`FUSION_CHECK_INTERVAL`、`FUSION_STARTUP_GRACE`、`FUSION_MAX_FAILURES`、
`FUSION_HEALTH_PORTS` 调整检查策略；开机定制时应在启动入口明确设置并导出变量。

HTTP可用不等于真实图像有效。部署后还须从Orin解码三路JPEG核对尺寸、源online状态，
并核验真实追踪新帧。不能以“进程已启动”代替画面/人体精度验收。

验证入口：在Linux执行 `python scripts/tests/test_camera_supervisor.py`。
测试模拟网络延迟、进程组异常终止、单接口失效、重复启动、主动停止以及过期PID；
不修改网卡或访问真实相机。macOS缺少Linux工具时跳过，在Orin执行实测。

回退：使用**新入口**执行stop，确认相关采集进程已退出，再从备份恢复旧的
`/etc/init.d/S90fusion-sensor` 并start。保留原 `start_fusion.sh`，不恢复或覆盖标定文件。

## RK1126B 源帧时间戳透传

以设备实际二进制为基线。当前已确认原程序 SHA256
`24b7063a69b77640865a6e3a6e611ba19d2eb9c840ce8da749e054eedf563803`
与独立 RK1126B 工程部署包 `fusion-ustreamer-rk1126b-frame-sync-20261022.tar.gz`
一致；从该工程内 `FusionSensor` 仓库提交
`42fb525e73b44221a65f78a41f5f52edee4865dd` 按原工具链参数重建，完整文件哈希也一致。
这项逐字节核验是使用该提交的依据，不能只用分支名或 `--version` 的 `4.9` 判定。

`build_capture_clock.py` 从已核实提交导出隔离副本，先重建未修改程序并检查上述哈希，
再应用 `ustreamer-clock.patch`，全量清理、编译和生成清单。不同基线必须另行核实；
不会自动升级为 RK1126B 工作区最新源码，也不写入正在进行红外验证的工作区。

```bash
docker build --platform linux/amd64 -f packaging/jetson/camera/clock-build.Dockerfile \
  -t multicamera-clock-build:20261109 packaging/jetson/camera
python packaging/jetson/camera/build_capture_clock.py \
  --rk-root /path/to/RK1126B --output output/deployment/capture-clock-new
```

仅六个源文件改变：帧结构与复制、V4L2 取帧及 HTTP 快照输出。
不修改红外处理算法、USB 采集、控制参数、图像、端口、左右绑定或共享内存协议。
`/snapshot` 在当前在线帧有 V4L2 元数据时附加：

| 字段 | 含义 |
| --- | --- |
| `X-Capture-Protocol` | `v4l2-monotonic-v1` |
| `X-Capture-Build` | 独立补丁修订；有效同域版本为 `mc-clock-2-42fb525e` |
| `X-Capture-Boot-Id` | 相机内核启动 UUID |
| `X-Capture-Stream-Ns` | 此次 STREAMON 的 CLOCK_MONOTONIC 时间 |
| `X-Capture-Monotonic-Ns` | 原始 V4L2 缓冲区时间，整数纳秒（驱动实际精度可能更低） |
| `X-Capture-Sequence` | 原始 V4L2 帧序号；0 有效，允许 uint32 回绕 |
| `X-Capture-V4L2-Flags` | 原始时钟、阶段、错误等标志；保留原值 |
| `X-Capture-Clock` / `X-Capture-Stage` | 从驱动标志解码，无法确认时为 unknown |
| `X-Capture-Send-Monotonic-Ns` | 生成本次快照回复时的 CLOCK_MONOTONIC 时间 |

旧 `X-UStreamer-*-Timestamp` 使用 CLOCK_MONOTONIC_RAW，保持原样；
**不能直接与新驱动时间相减**。用于映射的发送时间必须取新增同域字段。
离线画面和 USB 红外帧不生成虚构的 V4L2 时间。重复快照保留原帧序号及时间。
透传成功不表示已测得真实曝光误差界限，不能据此直接打开跨机统一 ID 合并。

原生测试 `tests/capture_clock.c` 在相机执行（使用相机动态库，不能直接在较旧 glibc 的
Orin 上运行交叉编译的相机程序）。Orin 用 `scripts/jetson/capture_clock_audit.py`
解码验证各路快照；`trace_capture_clock.sh` 用独立 ftrace instance 核对驱动时间与帧号，
退出后清理，不修改全局 tracing 或相机采集参数。

部署需先上传程序和原生测试到新的 `/opt/fusion-sensor/backups/<id>/`，
运行 `deploy_capture_clock.sh` 时明确设置 `CAPTURE_CLOCK_BACKUP`、
`CAPTURE_CLOCK_NEW_SHA`，并按现场基线设置 `CAPTURE_CLOCK_EXPECTED_SHA`。
脚本核验旧新哈希、备份原程序和各路进程标识，原子替换程序，仅重启 RGB 进程；
逐通道守护器接管的 RGB 由其重启，原先无人监管的 RGB 按原 argv 重启。
红外进程 PID、启动 token、可执行文件哈希及启动配置须保持不变，且需另行连续采样红外画面。
不要执行整机采集服务重启。设备 1 独立红外验证目录和程序不得替换。

回退只恢复本次备份的原程序，再以相同方式逐路重启 RGB；不要停止红外或重启整个服务。
