# Orin 轻量部署程序 0.9

主包仅包含代码、安装器、资源索引、依赖版本配置及少量兼容补丁，不包含模型、权重、wheel或CUDA大包。构建限制20MiB，当前约250KB。支持已有环境复用（reuse）与全新隔离环境创建（provision）。

## 支持的系统

| 配置 | 系统 | 推理依赖 |
| --- | --- | --- |
| JetPack 5 | L4T R35.4.1 / Ubuntu20.04 | Python3.8、NVIDIA torch2.1、CUDA11.4、torchvision0.16.2 |
| JetPack 6 | L4T R36.4.3 / Ubuntu22.04 | Python3.10、Jetson torch2.8、CUDA12.6、torchvision0.23 |

`deployment_profile=auto`按L4T选择已知配置，未知版本明确拒绝，不退回JetPack5，也不混装不同CUDA主版本。本程序不刷机、不创建OEM用户、不升级L4T/内核/显卡驱动。先通过SDK Manager或设备厂商镜像准备相应JetPack系统和登录账号。

## 全新设备

1. 解压轻量包，将`site.fresh.example.json`复制到**包目录之外**，填写服务用户名、安装目录、IP/端口、相机配置和标定目录。也可用`site.fresh.jp6.example.json`明确指定JetPack6。相机发现/添加目前仍需配置文件，安装器不扫描局域网、不复制其它硬件的有效标定声明。
2. 只读查看计划，显示资源清单、已知下载体积及最低磁盘空间（12GiB）：

```sh
python3 installer.py plan --bundle . --config /path/site.json
```

3. 在新设备运行：

```sh
sudo ./install.sh --config /path/site.json --system-deps --activate
```

系统组件已准备完毕时可省略`--system-deps`。不加`--activate`只完成暂存和GPU环境验收，不改线上服务。首次需要下载依赖和模型，设备端的下载体积与主包体积是不同概念。

新装顺序：平台检查→可选系统依赖→Python→空venv及固定pip引导→CPU/GPU依赖→固定源码及补丁→模型资源→CUDA/native op与应用检查→原子激活服务。所有pip安装仅写入受管理venv，不使用system/user site-packages，不修改已有fs-venv、全局torch或系统Python。

JetPack5默认从Python.org固定源码构建私有Python3.10，限制1–4个构建任务，不替换系统Python3.8。可将`provision.python310`设为已有兼容Python3.10解释器，仍新建隔离venv。JetPack6直接使用系统Python3.10。GPU环境由姿态和深度两个独立进程共享一套依赖，减少重复安装大型torch。

JetPack5默认按固定源码编译torchvision CUDA扩展，可提供已验证wheel避免编译；JetPack6使用校验固定的Jetson CUDA12.6 wheel。不会从通用PyPI悄悄替换GPU版torch。CUDA及NMS实际运行通过才算GPU环境验收通过。

`--system-deps`会安装CPU编译/运行组件及指定CUDA主版本的组件；APT模拟计划如涉及L4T、驱动或错误CUDA主版本则拒绝。已有Broker不会被停止；仅本程序首次新装mosquitto后，为启用自有Broker而停止新产生的默认服务。已有服务/端口冲突时需要使用不同端口或external模式，安装器不会抢占。

## 资源大小与来源

| 资源 | 当前文件体积 | 是否进入主包 |
| --- | ---: | --- |
| FoundationStereo checkpoint | 3,298,527,334字节，约3.30GB/3.07GiB | 否 |
| YOLO26n-pose | 7,878,574字节，约7.88MB | 否 |
| JetPack5 torch wheel | 约170MB | 否 |
| JetPack6 torch wheel | 约226MB | 否 |

资源索引固定文件SHA256和大小（源码另有限制），先尺寸再流式哈希；下载未完成或哈希不匹配绝不启用。HTTPS支持Range续传，服务器忽略Range时重新开始。模型在资源库/缓存只保存一份，源码目录通过链接引用；重试和升级不会将权重复制到每个版本。

默认来源：NVIDIA JetPack5 wheel目录、Jetson AI Lab的CUDA12.6 wheel索引（PyTorch官方Jetson文档推荐）、Python.org、固定提交的NVlabs/FoundationStereo与PyTorch/vision、Ultralytics发布资产。FoundationStereo官方权重由Google Drive文件夹提供，安装器只选择23-51-11的目标checkpoint，不下载整个多模型目录。当前网络无法访问Google时，请配置可用HTTPS镜像或提前准备本地资源库；不要将网络故障当成模型/设备故障。

`provision.asset_sources`可按资源ID指定**HTTPS URL或本机绝对文件路径**，哈希和大小仍按索引验证，不能通过换镜像绕过身份检查。可选ID包括torch、torchvision-wheel、torchvision-source、python-source、pip、foundation-source、foundation、pose。配置本地文件时只引用已验证资源，不重复复制3GB模型。移动/删除外部资源会使依赖它的服务失效。

`provision.pip_index_url`可指定HTTPS包索引，`provision.pip_cache_dir`可复用用户下载缓存。下载缓存不等于旧Python环境，业务包仍安装到新venv。使用缓存和资源库时同样执行GPU及源码检查，不把复用文件误报成已经在空白系统上全流程验收。

首次下载的已知大资源约3.5GB，另有按需解析的pip和APT下载；主包仍仅约250KB。没有资源的完全离线新机无法凭一个小包生成模型，需要另行提供资源库。Fast-FoundationStereo和未验收TensorRT引擎不在默认安装路径，后续可作为独立组件。

来源说明：
- https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- https://docs.pytorch.org/TensorRT/getting_started/jetpack.html
- https://github.com/NVlabs/FoundationStereo/tree/6e8806816b533e4d13ddbb95ffa907b797060a62

## 已有设备复用

用site.example.json，填写已有四个解释器及模型路径，environment_mode保持reuse。不会执行pip或修改外部环境。升级只恢复代码和资源引用，无法撤回用户原地修改的外部依赖；建议版本化资源目录。

## 数据、升级和回退

- 代码在releases/<release_id>，current原子切换；现场数据在shared/data，每版本保留标定小文件快照。
- 新装环境在shared/provisioned/<配方摘要>，缓存与资源独立存放；兼容配置不同不共用同一环境。失败记录阶段和原因，完成标记允许同配方续装，不自动删除历史环境、权重或失败日志。
- 可选import_state_from只补缺失的追踪运行/地面/MQTT/融合外参4份JSON，已有设置和密码不覆盖。移动相机或新硬件不能继承旧几何核验结果。
- 拒绝不安全prefix、未经本安装器管理的非空目录、服务名冲突和首次激活端口冲突。失败恢复旧current、服务文件和启动状态；共享运行数据不做时间倒退，不支持破坏性数据库迁移。

```sh
./doctor.sh --config /path/site.json --gpu --live
sudo /path/new-bundle/upgrade.sh --config /path/site.json --activate
sudo ./rollback.sh --config /path/site.json
```

## MQTT与验收边界

managed模式生成独立Broker、随机发布/只读账号及ACL，默认仅监听127.0.0.1，关闭匿名，配置文件600。重复安装不轮换凭据。external复用已有Broker；disabled不启用MQTT。已有1883 Broker不会被抢占，可以另选端口。新Broker默认不自动配置TLS。

软件健康、相机在线、地面/时钟就绪和模型精度分别报告。软件安装不会放宽融合门限、自动建立外参或伪造世界坐标。环境探针包含真实CUDA/NMS与模型类导入，不能替代完整深度/姿态推理和现场精度验证；完整模型测试需要另行记录。PyTorch2.8路径显式使用已校验的完整checkpoint加载方式；可在反序列化前按预期SHA256拒绝被替换的权重。

## 构建与完整性

```sh
python3 scripts/package_jetson.py --output /tmp/orin-dist --build-id 20261113.1
```

输出小型tar.gz、sha256及逐文件manifest，记录产品版本、基础提交及workspace_dirty。只读计划和模块加载不修改包内文件；主包排除权重、wheel、运行时和引擎，超过20MiB拒绝构建。使用前按可信渠道得到的外层sha256核对；内部清单用于完整性检查，不是数字签名。不创建Git标签、不发布Release。
