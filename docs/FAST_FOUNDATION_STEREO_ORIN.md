# Fast-FoundationStereo Orin 测试部署

2026-11-07 已部署到 `scs-orin-nx`，当前可访问地址 `192.168.31.148`。
此为独立、按需执行的测试入口，执行完退出，不设常驻模型服务。
统一网页当前深度模型仍为原 FoundationStereo；人体追踪、MQTT 和预览服务保持运行。

## 安装位置与身份

- 项目：`/home/nvidia/multicamera-jetson`
- 环境：`.fast-stereo-venv`，Python 3.8.10，继承既有 NVIDIA torch/CUDA/TensorRT。
- 源码：`models/fast-foundation-stereo_476f4249561f/`
- 上游：<https://github.com/NVlabs/Fast-FoundationStereo>
- 固定提交：`476f4249561f7c79ca707326954f9255643412a6`；源码归档来自该提交，未修改上游文件。
- 启动器：`packaging/jetson/run-fast-foundation-stereo.sh`
- 测试实现：`scripts/jetson/fast_foundation_smoke.py`
- 完整版本、文件SHA256和实测报告：`deployment/fast-foundation-stereo_20261107/manifest.json`。

复用 Orin 原目录中的研究权重，复制后保存SHA256，未覆盖原目录；上游未提供对应校验值，
本次不声称完成与上游发布摘要的独立比对。

| 权重 | SHA256 |
| --- | --- |
| `weights/20-30-48/model_best_bp2_serialize.pth` | `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692` |
| `weights/20-26-39/model_best_bp2_serialize.pth` | `d147587849b3482dd2921eef60dc2cd9e12690735612e91ee2782a92f8d1c59e` |

兼容环境为 NVIDIA torch `2.1.0a0+41361538.nv23.06`、CUDA 11.4、TensorRT 8.5.2.2、
独立安装的 torchvision `0.16.2+mc.nv23.6`、OpenCV 4.10.0.84、psutil 7.2.2；
其余现有依赖版本记于 manifest。未安装官方推荐的桌面 CUDA 12.4 / torch 2.6，
未替换系统或姿态环境依赖。由于 JetPack 5 缺少受支持的 Triton 编译链，
测试进程在导入上游前禁用 `torch.compile`，使用上游 eager PyTorch 相关体积计算。
TensorRT 路径也使用该实现连接两段引擎，不能视作完整 TensorRT 加速。
离线模式开启，测试时不下载模型。

## 后续执行

登录 Orin 后运行官方样例（默认 `20-30-48`、640×352、8次迭代）：

```bash
cd /home/nvidia/multicamera-jetson
bash packaging/jetson/run-fast-foundation-stereo.sh \
  --output data/fast-stereo-tests/demo-i8
```

快速4次迭代：

```bash
bash packaging/jetson/run-fast-foundation-stereo.sh --iterations 4 \
  --output data/fast-stereo-tests/demo-i4
```

另一组权重：

```bash
bash packaging/jetson/run-fast-foundation-stereo.sh \
  --checkpoint models/fast-foundation-stereo_476f4249561f/weights/20-26-39/model_best_bp2_serialize.pth \
  --output data/fast-stereo-tests/demo-20-26-39
```

测试自己的左右图：

```bash
bash packaging/jetson/run-fast-foundation-stereo.sh \
  --left /absolute/path/left_rectified.png \
  --right /absolute/path/right_rectified.png \
  --intrinsics /absolute/path/K.txt \
  --width 640 --height 352 --iterations 8 \
  --output data/fast-stereo-tests/camera-pair-01
```

必须先按相机标定校正、去畸变并确定左右顺序；这个入口不采集或自动校正原始相机画面。
尺寸参数是实际网络输入尺寸，脚本会缩放输入并同步缩放内参，补偿像素中心偏移。
支持32的整数倍尺寸，宽32–1920、高32–1088；这是入口限制，不代表所有尺寸已经实测。
上游建议宽度小于1000，优先使用接近原图宽高比的尺寸。

`K.txt` 第一行为对应输入文件分辨率下的左目3×3内参（9个数），第二行为米制正基线。
仅适用于左右校正主点相同的情形，即 `cx_left == cx_right`；其它投影几何不能直接使用
`depth = fx * baseline / disparity`。标定原始参数不得直接冒充校正内参。
自定义左右图时不自动使用样例内参；省略 `--intrinsics` 则只输出视差。
深度保存为米，非正视差或右目对应位置越界输出NaN；尚未增加业务距离范围过滤或精度校验。

每次输出 `disparity_px.npy`、`disparity_preview.png`（左/右/视差）、`report.json`，
提供有效内参时另有 `depth_m.npy`。请每次选择新的输出目录，以保留不同测试结果。
默认预热2次、计时5次，可用 `--warmup`、`--runs` 调整。不是流媒体采集或点云展示入口。

旧 TensorRT 引擎已复制至 `engines/640x352-i8-d192`，只作可选对照：

```bash
bash packaging/jetson/run-fast-foundation-stereo.sh --backend trt \
  --engine-dir models/fast-foundation-stereo_476f4249561f/engines/640x352-i8-d192 \
  --output data/fast-stereo-tests/legacy-trt
```

这组引擎已有导出来源无法核实，不能确认属于上述哪组权重；不作为默认后端或精度结论。
640×352、8次迭代、最大视差192是构建时固定设置，入口拒绝不匹配参数。
若要进行严格同权重比较，应从已记录权重重新导出并在本机重建引擎。

## 本机样例验证

同一官方960×540样例缩放到640×352，最大视差192，预热2次、随后5次CUDA同步计时：

| 路径 | 迭代 | 热推理中位 | 进程RSS |
| --- | ---: | ---: | ---: |
| PyTorch `20-30-48` | 8 | 440.78 ms | 3275 MiB |
| PyTorch `20-30-48` | 4 | 387.46 ms | 3276 MiB |
| PyTorch `20-26-39` | 8 | 530.15 ms | 3279 MiB |
| 原有 TensorRT 两段引擎＋eager相关体积 | 8 | 378.57 ms | 3074 MiB |

四组均通过有限且非空视差检查，保存了深度结果，并目视核对了样例视差图。
仅统计GPU同步的热前向过程，不包含缩放、主机到设备传输、深度换算、写盘、网络或多相机排队。
CUDA allocator峰值不含TensorRT自身分配；进程、CUDA与整机内存重叠，不可相加。
该结果不是完整管线帧率，未调整功耗模式或锁定频率，不应直接与官网3090结果或历史不同分辨率比较。
现场相机仍不可达，未做真实深度精度、设备2弯曲问题或跨机融合验证。

本地报告副本在 `data/deployment/fast-foundation-stereo_20261107/`（忽略目录），
Orin样例报告在 `deployment/fast-foundation-stereo_20261107/`。
该研究仓库的许可证及model card随源码保留；商业路径参考上游列出的商业权重与TAO实现，
本次部署用于研究测试。


## 2026-11-07 高分辨率资源实测

同一Orin、同一20-30-48权重、PyTorch FP16、eager相关体积、最大视差192，
每档独立新进程串行运行，预热2次后计时5次。测试前整机已用约3.55GiB；
预览、人体追踪与MQTT服务保持运行，但现场相机不可达，未叠加真实YOLO推理或三路深度并发。
输入为同一官方960×540图像缩放，仅验证资源规模，不增加图像细节或证明高分辨率精度。
1920×1088是满足32倍数约束的全高清附近网络尺寸，本次直接缩放到该尺寸；并非原生1080p采集。
尺寸校验上限已扩展到1920×1088，与本次实测范围一致。

| 网络输入 | 迭代 | 热前向中位 | CUDA已分配峰值 | CUDA保留峰值 | 整机已用峰值 | 整机最低可用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×352 | 8 | 442.97 ms | 0.71 GiB | 0.76 GiB | 5.36 GiB | 9.80 GiB |
| 960×544 | 8 | 1024.78 ms | 1.53 GiB | 1.71 GiB | 6.33 GiB | 8.82 GiB |
| 1280×736 | 8 | 1851.69 ms | 2.68 GiB | 2.90 GiB | 7.55 GiB | 7.61 GiB |
| 1920×1088 | 8 | 4054.79 ms | 5.84 GiB | 6.32 GiB | 11.03 GiB | 4.13 GiB |
| 1280×736 | 4 | 1607.06 ms | 2.68 GiB | 2.91 GiB | 7.50 GiB | 7.65 GiB |

上述CUDA数值来自PyTorch allocator；整机数值来自每100ms的psutil采样，同时保存tegrastats原始日志。
RSS峰值约3.20–3.27GiB，对统一内存GPU分配的覆盖与整机统计不同，不能用RSS推算完整模型占用。
进程、CUDA与整机内存重叠，表中列不可相加。整机最低可用使用MemAvailable口径，
包含可回收缓存，与总内存减used不是同一指标。
全部5档通过、无OOM、无交换空间使用；GPU利用率采样峰值均99%，GPU温度最高48.19°C。
利用率峰值不是全程平均，也不能据此诊断瓶颈。nvpmodel查询显示MAXN/0，但EMC上限读取权限不足，
没有切换功耗模式或锁频。测试时设置用户服务MemoryMax=9GiB、MemorySwapMax=0及运行时限，
监视器在整机可用低于2GiB、进程RSS超过8GiB或单档超过300秒时停止；本次均未触发。

1280档4次迭代比8次前向耗时减少约13.2%，CUDA峰值基本相同，表明该设置下减少迭代
不能显著降低峰值内存。1920档单次前向约4.05秒，不能据此称为多路实时深度。
后续现场测试可先从960档开始；1280适合低频更新，全高清更适合按需采集与离线比较。
这些判断仅针对当前PyTorch兼容路径，并非新的TensorRT引擎结果。

完整报告与原始采样：Orin `deployment/fast-foundation-highres_20261107/`；
本地副本 `data/deployment/fast-foundation-highres_20261107/`。文件身份保存于独立manifest，
保留此前部署和基准的历史manifest。测试完成后原有三个服务均active，内存回落且swap仍为0。

## TensorRT 单引擎构建与运行入口

`fast_foundation_export.py` 从同一已记录权重调用上游单ONNX包装器及相关体积实现，
使用opset16适配TensorRT8.5；以随机张量检查两种归一化分支及边界位置的体积算子等价性。
高分辨率可使用 `--half-trace` 降低导出峰值内存，原始权重文件不变。

```bash
cd /home/nvidia/multicamera-jetson
.fast-stereo-venv/bin/python scripts/jetson/fast_foundation_export.py \
  --root models/fast-foundation-stereo_476f4249561f \
  --checkpoint models/fast-foundation-stereo_476f4249561f/weights/20-30-48/model_best_bp2_serialize.pth \
  --width 1280 --height 736 --iterations 8 --half-trace \
  --output deployment/fast-foundation-trt-custom/1280x736-i8
.fast-stereo-venv/bin/python scripts/jetson/fast_foundation_build_trt.py \
  --directory deployment/fast-foundation-trt-custom/1280x736-i8
bash packaging/jetson/run-fast-foundation-stereo.sh \
  --backend trt-single --width 1280 --height 736 --iterations 8 \
  --engine-dir deployment/fast-foundation-trt-custom/1280x736-i8 \
  --output data/fast-stereo-tests/trt-1280-example
```

构建默认FP16，关闭TF32，保留Softmax和Reduce层FP32并强制精度约束，工作区上限2048MiB。
使用TensorRT8.5的内核选择启发式及core tactic预览功能；可用 `--exhaustive-tactics`
改为完整搜索，首次构建可能很慢。构建工作区上限不是运行时总内存上限。
同一父目录顺序构建时保存和复用 `timing.cache`，不得并发写同一缓存。
每份引擎保存 `export.json`、`model.yaml`、`build.json`，记录权重、ONNX、引擎摘要及构建参数。
入口拒绝已有ONNX/引擎覆盖，重建时请使用新目录。

`trt-single` 使用固定形状、预分配输入/输出CUDA缓冲区，图像归一化和GPU输入拷贝计入前向耗时。
尺寸、迭代、最大视差及配置必须与引擎构建一致；加载前验证引擎摘要及配置身份。
此入口独立于旧的 `trt` 两段引擎＋eager体积路径，不依赖旧引擎的未核验来源。
PyTorch allocator统计不覆盖TensorRT自有权重和工作区；报告另记context设备内存大小，
内存收益以独立采样的整机峰值为主。前向计时仍不含图像读取、缩放、上传和结果写盘。


## 2026-11-07：断开Host后的后台接管（构建/测试进行中）

按用户要求，将后续工作交给Orin用户服务 `multicamera-fast-stereo-benchmark.service`。
已确认 `Linger=yes`，服务不依赖Codex、Host或启动时的SSH连接；已关闭原始SSH启动连接，
重新连接确认服务active、原1280构建进程仍在推进，现有预览/追踪/MQTT也均active。
Orin需保持开机；此为当前开机周期内的临时服务，不宣称断电重启后自动续跑。

所有运行依赖、同权重ONNX、全高清固定形状计算图、参考结果、控制脚本及测试脚本均在Orin。
全高清直接追踪曾触及2GiB可用内存预留线而自动停止，随后以640×352动态导出，
在Host离线固定为1920×1088并优化到2348节点，已上传且摘要校验通过。
Host上的图处理工作已经结束；剩余编译、数值核验和配对性能测试均由Orin执行。
本机图工具的ONNX C++符号冲突通过独立序列化worker处理，未修改Orin的既有CUDA软件。

`scripts/jetson/fast_foundation_job.py` 接管现有构建，替换SSH进程的超时监控；
若原进程因断连退出，可从本机不可变ONNX接续构建。模型数值未通过门限时记录失败，
不把未通过结果计为成功；两档均通过后自动运行4组PyTorch/TensorRT配对资源测试与最终核验。
当前尚无本轮最终耗时/内存结果，不宣称TensorRT优化已验收。

进度与结果保存在：
`/home/nvidia/multicamera-jetson/deployment/fast-foundation-trt_20261107/`。
其中 `status.json` 为原子更新状态，`service.log` 为控制器日志，分辨率目录保留构建日志；
全部完成后生成 `RESULTS.md`、`measurements/summary.json`、`measurements/validation.json`
及 `job-manifest.json`；失败时 `status.json` 记录原因。查看：

```bash
systemctl --user status multicamera-fast-stereo-benchmark
cat /home/nvidia/multicamera-jetson/deployment/fast-foundation-trt_20261107/status.json
```

当前用户服务仅在本次任务运行，不改变网页深度后端或相机标定。
新控制器Ruff/编译通过，诊断回归37项通过、14项条件跳过；独立Systemd环境已实际核对
Torch2.1 NVIDIA、TensorRT8.5.2.2、CUDA可用与全部输入在Orin。

## 2026-11-09：640 档同权重双引擎＋CUDA GWC（尚未设备验收）

本轮限定 `20-30-48`，SHA256必须为
`98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692`，
上游提交固定为 `476f4249561f7c79ca707326954f9255643412a6`；
仅640×352、8次迭代、最大视差192，不扩尺寸、不升级系统。

新增 `scripts/jetson/fast_foundation_dual.py`，提供独立的 `export/build/validate` 入口。
导出前验证权重内容摘要及上游身份，以随机输入检查两段PyTorch包装器与完整模型一致性；
导出opset16的feature/post两段，特征与GWC边界固定FP16，视差输出FP32。
特征引擎接收RGB 0–255，包含原模型图像归一化。两段间共享预分配CUDA缓冲区，
由独立CUDA核构建GWC，不在推理时调用eager PyTorch GWC。
PyTorch仅负责设备张量、输入拷贝和当前CUDA流；这是独立实验入口，未接入网页后端。

`fast_foundation_gwc.cu` 按当前部署的eager `pytorch1` 语义实现：
FP32范数按各自的 `max(norm, 1e-12)` 归一化，再将操作数及乘积舍入到FP16，
FP32累加后写回FP16。无效左边界置零。没有直接采用上游附带CUDA核的
`dot / (norm_left * norm_right + 1e-5)` 公式，避免改变当前参考基线。
编译目标为Orin的sm_87，禁用FMA融合，不启用fast-math。

构建沿用现有TRT8.5启发式、关闭TF32、FP32保护与精度约束策略；未使用本机不存在的
`builder_optimization_level`。本轮未调整FP32保护范围，性能收益待设备实测。
feature/post分别保存 `export.json`、ONNX、引擎和 `build.json`；同一尺寸目录内顺序复用
`timing.cache`，不允许并发构建或写同一缓存。库与源码摘要保存于 `kernel.json`。
重新执行build只复用身份及摘要一致的已完成产物，拒绝覆盖不明或被修改的产物。
导出目录必须全新；失败的目录保留供诊断，下一次导出应选新目录。

设备环境允许执行时，先同步本轮三个Python/CUDA实现文件，再使用新目录：

```bash
cd /home/nvidia/multicamera-jetson
.fast-stereo-venv/bin/python scripts/jetson/fast_foundation_dual.py export \
  --root models/fast-foundation-stereo_476f4249561f \
  --checkpoint models/fast-foundation-stereo_476f4249561f/weights/20-30-48/model_best_bp2_serialize.pth \
  --directory deployment/fast-foundation-dual_640-next
.fast-stereo-venv/bin/python scripts/jetson/fast_foundation_dual.py build \
  --directory deployment/fast-foundation-dual_640-next
.fast-stereo-venv/bin/python scripts/jetson/fast_foundation_dual.py validate \
  --root models/fast-foundation-stereo_476f4249561f \
  --checkpoint models/fast-foundation-stereo_476f4249561f/weights/20-30-48/model_best_bp2_serialize.pth \
  --directory deployment/fast-foundation-dual_640-next \
  --left /absolute/path/left_rectified.png --right /absolute/path/right_rectified.png
```

执行时应继续使用既有资源监视器及2GiB可用内存预留线；上面的命令仅展示各阶段参数，
不代表当前有足够资源。validate检查两种归一化分支、零值/小范数、batch>1、非方形输入、
视差超过图宽和边界；再用实际feature引擎输出比较CUDA/eager GWC，最后比较完整
PyTorch与双引擎视差，以及同一双引擎换用eager GWC后的视差。
GWC容差为atol/rtol各0.002；整网沿用已有试验门限：有限值、MAE≤0.5px、
P95≤1px、>2px占比≤1%。保存原始视差、输入摘要、逐项误差和同步前向计时。
这些门限是转换一致性验收，不代表真实场景深度精度。

当前实际状态：权重摘要已在Orin核对一致，CUDA源码已通过现有nvcc编译，未执行核。
首次640导出在约11秒触及可用内存低于2GiB的保护后被终止，没有形成可用ONNX或新引擎。
日志保留于Orin `deployment/fast-foundation-dual_20261109/export.log` 和
`export.resources.json`。用户选择保留原网页深度及追踪等现有服务，故停止后续设备测试，
转为完成代码与本地检查。当前不宣称CUDA数值检查、双引擎构建、整网一致性或性能验收通过。
本地诊断回归、静态检查与语法检查通过；已有服务未停止，系统依赖未改动。

## 2026-11-09：分进程导出与内存保护下的继续验证

本轮继续保持原网页深度、人体追踪、预览和MQTT运行，保留既有2GiB MemAvailable保护线，
没有改变系统、功耗模式、分辨率、权重或迭代数。
所有设备重任务均由现有 `deployment/fast-foundation-trt_20261107/run_guarded.py` 包装，
每200ms采样；触线终止本次任务的进程组并保存资源报告，不停止其它服务。

`export` 现在通过两个独立子进程顺序追踪feature/post，中间特征写入 `features.npz`。
只保留当前分段所需模块，再释放完整模型、临时张量、CUDA allocator缓存和可回收的CPU堆。
上游 `TrtPostRunner.feature` 未参与forward，导出post前移除这一冗余模块引用。
GWC参考采用逐视差切片，保留每个操作的归一化、乘法、求和与舍入语义，避免展开六维体积。

GPU追踪仍触线，因此新增 `--trace-device cpu`（默认），在CPU上用原FP32权重追踪固定图，
特征和GWC边界保持FP16，post输入在图内转换到FP32。TensorRT构建时仍按原策略选择FP16
卷积/矩阵计算及FP32保护；CPU追踪不代表把FP32计算作为最终推理精度，也不证明TRT数值合格。
CPU分段导出耗时78.22秒，未触发保护；两份ONNX均通过checker，静态折叠后feature为671节点、
post为2011节点。原始和折叠后ONNX分别保留，并记录全部摘要。

参考结果复用2026-11-07完整eager PyTorch实测产物；复用前对照Host留存副本检查报告与视差摘要，
并检查权重、源码提交、输入文件、640×352、8次迭代和最大视差192身份。
`reference --cached-reference ...` 必须显式提供报告与视差的独立已知SHA256；不按目录名猜测身份。
CPU分段包装器对完整eager参考：MAE 0.00419382px、P95 0.01528168px、最大差0.50550079px、
>2px比例0，通过原有门限。此结果验证分段包装器，不能标为TensorRT整网验证。

`kernel-check` 可独立验证已构建CUDA库，无需加载TensorRT引擎。
Orin上六组小尺寸测试（两种归一化、零值/小值/随机值、batch=2、非方形、视差超过图宽）
均与原eager完全相同；逐视差参考也逐元素相同。
额外使用固定样例的CPU PyTorch特征（1×224×88×160）检查CUDA输出
1×8×48×88×160，最大绝对差0.0009765625，通过atol/rtol均为0.002的门限。
该固定权重实际 `normalize=false`；另一分支由小尺寸测试覆盖。
报告位于 `640x352-i8-cpu-sim/kernel-validation.json` 与 `kernel-fixture-validation.json`。

构建增加可选 `--core-tactics-only`，仅开启TRT8.5本机支持的JIT卷积tactic source位，
并将builder线程限制为1；工作区可降至64MiB，仍保留原精度保护和启发式。
这些设置不会限制TensorRT初始化、解析、权重、激活和其它分配的总和。
未使用 `builder_optimization_level`，本机也没有TACTIC_DRAM内存池接口。
`--serial-features` 另可导出batch=1的特征引擎，左右图分别执行并复制到post的独立缓冲区，
保留左目特征直到右目完成；该模式仍只有feature/post两个引擎。

本轮所有产物与资源日志根目录：Orin和Host副本均为
`deployment/fast-foundation-dual-staged_20261109/`（Host位于 `data/` 下）。
原始CPU双目图在 `640x352-i8-cpu/`，静态折叠图及核报告在 `640x352-i8-cpu-sim/`；
串行特征尝试单独保留于 `640x352-i8-serial*/`，不覆盖先前图或失败日志。
最终构建状态以本轮 `RESULTS.md` 和 `status.json` 为准。

串行特征图可复用已核验的post ONNX，因为post的特征/GWC绑定合同未改变。
组合目录中feature的 `serial_features=true` 描述其执行方式，post继续保留原导出身份，
不修改其ONNX；最终由 `DualEngineRunner` 校验实际绑定的形状和类型。
组合目录的 `reference.json` 明确注明早先的分段对比来自batch=2 CPU包装器；
这不能替代串行双引擎的最终对比。最终仍必须对照同一完整eager视差执行validate。

构建另有互斥的 `--precompiled-tactics-only`（tactic source位集为0），用于禁用外部、
edge-mask和JIT卷积算法来源；这是TRT8.5的既有接口，不是新版优化级别参数。
是否可覆盖本模型全部算子、是否完成构建及数值验收，必须以实际结果为准。
构建完成后释放解析图与builder，再用序列化缓冲区直接写引擎，避免复制一份同等大小的bytes。

运行期间另有服务控制操作：系统日志记录预览和追踪多次正常停止/启动，NRestarts=0，
并非本轮命令发起。已核对它们及Mosquitto当前active；因此不能宣称本轮期间服务从未重启，
也不将本轮耗时与内存采样作为稳定业务负载基准。后续报告保留这一环境变化。

### 本轮最终状态：导出和核通过，双引擎未完成

最后的串行特征＋单线程＋64MiB工作区＋tactic source位集0尝试运行1259.72秒，
最低采样MemAvailable为1.971531GiB，触发既有2GiB保护而终止；不是宣称发生系统OOM。
没有生成可核验的feature/post引擎或有效timing.cache，后处理引擎尚未开始实际编译。
因此整网TensorRT数值一致性与性能仍未验收，不能把包装器或独立CUDA核通过当成双引擎通过。

另一次服务重新启动后的核心/JIT构建运行733.91秒后由本轮主动终止，以切换算法来源；
该次资源报告的stop_reason为空，不应误写成内存保护退出。
最后核对预览、追踪、Mosquitto均active，追踪持续更新，MQTT connected；
整机可用约4.44GiB，Swap约110MiB（起初约2.25MiB，为整机观测，不能归因于单一进程）。
期间多次外部服务正常重启及业务负载变化已记录，不报告稳定性能结论。

本地最终诊断回归62通过、14条件跳过，Ruff、语法及diff检查通过。
Orin与Host均保留 `RESULTS.md`、`status.json`（incomplete）、`resource-summary.json`、
`service-health.json`、文件摘要及全部独立日志。后续复用已保存ONNX继续构建，无须重新导出；
仍须在保护线下生成两段引擎并完成validate后才能验收。


## 2026-11-13：192.168.31.28 JetPack 6 部署与高分辨率实测

已部署至新 Orin 的 `/home/nvidia/multicamera-fast-stereo`，独立于 Dashboard 的版本目录。
固定上游提交和 `20-30-48` 权重与前述旧设备一致，权重SHA256为
`98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692`。
权重从旧 Orin 复制后校验，并在受监控基准和启动器中于反序列化前再次校验。
没有复制旧设备的 TensorRT 引擎，不将其跨 JetPack/CUDA 版本复用。

系统为 L4T R36.4.3、CUDA12.6、Python3.10.12、torch2.8.0、torchvision0.23.0。
专用 `.venv` 通过显式 `.pth` 只读复用新安装器管理的固定 GPU 环境依赖，
不是复制整套Torch，不继承 `fs-venv` 或用户site，不修改共享GPU环境；
因此不能删除其引用的 `shared/provisioned/46610ceddefdc1c0/environments/gpu`。
本次部署没有新增pip安装，模型权重约60MB，含测试结果整个独立目录约104MB。
默认 Dashboard 深度模型仍为原 FoundationStereo；Fast提供独立按需测试入口。

```bash
# 登录 .28 后执行，按需运行，完成后退出；不依赖Host环境。
/home/nvidia/multicamera-fast-stereo/run-fast.sh \
  --width 1280 --height 736 --iterations 8 \
  --output /home/nvidia/multicamera-fast-stereo/results/manual-1280-01
```

本轮为 PyTorch FP16、eager PyTorch 相关体积、关闭torch.compile、最大视差192。
每档独立新进程串行运行，预热2次后5次CUDA同步计时；输入为官方960×540样例缩放，
并非原生1080p现场采集，不证明精度随分辨率提高。1920×1088为32倍数的近全高清输入。
功耗查询为MAXN_SUPER/0，没有切换功耗模式或锁频。测试前整机已用约4.06GiB。

| 输入/迭代 | 热前向中位ms | 前向FPS | CUDA已分配峰值GiB | CUDA保留峰值GiB | 整机已用峰值GiB | 整机最低可用GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640x352-i8 | 376.77 | 2.65 | 0.71 | 0.76 | 5.23 | 10.06 |
| 960x544-i8 | 859.49 | 1.16 | 1.53 | 1.71 | 5.25 | 10.04 |
| 1280x736-i8 | 1561.15 | 0.64 | 2.68 | 2.91 | 6.45 | 8.84 |
| 1920x1088-i8 | 3443.24 | 0.29 | 5.84 | 6.32 | 9.95 | 5.34 |
| 1280x736-i4 | 1393.53 | 0.72 | 2.68 | 2.91 | 6.99 | 8.30 |

整机内存为每100ms的psutil采样，另存NVIDIA tegrastats原始日志；CUDA为allocator峰值。
进程RSS各档约1.48–1.55GiB，与统一内存GPU占用不是同一口径，不能当作总占用。
CUDA、RSS和整机内存互有重叠，不得相加。热前向不含图像读取、缩放、上传、
深度换算、文件写出或网络，也不是三路并发帧率。近全高清CUDA保留峰值为6.32GiB。

五档均输出有限非空视差及米制深度，无OOM，无swap使用，未触发2GiB可用内存保护线。
最高GPU温度采样59.25°C。1280四次迭代较八次耗时下降约10.7%，CUDA峰值基本不变。
640档约2.65前向FPS，近全高清约0.29前向FPS；本轮没有构建或验收TensorRT。
现有Dashboard、追踪和mosquitto在测试前后均active，Dashboard HTTP200。
现场相机仍不可达，未完成相机真实深度精度、三路深度并发或跨机融合核验。

Orin完整报告与视差/深度数组：`/home/nvidia/multicamera-fast-stereo/results/20261113/`。
`summary.json`汇总五档，逐档保存 `report.json`、`.resources.json` 和 `.tegrastats.log`；
`manifest.json`记录源码、入口、输入与权重身份，`environment.json`记录依赖及复用边界。
本地副本（不含深度/视差数组）在 `data/deployment/fast-foundation-jp6_20261113/`。

## 2026-11-13：新Orin TensorRT10后台构建交接（进行中）

用户关闭Codex前，已启动 `.28` 的独立本机控制器
`/home/nvidia/multicamera-fast-stereo/scripts/trt10_workflow.py`，nohup后台运行，
控制器PPID=1。无需Host参与，保留现有Dashboard、追踪和Broker。
先从同权重、同尺寸且SHA256一致的ONNX在本机串行构建1280×736与1920×1088引擎，
再各自重跑PyTorch与TensorRT成对计时/内存测试，比较完整视差输出。
TensorRT10原生绑定仅链接到专用venv，未升级系统库或修改共享GPU依赖。
构建器已适配TRT10的builder_optimization_level=2，保留TRT8.5原有接口；
工作区2GiB、FP16卷积与FP32保护、关闭TF32；27项相关回归通过。

当前仍在首档编译，尚无可用引擎或TensorRT性能验收结论。
控制器每100ms采样，MemAvailable低于2GiB或单次构建超过1800秒时停止本次子任务，
失败保留日志，不终止其它服务；推理单档限300秒。断开SSH/Codex不依赖Host，
但本次没有设置断电/重启自动续跑。后续连接先读实际状态，不重复启动构建。

- 状态：`/home/nvidia/multicamera-fast-stereo/trt/20261113/status.json`
- 控制器日志：同目录`workflow.log`；逐档`build-*.log`和原生tegrastats采样。
- 产物：同目录两个尺寸子目录的`model.engine`、`build.json`，仅完成构建后生成。
- 测试汇总：同目录`summary.json`，完整流程结束后生成；`validation_failed`不能标为通过。
- 数值门限：全部有限，MAE≤0.5px、P95≤1px、>2px比例≤1%；只说明转换一致性。

