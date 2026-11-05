# RGB 双目 OPENCV5 最小离线验证

只读取原有 RGB 会话，复用项目的 `CharucoDetector`，不启动 Qt、访问设备、改写会话或应用结果。依赖装入独立环境，不改变 `pyproject.toml` 和 Windows 包。

## 运行

需要 mrcal wheel 支持的 Python 和操作系统；本次使用 macOS ARM64 / Python 3.12。Linux 可使用匹配的独立环境。Windows 原生和 WSL 的发布集成本轮未验证。

```sh
python3.12 -m venv /tmp/multicamera-mrcal-verify-venv
/tmp/multicamera-mrcal-verify-venv/bin/python -m pip install -r scripts/mrcal_validation/requirements.txt
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/compare_mrcal.py \
  --session offline/calib_20260520_104534 \
  --pair rgb_pair_1 \
  --output data/results/mrcal_opencv5_run
/tmp/multicamera-mrcal-verify-venv/bin/python -m pytest -q scripts/tests/test_compare_mrcal.py
```

输出目录必须为空，避免覆盖旧实验。重复实验可以增加 `--observations <此前输出>/observations.json`，源图片、板配置、相机组的哈希不匹配时拒绝复用。每次仍重新训练和评估，不复用求解结果。网络镜像选择不属于应用配置。

独立环境仅安装一个 OpenCV 发行包 `opencv-python-headless`；其 4.13 版本已包含本次需要的 ChArUco API。不要同时安装 contrib 和 headless 两个提供 `cv2` 的包。对比双方在同一进程使用同一个 OpenCV 检测缓存；应用原有 contrib 环境不变。

## 固定协议

- 按完整图片对划分，默认前段训练、最后 25% 留出，中间隔离两次采集。先划分再按角点有效性过滤。验证至少两对、训练至少六对；验证不用于筛图、初始化或调参。
- 每对只使用两目共同且非共线的至少六个 ChArUco 角点。ID 排序后映射到规则网格，双方使用完全相同的点；局部缺点在 mrcal 中用负权重屏蔽。检测只执行一次，原图不改。
- 棋盘坐标统一到第一个内角点，米制、等距；根据实际 `object_points` 验证 ID/行列顺序。投影前验证 mrcal OPENCV5 与 OpenCV 的公式一致。
- OpenCV 分别比较先单目标定再固定内参的双目标定，以及相同初始 K/D 的联合双目标定。mrcal 两种策略都从训练得到的 OpenCV 固定内参结果开始，固定参考左目，优化右目外参和各帧板姿态。它是后端替换实验，不是独立冷启动实验。
- 所有策略固定五项普通 OpenCV 畸变参数；不使用 fisheye、样条、板形变、异常剔除或正则。保留同一个观测掩码。
- 统一二维点 RMS：`sqrt(mean(dx² + dy²))`，额外报告点误差中位数、P95、图像外侧 20% 区域误差和逐采集误差。不把后端原生 RMS 混成统一指标。
- 留出数据只用统一的 OpenCV PnP / SciPy LM 估计每次板的六自由度姿态，固定所有相机参数。`joint_pose` 使用双目观测拟合板姿态，是消除板姿态后的残差，不能单独证明外参准确。
- 更直接的 `cross_camera_transfer`：只用左目拟合板姿态后预测右目，再反向执行；合并两个方向的二维点残差。不会用被预测相机修正该次板姿态。
- 校正后同名角点垂直误差为辅助指标，每个模型使用自身 `alpha=0` 的校正尺度，不能作为严格同尺度优劣结论。主要比较原始像素域的预测误差。
- 时间只报告双目求解，公共检测、初始化、文件写入、评估不计入。OpenCV 单目初始化时间单独记录。

## 输出和解释

- `observations.json`：原图相对路径、SHA-256、检测计数、共同 ID 与像素坐标、板参数、数据哈希。
- `split.json`：训练、隔离、验证及不可用的完整采集 ID。
- `report.json` / `summary.md`：版本、脚本哈希、协议、四种策略参数及统一误差。`status=passed` 只说明实验协议和数值检查完成，不表示标定质量合格。
- `mrcal_{fixed,joint}_{0,1}.cameramodel`：包含训练观测的原生模型，可重新分析，不经过有损的 Kalibr 导出。
- `failure.json`：失败原因；失败实验不会产生成功报告或触及应用结果。

连续尾段加隔离只能减少相邻帧泄漏，不能证明所有板姿态独立。旧数据缺少曝光同步证据、独立重采与实测外参真值，因此本次不能据此证明泛化精度、绝对基线准确或不确定度可信。不要根据留出误差反复改划分、过滤或调参后仍把它称作未使用过的验证集。要进一步选模型，应另设调参集和最终测试集。

官方依据：[安装与平台](https://mrcal.secretsauce.net/install.html)、[外部检测器与部分网格](https://mrcal.secretsauce.net/recipes.html)、[模型定义](https://mrcal.secretsauce.net/lensmodels.html)、[优化 API](https://mrcal.secretsauce.net/mrcal-python-api-reference.html)。

## 异常角点与不确定度验证

复用前一步输出的原生 mrcal 联合优化模型及角点缓存，不重新选取训练/验证图片：

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/validate_mrcal.py \
  --baseline data/results/mrcal_opencv5_final \
  --session offline/calib_20260520_104534 \
  --output data/results/mrcal_robustness_uncertainty_run \
  --samples 300 --seed 20261026
/tmp/multicamera-mrcal-verify-venv/bin/python -m pytest -q scripts/tests
```

输出目录必须为空。默认协议会在任何求解前写入 `protocol.json`，本次实际执行结果见 `ROBUSTNESS_RESULTS.md`。

异常角点分两种数据域：沿用真实采图姿态和缺点掩码生成精确投影并加 0.2 px 高斯噪声，以及直接使用原有真实角点。每域各做三次固定种子的实验，分别不注入、注入 1%/8 px 和 5%/20 px 错点。错点方向随机，幅度固定，左右相机的角点观测独立抽取；不修改图片或已有角点文件。每次从同一模型开始，比较开启/关闭 mrcal 默认异常剔除。单独统计输入缺点、注入点、命中点和额外剔除点；仅合成域的非注入点具有已知的“正常点”真值。真实数据上的额外剔除不能直接解释为误报。

不确定度实验从原模型和每次板姿态生成精确 OPENCV5 观测，以此作为模型一致真值。分别注入每坐标 sigma=0.1/0.3/0.6 px 的独立高斯噪声，各进行 300 次独立重标定，总计 900 次，固定缺点掩码，关闭异常剔除、正则和板形变。每次均从相同真值开始。查询每目 3×3 网格、距离 1 m/3 m/无穷远的投影协方差，使用已知注入 sigma，不从残差反推 sigma。实测投影按官方测试的 `mean-pcam` 方法补偿板姿态与相机坐标变化，避免将坐标规约差异混入采样误差。

独立样本数是每个查询的重标定次数，不将同一次标定中的 54 个相关查询伪装成独立样本。检验包括白化后两个坐标的样本方差（卡方区间）及 95% 椭圆覆盖率（二项检验），在所有查询/噪声级别上作 Bonferroni 校正、族错误率 0.05。完整经验协方差、预测协方差、标准差比、覆盖率和失败试次写入 `report.json`；各噪声级别的投影样本写入 `.npz`，可独立复算。失败试次不会悄悄从成功结论中删除。

脚本的 `status=completed` 表示实验完成；能力是否通过应读取统计检查和恢复指标。通过只覆盖这套模型、噪声和几何条件，不等于实际镜头准确、真实噪声服从独立高斯，或完整双目测距不确定度可信。

专项测试包含宽覆盖正例、错误 sigma 反例和窄覆盖外推反例，不会为了让退化输入“通过”而放宽统计门槛。官方依据：[不确定度及局限](https://mrcal.secretsauce.net/uncertainty.html)、[采样验证实现](https://github.com/dkogan/mrcal/blob/v2.5.2/test/test-projection-uncertainty.py)。

## 样条模型与视场边缘

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/compare_mrcal_spline.py \
  --baseline data/results/mrcal_opencv5_final \
  --session offline/calib_20260520_104534 \
  --output data/results/mrcal_spline_run --samples 200
/tmp/multicamera-mrcal-verify-venv/bin/python -m pytest -q scripts/tests
```

仍使用原始 24 对训练、9 对验证、共同角点和缺点掩码。预先固定两档三次样条 `Nx=12,Ny=8` 和 `Nx=16,Ny=10`，水平定义域 120°，不根据留出误差选择密度。训练 OPENCV5 模型给出的水平视场约 109°，只用于选取初始化定义域。先在训练 OPENCV5 的射线网格上拟合样条作为初值；这些虚拟网格点不进入实际标定目标函数。

按官方方法固定样条 core 的四项参数（它们与样条校正冗余），优化样条系数、双目外参和板姿态，使用内置弱正则。对照包括 OPENCV5 原无正则模型及同样开启正则的 OPENCV5。所有模型关闭异常剔除和板形变。无正则样条作为欠约束诊断，结果完整记录，即使求解器返回了有限参数，也不据此认定可用。

评估全部通过原生 `mrcal.project()`，不会将样条转换成 K/D。每个模型用相同的 SciPy LM 仅拟合验证板姿态，计算两个方向的跨目预测误差。误差在原始像素域比较，按目标相机的实际角点位置分成中心、外侧 20% 边缘、双方向均处于边缘的角区，以及左/右/上/下；角区与单边存在明确重叠，不把重叠计数累加成总点数。

不确定度图在每目 17×11 网格上查询 1m/3m/无穷远，采用相同已知 sigma=0.2 px、`mean-pcam`。报告中心与边缘的最大方向标准差中位数、P95、最大值及每查询原始数值。无效协方差或投影往返失败单列为错误，不以零、不确定或低置信值替代。

OPENCV5 正则对照和主样条模型各再做 200 次精确投影 + 已知噪声的独立重标定，保留各自正则设置，以 3×3 网格、三种距离检查解析不确定度。统计规则沿用前一轮，并对两种模型的 108 个查询作多重比较校正。该验证只检查模型一致的噪声传播，不能证明真实镜头模型误差已消除。

输出 `protocol.json`、`report.json`、`summary.md`、每路原生 `.cameramodel`，以及 Monte Carlo 投影样本 `.npz`。另记录训练覆盖分布和每个样条系数受多少真实观测直接约束；视场外控制点可能本来就无观测，须结合不确定度及无正则诊断解释。实际结果见 `SPLINE_RESULTS.md`。

官方依据：[样条参数选择](https://mrcal.secretsauce.net/how-to-calibrate.html)、[原生投影模型](https://mrcal.secretsauce.net/lensmodels.html)、[官方求解流程中固定样条 core](https://github.com/dkogan/mrcal/blob/v2.5.2/mrcal-calibrate-cameras)。

## 16×10 无效协方差原因诊断

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/diagnose_mrcal_spline.py \
  --spline-baseline data/results/mrcal_spline_verified \
  --opencv-baseline data/results/mrcal_opencv5_final \
  --output data/results/mrcal_spline_diagnosis_run
```

只读取先前原生模型；输出独立诊断，不覆盖历史标定。先复现每个查询的原始反投影、归一化射线、投影往返、协方差，然后定位非有限值首次出现的位置。严格拒绝零向量及非有限的投影往返；`normalize=True` 返回有限值不代表反投影成功。

正常查询下，对比 mrcal 原生协方差与独立列主元 QR 计算的正则化 sandwich 协方差。QR 不形成 `JᵀJ`，也不使用 mrcal 的分解器；但与原生实现共用同一个模型 Jacobian 和投影梯度，所以它验证的是线性代数路径，并非全部导数实现。问题点附近另用数值差分核对投影导数、扫描 Jacobian 行列式，并从多个初值寻找反投影根。临时探针只在当前进程拦截私有函数，退出或异常时恢复，不修改 mrcal 安装文件。

控制包括相同真实观测下改为 16×12/FOV120、16×10/FOV140，以及 OPENCV5 oracle 精确投影下的原采图几何与新增 24 个角区姿态。所有控制不使用留出误差做模型选择。合成增广仅用于因果诊断，不是假装新采了照片，也不会写入会话或交付为应用标定。结果见 `SPLINE_DIAGNOSIS.md`。

## 16×10/FOV140 同划分比较

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/validate_mrcal_fov140.py \
  --opencv-baseline data/results/mrcal_opencv5_final \
  --spline-baseline data/results/mrcal_spline_verified \
  --session offline/calib_20260520_104534 \
  --output data/results/mrcal_fov140_run --samples 200
```

复用已验证的 OPENCV5 弱正则模型和 12×8/FOV120 原生模型，仅从相同训练初值新求解 16×10/FOV140。三者使用原角点、原 24/9 对训练/验证划分，同样关闭异常剔除与板形变、开启弱正则。原生模型与既有报告参数和哈希核对后再运行，不自动采用诊断控制结果。

边缘误差继续使用相同原始像素坐标分区、相同板姿态拟合程序。反投影扩展到包括像素边界的 65×37 网格（每目 2405 点），并独立检查每目 1083 个留出角点及历史失败点。判定要求非零有限射线、有限往返且误差 ≤1e-5 px。另在同一组 OPENCV5 参考物理射线上检查局部 Jacobian 方向，避免不同模型的逆解分支掩盖折叠。

条件不确定度仍在 17×11 网格、1m/3m/无穷远、sigma=0.2 px 下计算。三种模型各重新做 200 次加噪重标定，加噪查询推进到每方向 2.5%、50%、97.5%（上一轮 Monte Carlo 为 10%、50%、90%），因此不能直接沿用旧样本；对三模型合计 162 个查询统一作多重比较校正。结果写入独立目录，包含原生模型、协议、完整报告及 `.npz` 样本，不覆盖历史记录。实际结论见 `FOV140_RESULTS.md`。

## 新会话上的冻结模型测试

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/test_mrcal_external.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_fov140_comparison \
  --original-session offline/calib_20260520_104534 \
  --output data/results/mrcal_test1_run
```

此入口不调用相机标定，只检测新图、拟合每帧板姿态并评估冻结的三种原生模型；拒绝板规格、图像尺寸不匹配及与旧会话文件完全重复，不自动缩放或交换左右。相机安装/对焦/视场是否保持不变本次用户不确定，因此结果明确标为跨会话迁移诊断，不能作为最终泛化验收。

同时报告单目独立拟合残差和双目跨目预测残差，并从每帧独立板姿态推算相机关系差，仅作不一致诊断，不采用新的外参。所有源参数、脚本和输入图片保留哈希，结果目录必须为空。实际 test1 结果见 `TEST1_RESULTS.md`。

## test1 内重新训练三种固定配置

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/compare_mrcal_session.py \
  --session data/test1/calib_20261019_152613 \
  --observations data/results/mrcal_test1_frozen/observations.json \
  --output data/results/mrcal_test1_retrain_run
```

使用新会话自己的训练图片求解内参和双目外参，不沿用旧会话参数。冻结配置为 OPENCV5、12×8/FOV120、16×10/FOV140；共用前段训练、末尾 25% 验证及中间两对隔离，本数据得到 16/6/2 对。划分与配置先于求解写入 `protocol.json`。角点缓存按原图与板配置哈希核对；不根据误差筛点、换划分或改变 FOV。

物理左右相机可能反向，但保留文件标记；新外参按文件角色重新求解，不自动交换原图、不要求平移沿用旧符号。三模型开启弱正则、关闭异常剔除和板形变。验证只拟合板姿态，不更新相机参数。输出独立原生模型和误差报告，不写入应用会话结果。本轮仅做会话内误差比较，不追加噪声、不确定度或稳定性实验。实际结论见 `TEST1_RETRAIN_RESULTS.md`。

## test1 采图质量和模型适配性

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/assess_mrcal_capture.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/mrcal_test1_quality_run
```

只读现有图片、角点和新会话原生模型，不重新标定、不删帧。图片指标取角点凸包 ROI，报告灰度分位数、黑/白截断比例及 Laplacian 方差；这些受印刷黑白图案和板大小影响，只作描述，不使用固定阈值自动判模糊、欠曝或过曝。逐帧板姿态用于距离/倾角分布，不等于测量真值。

同时记录 8×6 网格角点数和每格覆盖的独立采集次数，分开报告训练/隔离/验证；用冻结相机参数重估板姿态，生成逐帧残差、空间残差均值和单目姿态推算的相机关系变化。反投影采用前一轮严格检查；协方差查询保留无效计数，无效值写 `null`，不填零，且有效子集统计明确分母。统一 sigma=0.2 px 查询 1m/3m/无穷远，作为条件噪声敏感性，不代表真实曝光噪声已知或完整测距误差。实际结论见 `TEST1_QUALITY_RESULTS.md`。

## test1 留出三角测量与三维不确定度

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/assess_mrcal_triangulation.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/mrcal_test1_triangulation_run --samples 1000
/tmp/multicamera-mrcal-verify-venv/bin/python -m pytest -q scripts/tests
```

读取同一训练结果的三种冻结模型，只三角化固定留出 0022、0023、0027–0030，不重新拟合相机或板姿态。使用相同 `leecivera_mid2`，直接比较 40mm 相邻点距、全部无序点对距离和逐帧最佳拟合平面偏差；不对重建点做尺度修正。深度定义为文件 camera0 的 Z，保留原文件角色和外参方向。

标定噪声与本次角点观测噪声各假设每坐标 sigma=0.2px、左右观测独立，开启相机坐标稳定化。分别记录标定、观测和联合三维协方差，距离不确定度使用完整点间交叉协方差。每帧取中心附近和最外缘两个确定性查询，各做 1000 次固定相机观测加噪，检查深度方差；36 查询统一 Bonferroni 校正，族错误率 0.05。该检查不包含重新标定，不能视为完整标定三维协方差的采样验证。

`report.json` 保存逐帧和分区统计、数值检查和模型哈希；每模型每帧 `.npz` 保存像素、三维点、名义板坐标及完整协方差以便复算。板尺寸/平整度未经独立量测，标定和评价共享名义板规格，不能据此验证全局绝对尺度；最佳拟合平面的误差也不能证明绝对深度精度。结果见 `TEST1_TRIANGULATION_RESULTS.md`。

## test1 双目校正、有效视场和边缘拉伸

```sh
/tmp/multicamera-mrcal-verify-venv/bin/python scripts/assess_mrcal_rectification.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/mrcal_test1_rectification_run
/tmp/multicamera-mrcal-verify-venv/bin/python -m pytest -q scripts/tests
```

相机与板姿态均不重新拟合，直接变换原固定留出同名角点，比较三模型的垂直像素差及极线平面角度差。分别评估 PINHOLE 和 LATLON 输出；各投影内固定输出尺寸、焦距与主点，不使用每模型自动缩放或裁边。110°×82° 中心视角只由训练得到的 OPENCV5 稠密图像边界推定，未按验证角点选取。相机校正朝向分别根据各自训练外参确定。

有效视场报告双目共同有效像素、立体角积分及最大无黑边矩形；有效需完整双线性采样范围在源图内。该掩码不保证近处目标无视差越界或遮挡。拉伸为原始像素到校正像素映射的局部 Jacobian 奇异值及其比值，同时报告中心归一化倍率。每目 65×37 查询，在三模型共同可见源像素集合上统计；不把无效查询填零。输出原生校正模型、稠密映射/掩码、拉伸查询、逐帧误差及首末留出帧的对照图。实际结论见 `TEST1_RECTIFICATION_RESULTS.md`。

## Triggs / Calibration Wizard 信息与权重原型

```sh
.venv/bin/python scripts/verify_calibration_information.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/calibration_information_test1_replay
.venv/bin/python scripts/verify_calibration_information.py \
  --session offline/calib_20260520_104534 \
  --baseline data/results/mrcal_opencv5_final \
  --output data/results/calibration_information_old_replay
```

这是独立 SciPy/OPENCV5 联合优化原型，不接入应用。复用历史划分和共同角点，比较等权、梯度强度标量权重、保留 xy 相关性的二维权重。梯度信息归一化只用训练图，界限预先固定，不删除角点、不从留出残差反推权重。信息规划使用板姿态消元后的双目内参边缘信息，检查双目投影边距、正深度和最小角点间距；候选仅有限网格，未来角点质量用假设模糊的合成交叉点估计，不宣称完整复现论文的连续全局搜索。

两组真实数据的加权收益不一致，当前不采用到应用。完整阅读结论、负结果、已知噪声线性化控制及边界见 `CALIBRATION_INFORMATION_RESULTS.md`。

## 新采集数据的 mrcal 引入必要性

2026-10-30 会话的受控比较见 [必要性评估](NEW_CAPTURE_MRCAL_NECESSITY_RESULTS.md)。
相同 OPENCV5 下 OpenCV 与 mrcal 留出误差基本一致；既定两档样条在训练缺少下边缘覆盖时，留出和三维误差明显变差，暂不引入应用运行时。

独立入口 `scripts/assess_mrcal_necessity.py` 会在固定虚拟网格上验证初始化射线，显式记录无效位置，真实训练与留出角点不被过滤；最终冻结模型审计不要求参考 OPENCV5 的全图反投影处处有效。
本机环境启动方式、完整协议、失败记录、数值结果和适用限制见上述报告。

## DiscoCal 测量模型与观测不确定度迁移

研究与三组固定留出结果见 [DiscoCal 离线迁移评估](DISCOCAL_TRANSFER_RESULTS.md)。
本轮使用完整 ChArUco 检测器的图像扰动估计协方差，比较等权、标量与二维权重；这是迁移思路验证，不是原生圆板 DiscoCal 复现。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_discocal_transfer.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/discocal_transfer_test1_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/assess_discocal_uncertainty.py \
  --experiment data/results/discocal_transfer_test1_replay \
  --observations data/results/mrcal_test1_retrained_verified/observations.json
.venv/bin/python -m pytest -q scripts/tests/test_discocal_transfer.py
```

首个入口要求输出目录尚不存在，保留既有实验；各会话数据和缓存参数见报告。仅训练图片参与协方差估计，留出始终作未加权评价。不安装或导入 DiscoCal，不需要 mrcal。
第二入口使用完整板姿态的线性化传播；真实扰动存在漏检时明确标记整套协方差不可用，不填零或伪造精度图。协议中的噪声、收缩与限幅是预设敏感性条件，不能当作真实传感器噪声或绝对置信区间。

## TartanCalib 迭代检测与亚像素定位迁移

论文/源码核对、三组固定留出和边缘结果见 [TartanCalib迁移评估](TARTANCALIB_TRANSFER_RESULTS.md)。
原型保留原共同ID；主评价始终使用冻结原始留出角点，另有对所有模型相同的图像细化参考，不用各模型自己的留出重检测结果评价自己。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_tartancalib_transfer.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/tartancalib_transfer_test1_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/diagnose_tartancalib_transfer.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --experiment data/results/tartancalib_transfer_test1_replay
.venv/bin/python -m pytest -q scripts/tests/test_tartancalib_transfer.py
```

首个入口需要未使用的输出目录及OpenCV 4.13，固定其ChArUco像素约定；比较固定窗口、两种比例窗口和三轮原生内参重检测叠加细化。第二入口要求对应实验存在且其 `native_ablation/` 尚不存在，单独验证保留原生细化的三轮重检测，并只在训练图像上挑战焦距与定位初值。未引入TartanCalib/ROS/mrcal依赖，没有新增UI入口。正式结果采用 `tartancalib_transfer_final_old/test1/new/`，完整协议与限制见报告。

## calibcam 原生 OPENCV5 联合优化对照

源码核对、三组固定留出、边缘与收敛结果见 [calibcam 对照评估](CALIBCAM_COMPARISON_RESULTS.md)。调用固定提交的原生投影、JAX 导数和联合求解核心；显式开放五项畸变，固定 xi/skew，复用冻结角点。与应用实际联合标定、10000 次参考及既有解析优化比较，另设原生投影的稠密求解控制。

独立环境准备（从项目根目录运行；仅首次执行克隆和建环境）：

```sh
git clone https://github.com/bbo-lab/calibcam.git data/research/calibcam_source
git -C data/research/calibcam_source checkout ca2a49d61252995708730f7d9aebca024a3dfceb
uv venv --python 3.10 data/.calibcam-eval-venv
uv pip install --python data/.calibcam-eval-venv/bin/python \
  -e data/research/calibcam_source \
  'bbo-calibcamlib==0.5.3' 'jax==0.4.38' 'jaxlib==0.4.38' \
  'numpy==2.2.6' 'scipy==1.15.3' 'opencv-contrib-python-headless==4.13.0.92' pytest
```

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  data/.calibcam-eval-venv/bin/python scripts/compare_calibcam.py \
  --session offline/calib_20260520_104534 \
  --baseline data/results/mrcal_opencv5_final \
  --output data/results/calibcam_opencv5_old_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  data/.calibcam-eval-venv/bin/python scripts/compare_calibcam.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/calibcam_opencv5_test1_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  data/.calibcam-eval-venv/bin/python scripts/compare_calibcam.py \
  --session data/mrcal_20261030/calib_20261030_135342 \
  --baseline data/results/mrcal_20261030_detection \
  --output data/results/calibcam_opencv5_new_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  data/.calibcam-eval-venv/bin/python -m pytest -q scripts/tests/test_compare_calibcam.py
```

所有输出目录须尚不存在；首次 JAX 导入/编译耗时较长。`report.status=completed` 表示实验完成，各方法收敛看 `solver.success`，达到次数上限的解保留失败状态。正式产物目录去掉上述 `_replay` 后缀。实验原生源码未修改，应用环境没有安装 calibcam/JAX，不改应用 UI 或 Windows 打包。
