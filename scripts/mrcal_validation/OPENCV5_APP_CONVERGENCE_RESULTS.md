# 应用 OPENCV5 联合优化与收敛检查（2026-10-29）

应用已经实现联合精修；缺失的是针孔单目和双目求解的显式收敛条件。本轮只补该条件，不改变默认策略、不重复实现联合优化，也不改 UI 或 Windows 打包。

## 已有实现与最小修改

- 默认“标准”策略使用 `CALIB_FIX_INTRINSIC`；已有“联合精修”使用 `CALIB_USE_INTRINSIC_GUESS`，允许双目阶段更新两目 K/D 及 R/T。
- `StereoCalibrator` 用求解器返回的新内参生成校正映射，成功后才写回；`PairCalibrationWorker` 在联合更新内参后用原选用观测重估板姿态及单目误差。现有实现完整，本轮未改。
- 原针孔 `calibrateCamera`、`stereoCalibrate` 未传 criteria，依赖 OpenCV 默认值。新增共享 `PINHOLE_CALIBRATION_CRITERIA`：`EPS | MAX_ITER`、上限 1000 次、epsilon=1e-10，分别传给两处针孔调用。EPS 允许提前退出，上限不是强制迭代次数。
- 100 次的训练解仍有变化，1000 次与 10000 次参考参数一致，因此选择 1000；依据是训练解稳定性，不按留出误差选最优上限。该上限不保证所有可能输入都收敛，OpenCV 接口也没有给出本次实际迭代数或停止原因。
- 鱼眼、固定内参/联合精修 flags、默认选择、数据筛选、导出格式及参数提交边界不变。双目日志增加实际 flags 与 criteria。

## 应用真实求解路径验证

使用 `.venv` 内 OpenCV 4.13.0，读取历史哈希核对的同一批共同角点，逐帧通过应用 `add_detection` / `add_detection_pair` 送入实际 calibrator。原会话训练24对、隔离2对、验证9对；test1 训练16对、隔离2对、验证6对。未使用 test3，也未重分留出集、删点或改变镜头模型。

“修改前默认”控制在进程内去掉 criteria，复现旧调用；100/1000/10000 对照只替换 criteria。修改后再运行无覆盖条件的应用代码，并重新与10000次参考比较。所有覆盖都在独立离线进程内，未修改安装环境或应用状态。

留出指标为原始像素二维点跨目预测 RMS，两方向合并；每幅验证图只用单目拟合板姿态预测另一目，不更新相机参数。校正图像的自动缩放误差不是主比较指标。单目标定观测固定为相同共同角点子集，因此是受控后端比较，不涵盖应用可能存在的额外单目观测和不同筛选策略。

| 数据 | 策略 | 修改前留出 RMS px | 修改后留出 RMS px | 改善 |
|---|---|---:|---:|---:|
| 原会话 | 固定内参 | 0.797791 | 0.797791 | 基本不变 |
| 原会话 | 联合精修 | 0.705997 | 0.513190 | 27.3% |
| test1 | 固定内参 | 0.513184 | 0.513184 | 基本不变 |
| test1 | 联合精修 | 0.510726 | 0.431530 | 15.5% |

联合精修训练 RMS 分别为 0.297109→0.291928、0.255518→0.248405px；100次上限的留出误差为0.631249、0.449070px，仍未达到稳定参考解。两数据、两策略在修改后与10000次参考的 K/D/R/T 最大逐元素差为0（本机同版本结果）。

联合求解及校正映射耗时从原默认约0.1–0.3秒增加至约4–9秒；单次运行包含并发机器负载，不能当作严格性能基准。固定内参求解耗时仍为几十毫秒数量级。后台工作线程调用位置不变；用户选择联合精修时会等待更久。

## 复现与验证边界

```sh
.venv/bin/python scripts/verify_opencv5_convergence.py \
  --session offline/calib_20260520_104534 \
  --baseline data/results/mrcal_opencv5_final \
  --output data/results/opencv5_convergence_replay
.venv/bin/python scripts/verify_opencv5_convergence.py \
  --session data/test1/calib_20261019_152613 \
  --baseline data/results/mrcal_test1_retrained_verified \
  --output data/results/opencv5_convergence_test1_replay --application-only
```

第一条会在当前代码上重现默认条件及多个上限；`--application-only` 不覆盖应用 criteria，只额外计算长迭代参考。输出目录必须不存在，报告记录原图数据哈希、历史划分、实际 API 的 flags/criteria、源码哈希、矩阵、求解时间及留出误差。汇总为 `data/results/opencv5_app_convergence_summary.json`，逐项报告位于 `opencv5_app_convergence_{old,test1}_{before,after}` 目录。

新增数值回归构造固定种子、含0.2px噪声的小倾角观测，确保它能暴露默认联合求解的提前停止；应用结果必须改善目标函数并匹配10000次参考，同时验证更新内参后的旧单目指标失效、重新复核后有效。既有策略 flags、求解结果提交、校正映射及失败回滚专项继续通过。

这些是已有固定留出集上的算法比较，并非新独立盲测或硬件绝对精度证明。不补拍、不证明同步、不更换 OPENCV5 的模型表达能力。用户使用默认“标准”策略时，不应预期表中联合精修的收益。

最终验证：完整应用回归1060项通过、6项跳过；研究脚本回归54项通过；源码标定自检、Ruff、编译及差异检查通过。应用源码与报告哈希、改后参数与长迭代参考参数均已复核。
