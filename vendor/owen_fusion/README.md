# Owen 融合流程快照

来源：Owen 的 `gaussians/tools/orbbec_capture` 及其 UI 内的标记板外参工具。
原文件 SHA256 见 `UPSTREAM_MANIFEST.json`；沿用上游 Apache-2.0 许可证（LICENSE）。

用于 Orin 的入口为 `python -m scripts.jetson.fusion_workflow`，不是旧 SDK 采集入口。
原来的 `run_multicam_fusion.py` 含清理采集目录、旧 Orbbec SDK 与路径假设，仅保留参考，
不应在现有相机或已有数据目录直接运行。视频板布局估计器也仅保留参考，未用它推测箱子码布局。

迁移适配：

- `make_multicamera_pointcloud.py` 和 `make_tsdf_mesh.py` 的 Plotly 输出改为内嵌 JS，供离线下载查看。
- 其余核心过滤、体素保留首点、TSDF、外参均值实现保持原样。
- 当前采集用 FoundationStereo 的校正左目 RGB 和米制深度，导出已配准 RGB-D（uint16 毫米）。
- 当前外参用各设备共同的一个 AprilTag 36h11（100mm 黑色外框）定义世界坐标，
  不复用旧 Orbbec 外参。右目检测位姿经基线变换转换至校正左目坐标。
- Open3D 在独立 `.fusion-venv` 子进程按需加载；网页服务及桌面应用不导入 Open3D。
