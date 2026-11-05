"""Explicit on-demand calibration/capture/fusion jobs for the unified preview."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import signal
import time
import uuid
from pathlib import Path

from aiohttp import web

from .fusion_geometry import validate_active
from .fusion_workflow import write_json
from .stereo_preview import QUALITY_PRESETS, RESOLUTIONS


class FusionController:
    def __init__(self, config, stereo, settings, bundle_root, root=None):
        self.config, self.stereo, self.settings = config, stereo, settings
        self.bundle_root = Path(bundle_root or "bundles").resolve()
        self.root = Path(root or "data/fusion").resolve()
        self.python = Path(".fusion-venv/bin/python").absolute()
        self.task = None
        self.process = None
        self.job = None
        # Keep the latest reviewable result available after service restarts.
        records = [p for p in self.root.glob("*/job.json")
                   if re.fullmatch("[0-9a-f]{32}", p.parent.name)]
        if records:
            try:
                latest = max(records, key=lambda p: p.stat().st_mtime)
                previous = json.loads(latest.read_text())
                if previous.get("id") == latest.parent.name:
                    self.job = previous
                    report_path = latest.parent / "result/alignment_report.json"
                    if previous.get("action") == "fuse" and report_path.exists():
                        self.job["alignment_report"] = json.loads(report_path.read_text())
                        if not self.job["alignment_report"]["passed"]:
                            self.job.update(state="needs_adjustment",
                                            message="旧结果未通过尺度或方向检查，请重新标定融合")
                    if previous.get("state") == "running":
                        self.job.update(state="cancelled", message="服务重启，旧任务未完成，请重新启动")
                        self.save_job()
            except (OSError, ValueError, TypeError):
                self.job = None

    def active(self):
        try:
            data = json.loads((self.root / "active_calibration.json").read_text())
            ids = list(data["cameras"])
            devices = [d for d in self.config["devices"] if d["id"] in ids]
            if len(devices) < 2:
                return None
            validate_active(data, devices)
            return data
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save_job(self):
        write_json(self.root / self.job["id"] / "job.json", self.job)

    async def status(self, request):
        active = self.active()
        return web.json_response(
            {
                "job": self.job,
                "busy": bool(self.task and not self.task.done()),
                "installed": self.python.exists(),
                "active_calibration": None
                if not active
                else {
                    "devices": list(active["cameras"]),
                    "world_frame": active["world_frame"],
                    "tag_side_mm": active["tag_side_mm"],
                    "anchor_tag_id": active["anchor_tag_id"],
                    "created_at_unix_ms": active["created_at_unix_ms"],
                    "measurement_accuracy_verified": False,
                },
            }
        )

    async def start(self, request):
        if (
            request.headers.get("Origin") != f"{request.scheme}://{request.host}"
            or not secrets.compare_digest(
                request.headers.get("X-Camera-CSRF", ""), self.settings.csrf
            )
            or request.content_type != "application/json"
        ):
            raise web.HTTPForbidden(text="请从统一界面启动融合流程")
        if self.task and not self.task.done():
            raise web.HTTPConflict(text="已有标定或融合任务正在执行")
        if self.settings.write_lock.locked():
            raise web.HTTPConflict(text="相机参数正在写入，请稍后启动融合")
        if getattr(self.stereo, "photo_busy", False):
            raise web.HTTPConflict(text="正在拍照，请完成后重试")
        if not self.python.exists():
            raise web.HTTPServiceUnavailable(text="融合环境尚未安装")
        try:
            data = await request.json()
            allowed = {
                "action",
                "devices",
                "tag_size_mm",
                "tag_id",
                "samples",
                "frames",
                "resolution",
                "quality",
                "max_depth_m",
            }
            if not isinstance(data, dict) or not set(data) <= allowed:
                raise ValueError("无效任务参数")
            action = data.get("action")
            if action not in ("calibrate", "fuse"):
                raise ValueError("无效任务类型")
            ids = data.get("devices")
            known = {d["id"] for d in self.config["devices"]}
            if (
                not isinstance(ids, list)
                or not 2 <= len(ids) <= 3
                or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids)
                or not set(ids) <= known
            ):
                raise ValueError("请选择两台或三台设备")
            side = data.get("tag_size_mm", 80)
            if type(side) not in (int, float) or not math.isfinite(side) or not 10 <= side <= 1000:
                raise ValueError("码的黑色外边框边长范围为10–1000mm")
            tag = data.get("tag_id", -1)
            if type(tag) is not int or not -1 <= tag <= 586:
                raise ValueError("AprilTag编号范围为0–586，-1为自动选择共同码")
            samples, frames = data.get("samples", 3), data.get("frames", 3)
            if type(samples) is not int or not 3 <= samples <= 8:
                raise ValueError("标定采样次数范围3–8")
            if type(frames) is not int or not 1 <= frames <= 5:
                raise ValueError("融合帧数范围1–5")
            resolution, quality = data.get("resolution", "480x270"), data.get("quality", "fast")
            if resolution not in RESOLUTIONS or quality not in QUALITY_PRESETS:
                raise ValueError("无效深度分辨率或模型档位")
            far = data.get("max_depth_m", 4)
            if type(far) not in (int, float) or not math.isfinite(far) or not 0.5 <= far <= 10:
                raise ValueError("融合最大深度范围0.5–10m")
            if action == "fuse":
                active = self.active()
                if active is None:
                    raise ValueError("请先完成至少两台相机的外参标定")
                validate_active(active, [d for d in self.config["devices"] if d["id"] in ids])
            params = {
                "devices": ids,
                "tag_size_mm": side,
                "tag_id": tag,
                "samples": samples,
                "frames": frames,
                "resolution": resolution,
                "quality": quality,
                "max_depth_m": far,
            }
        except (ValueError, TypeError, KeyError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        if ((self.task and not self.task.done())
                or getattr(self.stereo, "photo_busy", False)
                or self.settings.write_lock.locked()):
            raise web.HTTPConflict(text="已有任务正在执行，请完成后重试")
        self.job = {
            "id": uuid.uuid4().hex,
            "action": action,
            "state": "running",
            "message": "任务已启动",
            "parameters": params,
            "created_at_unix_ms": round(time.time() * 1000),
        }
        self.save_job()
        self.stereo.batch_active = True
        self.task = asyncio.create_task(self.run(action, params))
        return web.json_response(self.job, status=202)

    async def command(self, args, folder):
        env = dict(
            os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"
        )
        with (folder / "workflow.log").open("w") as log:
            self.process = await asyncio.create_subprocess_exec(
                str(self.python),
                "-m",
                "scripts.jetson.fusion_workflow",
                *args,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                return await asyncio.wait_for(self.process.wait(), 600)
            finally:
                if self.process.returncode is None:
                    await self.stop_process()
                self.process = None

    async def stop_process(self):
        if self.process and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                pass

    async def run(self, action, params):
        folder = self.root / self.job["id"]
        try:
            write_json(folder / "devices.json", self.config)
            if action == "calibrate":
                self.job["message"] = "正在采集左右目并检测共同AprilTag，请保持相机和码静止"
                self.save_job()
                code = await self.command(
                    [
                        "calibrate",
                        "--config",
                        str(folder / "devices.json"),
                        "--bundle-root",
                        str(self.bundle_root),
                        "--devices",
                        *params["devices"],
                        "--tag-size-mm",
                        str(params["tag_size_mm"]),
                        "--tag-id",
                        str(params["tag_id"]),
                        "--samples",
                        str(params["samples"]),
                        "--out",
                        str(folder / "calibration"),
                    ],
                    folder,
                )
                report = json.loads((folder / "calibration/calibration_report.json").read_text())
                self.job["calibration_report"] = report
                if code != 0 or not report["quality_passed"]:
                    self.job.update(
                        state="needs_adjustment", message=report.get("error", "外参检查未通过")
                    )
                else:
                    write_json(self.root / "active_calibration.json", report)
                    self.job.update(
                        state="completed",
                        message="外参一致性检查通过，已用于所选设备融合；测量精度尚未验收",
                    )
            else:
                active = self.active()
                devices = [d for d in self.config["devices"] if d["id"] in params["devices"]]
                validate_active(active or {}, devices)
                subset = dict(
                    active, cameras={d["id"]: active["cameras"][d["id"]] for d in devices}
                )
                write_json(folder / "extrinsics.json", subset)
                total = len(devices) * params["frames"]
                count = 0
                self.job["capture_metrics"] = []
                self.stereo.batch_active = True
                try:
                    for index in range(params["frames"]):
                        for device in devices:
                            self.job["message"] = (
                                f"采集真实双目深度 {count + 1}/{total} · {device['id']}"
                            )
                            self.save_job()
                            result = await self.stereo._compute(
                                device["id"],
                                RESOLUTIONS[params["resolution"]],
                                params["quality"],
                                folder / "captures" / device["id"],
                                index,
                            )
                            if result is None:
                                raise ValueError("采集期间相机参数发生变化，请重新启动任务")
                            self.job["capture_metrics"].append({
                                k: result[k] for k in ("device_id", "valid_fraction",
                                    "median_depth_m", "inference_ms", "receive_skew_ms",
                                    "received_at_unix_ms", "point_frame")
                            })
                            count += 1
                finally:
                    self.stereo.batch_active = True
                self.job["message"] = "正在执行时域过滤、公共坐标点云合并与CPU TSDF融合"
                self.save_job()
                code = await self.command(
                    [
                        "fuse",
                        "--captures",
                        *[str(folder / "captures" / d["id"]) for d in devices],
                        "--extrinsics",
                        str(folder / "extrinsics.json"),
                        "--frames",
                        str(params["frames"]),
                        "--max-depth-m",
                        str(params["max_depth_m"]),
                        "--out",
                        str(folder / "result"),
                    ],
                    folder,
                )
                if code != 0:
                    check_path = folder / "result/alignment_report.json"
                    if check_path.exists():
                        report = json.loads(check_path.read_text())
                        if not report["passed"]:
                            self.job.update(state="needs_adjustment", alignment_report=report,
                                message="标记外参与深度的尺度或方向不一致，已暂停融合；请复核黑色码尺寸与深度")
                            return
                    raise ValueError("融合失败，请下载本次任务日志查看原因")
                output = json.loads((folder / "result/fusion_result.json").read_text())
                self.job.update(
                    state="completed",
                    message=f"融合完成 · {output['point_count']} 点",
                    point_count=output["point_count"],
                    alignment_report=output["alignment_report"],
                    world_frame=output["world_frame"],
                )
        except asyncio.CancelledError:
            self.job.update(state="cancelled", message="服务停止，任务已取消")
            raise
        except Exception as exc:
            self.job.update(state="failed", message=str(exc))
        finally:
            self.stereo.batch_active = False
            self.job["finished_at_unix_ms"] = round(time.time() * 1000)
            self.save_job()

    async def result(self, request):
        ident = request.match_info["job"]
        if not re.fullmatch("[0-9a-f]{32}", ident):
            raise web.HTTPNotFound()
        path = self.root / ident / "result/fusion_result.json"
        if not path.exists():
            raise web.HTTPNotFound()
        try:
            extrinsics = json.loads((path.parent.parent / "extrinsics.json").read_text())
            ids = set(extrinsics.get("cameras", {}))
            devices = [d for d in self.config["devices"] if d["id"] in ids]
            if len(devices) < 2 or {d["id"] for d in devices} != ids:
                raise ValueError("Unknown camera extrinsics")
            validate_active(extrinsics, devices)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise web.HTTPConflict(text="该结果使用的标定已失效，请重新标定后融合") from exc
        data = json.loads(path.read_text())
        check_path = path.parent / "alignment_report.json"
        if check_path.exists() and not json.loads(check_path.read_text())["passed"]:
            raise web.HTTPConflict(text="当前结果未通过深度尺度一致性检查")
        return web.json_response(
            {
                k: data[k]
                for k in (
                    "points",
                    "colors",
                    "point_count",
                    "point_unit",
                    "world_frame",
                    "measurement_accuracy_verified",
                )
            }
        )

    async def artifact(self, request):
        ident, name = request.match_info["job"], request.match_info["name"]
        mapping = {
            "pointcloud.ply": "result/pointcloud/multi_orbbec_world_rgb_pointcloud.ply",
            "pointcloud.html": "result/pointcloud/multi_orbbec_world_rgb_pointcloud.html",
            "mesh.ply": "result/tsdf/tsdf_mesh_clean.ply",
            "mesh.html": "result/tsdf/tsdf_mesh_clean.html",
            "calibration.json": "calibration/calibration_report.json",
            "workflow.log": "workflow.log",
            "alignment.json": "result/alignment_report.json",
            "report.json": "result/fusion_result.json",
        }
        if not re.fullmatch("[0-9a-f]{32}", ident) or name not in mapping:
            raise web.HTTPNotFound()
        path = self.root / ident / mapping[name]
        if not path.is_file():
            raise web.HTTPNotFound()
        response = web.FileResponse(path)
        response.headers["Content-Disposition"] = f'attachment; filename="{name}"'
        return response

    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
            await self.stop_process()
            await asyncio.gather(self.task, return_exceptions=True)
