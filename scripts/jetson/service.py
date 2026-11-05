"""Run with python -m scripts.jetson.service; never activates camera services."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

from .core import Frame, Pairer, Rectifier, depth_points, load_config


class DeviceWorker:
    """Single writer per device; caller submits at most one pair at a time."""

    def __init__(self, device: dict, config: dict, path: Path):
        self.device = device
        self.pairer = Pairer(device, config["max_skew_ms"], config["max_age_ms"])
        self.rectifier = Rectifier(device, path)

    def replay(self, path: Path) -> dict:
        # NPZ has no pickle objects and is a trusted, local recording, not an HTTP upload.
        with np.load(path, allow_pickle=False) as data:
            if (str(data["device_id"].item()) != self.device["id"]
                    or str(data["calibration_sha256"].item()) != self.rectifier.sha256):
                raise ValueError("Replay device/calibration binding mismatch")
            frames = [Frame(self.device["id"], side, int(data[side + "_timestamp_ns"].item()),
                            data[side]) for side in ("left", "right")]
            frames.sort(key=lambda f: f.timestamp_ns)
            pair = None
            for frame in frames:
                pair = self.pairer.push(frame, frames[-1].timestamp_ns)
            if pair is None:
                raise ValueError("Replay pair rejected: stale, duplicate or excessive skew")
            images = [self.rectifier.rectify(frame) for frame in pair]
            points = None
            if "depth" in data:
                if (str(data["depth_frame"].item()) != "rectified_left_axial_z"
                        or int(data["depth_timestamp_ns"].item()) != pair[0].timestamp_ns):
                    raise ValueError("Depth registration/timestamp mismatch")
                points = depth_points(
                    data["depth"], self.rectifier.projection,
                    unit=str(data["depth_unit"].item()),
                    expected_size=tuple(self.device["resolution"]["left"]))
        return {"device_id": self.device["id"], "mode": "offline_candidate_replay",
                "calibration_sha256": self.rectifier.sha256,
                "physical_binding_verified": self.device["physical_binding_verified"],
                "pair_skew_ms": abs(pair[0].timestamp_ns - pair[1].timestamp_ns) / 1e6,
                "rectified_shapes": [list(image.shape) for image in images],
                "point_count": None if points is None else len(points),
                "point_unit": "m", "point_frame": self.device["id"] + ":C_L_rect",
                "fusion": "disabled_no_cross_device_extrinsics",
                "depth_status": "not_supplied" if points is None else "supplied_registered_depth"}


def inspect_devices(config: dict, bundle_root: Path | None) -> tuple[dict, dict]:
    workers, states = {}, {}
    for device in config["devices"]:
        ident = device["id"]
        path = (bundle_root / ident / "calibration_mm.yml" if bundle_root
                else Path(device["calibration_path"]))
        state = {"device_id": ident, "expected_mac": device["expected_mac"],
                 "source_session": device["source_session"],
                 "physical_binding_verified": device["physical_binding_verified"],
                 "binding_note": device["binding_note"],
                 "raw_endpoints": {k: f"http://{device['host']}:{p}/stream"
                                   for k, p in device["raw_ports"].items()},
                 "rectification_service": f"http://{device['host']}:8090",
                 "live_capture": "not_implemented", "ready": False,
                 "cuda": "not_assumed", "depth": "awaiting_registered_depth",
                 "transport_reachable": None, "observed_mac": None}
        try:
            workers[ident] = DeviceWorker(device, config, path)
            state["calibration"] = "loaded_candidate"
            state["calibration_sha256"] = workers[ident].rectifier.sha256
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            state["calibration"] = "error"
            state["error"] = str(exc)
        states[ident] = state
    return workers, states


def make_app(states: dict) -> web.Application:
    app = web.Application(client_max_size=1024)

    async def health(request):
        return web.json_response({"status": "framework_only", "ready": False,
                                  "live_capture": "not_implemented",
                                  "devices": states})

    async def readiness(request):
        return web.json_response({"ready": False, "reason": "live_transport_not_implemented"},
                                 status=503)

    async def device_state(request):
        ident = request.match_info["device_id"]
        if ident not in states:
            raise web.HTTPNotFound()
        return web.json_response(states[ident])

    app.router.add_get("/health", health)
    app.router.add_get("/ready", readiness)
    app.router.add_get("/api/devices/{device_id}", device_state)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("packaging/jetson/devices.json"))
    parser.add_argument("--bundle-root", type=Path,
                        help="Local copied bundles; does not download or activate parameters")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--device", choices=("device_1", "device_2", "device_3"))
    args = parser.parse_args()
    if args.replay and not args.device:
        parser.error("--replay requires --device")
    config = load_config(args.config)
    workers, states = inspect_devices(config, args.bundle_root)
    if args.replay:
        if args.device not in workers:
            parser.error("Selected device calibration is unavailable")
        # CPU work lives outside the status event loop; no CUDA/Open3D requirement.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=args.device) as pool:
            result = pool.submit(workers[args.device].replay, args.replay).result()
        print(json.dumps(result, indent=2))
        return 0
    if args.check:
        print(json.dumps(states, indent=2))
        return int(len(workers) != len(states))
    web.run_app(make_app(states), host=config["listen"], port=config["port"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
