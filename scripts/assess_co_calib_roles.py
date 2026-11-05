"""Supplement the frozen selection ablation with independent role allocation.

Uses upstream default numeric gates and covisible-first allocation on the same
paired ChArUco cache. No NN detector, AprilTag tie-break or Kalibr reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

import assess_co_calib_selection as co


def role_indices(metrics):
    audit = co.upstream_roles(metrics)
    paired = audit["covisible_indices"]
    mono, anchors, fills, targets = [], [], [], []
    for camera in range(2):
        selected = set(paired)
        anchor = sorted((i for i in audit["anchor_eligible_per_camera"][camera] if i not in selected),
                        key=lambda i: (-metrics[i][camera]["span"],
                                       -metrics[i][camera]["iso"], i))
        selected.update(anchor)
        counts = np.zeros(48, int)
        for i in selected:
            counts[metrics[i][camera]["cells"]] += 1
        eligible = [i for i, pair in enumerate(metrics) if pair[camera]["span"] >= .2
                    and pair[camera]["iso"] >= .3 and pair[camera]["points"] >= 12]
        seed_mean = np.mean([len(metrics[i][camera]["cells"]) for i in selected]) if selected else 0
        cand_mean = np.mean([len(metrics[i][camera]["cells"]) for i in eligible]) if eligible else 0
        target = min(8, int(np.ceil(max(1., .65 * seed_mean, .25 * cand_mean))))
        added = []
        while True:
            remaining = [i for i in eligible if i not in selected]
            def score(i):
                occupied = counts[metrics[i][camera]["cells"]]
                gain = int(np.sum(occupied == 0) * 2 + np.sum((occupied > 0) & (occupied < target)))
                return (gain, int(np.sum(occupied == 0)), metrics[i][camera]["span"],
                        metrics[i][camera]["iso"], -i)
            if not remaining:
                break
            best = max(remaining, key=score)
            if score(best)[0] <= 0:
                break
            selected.add(best)
            added.append(best)
            counts[metrics[best][camera]["cells"]] += 1
        mono.append(sorted(selected))
        anchors.append(anchor)
        fills.append(added)
        targets.append(target)
    return {"stereo": paired, "mono": mono, "anchors": anchors, "fills": fills,
            "grid_targets": targets}


def solve_roles(data, train, holdout, metrics, allocation, model):
    started = time.perf_counter()
    phase = "input"
    try:
        left = co.calibrators(data, [train[i] for i in allocation["mono"][0]], model)[0]
        right = co.calibrators(data, [train[i] for i in allocation["mono"][1]], model)[1]
        stereo = co.calibrators(data, [train[i] for i in allocation["stereo"]], model)[2]
        if model == co.CameraModel.FISHEYE:
            stereo = stereo.copy_with_common_ids()
        phase = "left_intrinsics"
        li = left.calibrate()
        phase = "right_intrinsics"
        ri = right.calibrate()
        phase = "stereo"
        result = stereo.calibrate(li, ri)
        phase = "evaluation"
        evaluated = co.evaluate(data, holdout, li, ri, result, left.board)
        answer = {"status": "ok", "evaluation": evaluated,
                  "train_rms_px": result.rms_error,
                  "shared_stereo_ids": stereo.frame_corner_ids[0].tolist()}
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        answer = {"status": "failed", "phase": phase, "error": str(exc)}
    answer["seconds"] = time.perf_counter() - started
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.baseline / "report.json").read_text())
    output = args.baseline / "roles_report.json"
    if output.exists():
        raise ValueError("Refusing to replace existing role report")
    result = {"source_report_sha256": co.digest(args.baseline / "report.json"),
              "script_sha256": co.digest(__file__), "helper_sha256": co.digest(co.__file__),
              "limits": ["Upstream-inspired ChArUco paired-cache adaptation, not exact reproduction",
                         "No equal-budget claim for role allocation versus fixed-budget rankings",
                         "Historical reused holdouts; no independent 3-D truth"], "datasets": {}}
    for name, directory in [("test1", Path("data/results/mrcal_test1_retrained_verified")),
                            ("historical", Path("data/results/mrcal_opencv5_final")),
                            ("pairs_fisheye", None)]:
        data = json.loads(((directory / "observations.json") if directory
                           else args.baseline / "pairs_observations.json").read_text())
        previous = report["datasets"][name]
        train = [r for r in data["records"] if r["capture_id"] in previous["split"]["train"]
                 and r["usable"]]
        holdout = [r for r in data["records"] if r["capture_id"] in previous["split"]["validation"]
                   and r["usable"]]
        metrics = [previous["frame_metrics"][r["capture_id"]] for r in train]
        allocation = role_indices(metrics)
        row = {"allocation": allocation,
               "counts": {"left": len(allocation["mono"][0]), "right": len(allocation["mono"][1]),
                          "stereo": len(allocation["stereo"]),
                          "unique_pairs": len(set().union(*map(set, allocation["mono"])))},
               "models": {}, "fisheye_intersection_audit": []}
        for model in co.CameraModel:
            row["models"][model.value] = solve_roles(data, train, holdout, metrics, allocation, model)
        objects = np.array(data["object_points_m"])
        # Effective geometry after cross-frame ID intersection, without changing fits.
        for run in previous["models"]["fisheye"]:
            if run["status"] != "ok" or run["policy"].startswith("random"):
                continue
            common_ids = np.array(run["shared_stereo_ids"])
            spans = [[], []]
            for record in train:
                if record["capture_id"] not in run["selected_ids"]:
                    continue
                _, indices, _ = np.intersect1d(record["ids"], common_ids, return_indices=True)
                for camera, side in enumerate(("left", "right")):
                    effective = co.frame_metrics(np.asarray(record[side])[indices], objects[common_ids],
                                                 data["image_size"])
                    spans[camera].append(effective["span"])
            row["fisheye_intersection_audit"].append({"policy": run["policy"], "budget": run["budget"],
                "retained_ids": len(common_ids), "span_median_after": [float(np.median(s)) for s in spans]})
        result["datasets"][name] = row
        co.write(output, result)
        print(name, row["counts"], {m: x["status"] for m, x in row["models"].items()}, flush=True)


if __name__ == "__main__":
    main()
