"""Read-only, source-ID based review of a completed stereo solve."""

from __future__ import annotations

import math

from .assessment import CalibrationEvaluator
from .models import StereoPairCalibration


def _valid_error(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError, OverflowError):
        return False


def review_pair_frames(pair: StereoPairCalibration) -> dict:
    """Suggest review, never delete observations or claim independent validation."""
    issues, exclusions, limitations, thresholds = {}, {}, [], {}
    evaluated = set()

    def issue(frame_id, reason):
        issues.setdefault(str(frame_id), []).append(reason)

    for side, label, intr in (("left", "左内参", pair.left_intrinsics),
                              ("right", "右内参", pair.right_intrinsics)):
        ids, errors = intr.frame_ids, intr.per_frame_errors
        if ids is None or errors is None or len(ids) != len(errors):
            limitations.append(f"{label}缺少可靠源编号或完整逐帧误差，无法逐对定位")
            continue
        if len(set(map(str, ids))) != len(ids) or any(value is None for value in ids):
            limitations.append(f"{label}源编号缺失或重复，无法逐对定位")
            continue
        if intr.per_frame_error_metric != "euclidean_rms_px":
            limitations.append(f"{label}逐帧误差口径未确认，未按 RMS 阈值判断")
            continue
        absolute = CalibrationEvaluator.INTRINSIC_THRESHOLDS[intr.model][-1]
        relative = max(.1, 2.5 * float(intr.rms_error)) if _valid_error(intr.rms_error) else None
        if not _valid_error(relative):
            relative = None
        thresholds[side] = {"absolute_rms_px": absolute, "relative_rms_px": relative}
        for frame_id, error in zip(ids, errors):
            evaluated.add(str(frame_id))
            if not _valid_error(error):
                issue(frame_id, f"{label}逐帧 RMS 无效")
                continue
            error = float(error)
            if float(error) > absolute:
                issue(frame_id, f"{label} RMS={error:.4f}px > 复核阈值 {absolute:.4f}px")
            elif relative is not None and float(error) > relative:
                issue(frame_id, f"{label} RMS={error:.4f}px > 相对复核阈值 {relative:.4f}px"
                      "（整体 RMS 的2.5倍，最低0.1px）")
        if not len(ids):
            limitations.append(f"{label}无逐帧记录，无法逐对定位")

    alignment = pair.calibration_info.get("rectification", {})
    records = alignment.get("per_frame", [])
    if records:
        thresholds["rectification_rms_px"] = 1.0
        for record in records:
            frame_id = record.get("frame_id")
            if frame_id is None:
                limitations.append("校正残差缺少源编号，无法逐对定位")
                continue
            evaluated.add(str(frame_id))
            error = record.get("rms_px")
            if record.get("status") != "ok" or not _valid_error(error):
                issue(frame_id, "同 ID 校正残差无效")
            elif float(error) > 1.:
                issue(frame_id, f"同 ID 校正残差 RMS={float(error):.4f}px > 复核阈值 1.0000px"
                      "（受校正像素尺度影响）")
    else:
        limitations.append("未记录逐对同 ID 校正残差，未完成该项检查")

    selection = pair.calibration_info.get("selection", {})
    for side, label in (("left", "左内参"), ("right", "右内参"), ("stereo", "双目外参")):
        for frame_id in selection.get(side, {}).get("excluded_ids", []):
            exclusions.setdefault(str(frame_id), []).append(label)

    def records_for(values, key):
        return [{"frame_id": value, key: values[value]}
                for value in sorted(values, key=lambda value: (0, int(value), value)
                                    if value.isdecimal() else (1, 0, value))]

    return {
        "schema_version": 1,
        "dataset": "training_observations",
        "evaluated_pair_count": len(evaluated),
        "thresholds": thresholds,
        "review_pairs": records_for(issues, "reasons"),
        "excluded_pairs": records_for(exclusions, "stages"),
        "limitations": list(dict.fromkeys(limitations)),
    }


def frame_review_log_lines(pair_name: str, report: dict, sources: dict) -> list[str]:
    """Format one line per source pair for both the UI and persistent log."""
    reviews, exclusions = report["review_pairs"], report["excluded_pairs"]
    lines = [f"{pair_name}: 标定后逐对检查：已检查 {report['evaluated_pair_count']} 对，"
             f"建议复核 {len(reviews)} 对，筛选未选用 {len(exclusions)} 对"
             "（训练集，非独立验证）"]

    def source_text(frame_id):
        paths = sources.get(frame_id)
        return f"L={paths[0]}；R={paths[1]}" if paths else "原图路径未记录，请按源编号查找"

    for record in reviews:
        frame_id = record["frame_id"]
        lines.append(f"{pair_name} [{frame_id}]: 建议复核 pair — "
                     + "；".join(record["reasons"]) + "；" + source_text(frame_id))
    for record in exclusions:
        frame_id = record["frame_id"]
        lines.append(f"{pair_name} [{frame_id}]: 筛选未选用 — "
                     + "、".join(record["stages"]) + "；策略取舍不等于图像不合格；"
                     + source_text(frame_id))
    for limitation in report["limitations"]:
        lines.append(f"{pair_name}: 逐对检查范围提示 — {limitation}")
    if reviews:
        lines.append(f"{pair_name}: 请在“逐帧误差检查”按源编号复核原图、配对及检测；"
                     "需要剔除时调整输入后重新标定。本次未自动剔除上述疑似异常 pair。")
    elif report["evaluated_pair_count"]:
        lines.append(f"{pair_name}: 已记录指标中未发现超过复核阈值的 pair；不代表全部图像或实际精度合格。")
    return lines
