"""Explicit calibration export scope, format and content preview."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import re

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QPlainTextEdit, QVBoxLayout,
)

from ...calibration.models import MultiCameraRig
from ...calibration.multiview import MultiViewCalibrator
from ...io.export import (
    export_rig_json, export_rig_kalibr_yaml, export_rig_opencv_standard_yaml,
    export_rig_opencv_yaml,
    validate_rig_kalibr_chain,
    KalibrDistortionError,
)
from ..theme import TEXT_DIM, WARNING, app_icon


class ExportDialog(QDialog):
    FORMATS = {
        "json": ("JSON", "JSON (*.json)", ".json", export_rig_json),
        "opencv": ("OpenCV 完整 YAML", "OpenCV YAML (*.yml *.yaml)", ".yml", export_rig_opencv_yaml),
        "standard": ("OpenCV 标准 Stereo YAML（仅单组双目）", "Stereo YAML (*.yml *.yaml)",
                     ".yml", export_rig_opencv_standard_yaml),
        "kalibr": ("Kalibr camchain YAML", "Kalibr YAML (*.yaml *.yml)", ".yaml", export_rig_kalibr_yaml),
    }

    def __init__(self, rig: MultiCameraRig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出标定结果")
        self.resize(600, 430)
        self._rig = MultiCameraRig.from_dict(rig.to_dict())
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._scope = QComboBox()
        self._scope.addItem("全部标定结果", ("all", ""))
        for name, pair in self._rig.pairs.items():
            self._scope.addItem(f"双目组：{name}（{pair.left_id} / {pair.right_id}）", ("pair", name))
        ref_pair = next((p for p in self._rig.pairs.values()
                         if p.left_id == self._rig.reference_camera), None)
        if ref_pair is not None:
            for cid in self._rig.aux_cameras:
                self._scope.addItem(f"融合结果：参考 RGB 双目 + {cid}", ("fusion", cid))
        form.addRow("导出对象", self._scope)
        self._format = QComboBox()
        for key, (label, _, _, _) in self.FORMATS.items():
            self._format.addItem(label, key)
        form.addRow("文件格式", self._format)
        layout.addLayout(form)
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMinimumHeight(145)
        self._summary.setAccessibleName("导出内容清单")
        layout.addWidget(self._summary, 1)
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color:{TEXT_DIM};")
        layout.addWidget(self._note)
        self._error = QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color:{WARNING};")
        layout.addWidget(self._error)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Save).setText("选择保存位置...")
        self._buttons.button(QDialogButtonBox.Save).setIcon(app_icon('download'))
        self._buttons.button(QDialogButtonBox.Save).setProperty('class', 'primary')
        self._buttons.button(QDialogButtonBox.Save).setDefault(True)
        self._buttons.button(QDialogButtonBox.Cancel).setText("取消")
        self._buttons.button(QDialogButtonBox.Cancel).setIcon(app_icon('x'))
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._scope.currentIndexChanged.connect(self._scope_changed)
        self._format.currentIndexChanged.connect(self._refresh)
        self._scope_changed()

    @property
    def format_key(self) -> str:
        return self._format.currentData()

    @property
    def file_filter(self) -> str:
        return self.FORMATS[self.format_key][1]

    @property
    def suggested_filename(self) -> str:
        scope, key = self._scope.currentData()
        name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', key).strip(' .')
        suffix = f'_{name}' if name else ''
        return f"calibration_{scope}{suffix}_{self.format_key}{self.file_suffix}"

    @property
    def file_suffix(self) -> str:
        return self.FORMATS[self.format_key][2]

    def selected_rig(self) -> MultiCameraRig:
        scope, key = self._scope.currentData()
        if scope == "all":
            return self._rig
        if scope == "pair":
            pair = self._rig.pairs[key]
        else:
            pair = next(p for p in self._rig.pairs.values() if p.left_id == self._rig.reference_camera)
        # Rebase isolated stereo groups instead of retaining unrelated global extrinsics.
        calibrator = MultiViewCalibrator(reference_camera=pair.left_id)
        calibrator.add_pair_calibration(pair)
        rig = calibrator.calibrate()
        if scope == "fusion":
            rig.aux_cameras[key] = self._rig.aux_cameras[key]
        return rig

    def _scope_changed(self):
        pair_scope = self._scope.currentData()[0] == "pair"
        index = self._format.findData("standard")
        self._format.model().item(index).setEnabled(pair_scope)
        self._format.model().item(index).setToolTip('先在“导出对象”中选择具体双目组' if not pair_scope else '')
        if self.format_key == "standard" and not pair_scope:
            self._format.setCurrentIndex(self._format.findData("opencv"))
        self._refresh()

    def _refresh(self):
        rig = self.selected_rig()
        included_ids = list(dict.fromkeys(
            [cid for pair in rig.pairs.values() for cid in (pair.left_id, pair.right_id)] +
            list(rig.aux_cameras)
        ))
        included = set(included_ids)
        available = {cid for pair in self._rig.pairs.values() for cid in (pair.left_id, pair.right_id)}
        available.update(self._rig.aux_cameras)
        self._summary.setPlainText(
            f"对象：{self._scope.currentText()}\n参考相机：{rig.reference_camera}\n"
            f"包含相机：{'、'.join(included_ids)}\n"
            f"双目参数：{len(rig.pairs)} 组；辅助标定：{len(rig.aux_cameras)} 路\n"
            f"不包含：{'、'.join(sorted(available - included)) or '无'}\n"
            f"建议文件名：{self.suggested_filename}"
        )
        notes = {
            "json": "包含所选对象的内参、外参及质量指标。",
            "opencv": "保留所选对象的项目完整字段；包含辅助相机时写入 AUX→RGB 左目变换。",
            "standard": "仅包含所选双目组的 M1/D1/M2/D2/R/T 等标准字段，不包含任何红外/辅助参数。",
            "kalibr": "输出可表达的相机链式参数；非零针孔高阶畸变不能无损转换。相机之间须有相对外参，不等同 Kalibr 原生采集流程。",
        }
        note = notes[self.format_key]
        if self._scope.currentData()[0] == "fusion":
            note += " 融合导出保留参考 RGB 双目基础参数（含 RGB 右目），跨模态配对仍为 RGB 左目与辅助相机。"
        if self.format_key == 'standard' and any(pair.left_intrinsics.model.value == 'fisheye' for pair in rig.pairs.values()):
            note += ' 当前为鱼眼模型，第三方必须按 fisheye/equidistant 解释畸变，不能按普通针孔处理。'
        self._note.setText(note)
        problem = ""
        if self.format_key == "standard" and self._scope.currentData()[0] != "pair":
            problem = "标准 Stereo 格式必须明确选择一组双目。"
        if self.format_key == "kalibr":
            try:
                validate_rig_kalibr_chain(rig)
            except KalibrDistortionError:
                problem = '当前高阶畸变无法无损转换为 Kalibr radtan；请改用 JSON/OpenCV，或核实后重新进行约束标定。'
            except ValueError as exc:
                problem = f"Kalibr 参数不兼容或外参链不完整，请核对导出对象：{exc}"
        self._error.setText(problem)
        self._buttons.button(QDialogButtonBox.Save).setEnabled(not problem)

    def export_to(self, path: str | Path) -> Path:
        if self.format_key == "standard" and self._scope.currentData()[0] != "pair":
            raise ValueError("标准 Stereo 格式必须明确选择一组双目")
        if self.format_key == "kalibr":
            validate_rig_kalibr_chain(self.selected_rig())
        path = Path(path)
        allowed = {".json"} if self.format_key == "json" else {".yml", ".yaml"}
        if path.suffix.lower() not in allowed:
            raise ValueError(f"文件扩展名与所选格式不一致，请使用 {' / '.join(sorted(allowed))}")
        # A failed writer must not truncate a previously exported calibration file.
        with TemporaryDirectory(prefix=".multicamera-export-", dir=path.parent, ignore_cleanup_errors=True) as temp:
            staged = Path(temp) / path.name
            self.FORMATS[self.format_key][3](self.selected_rig(), staged)
            if not staged.is_file() or not staged.stat().st_size:
                raise OSError("导出未生成有效文件")
            os.replace(staged, path)
        return path
