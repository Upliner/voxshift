"""Desktop control panel for the realtime pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from . import audio
from .engines.base import ENGINES
from .session import Options, build_engine, start_session
from .voices import list_voices


class Loader(QThread):
    """Loads models off the UI thread; this takes several seconds."""

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, opts: Options):
        super().__init__()
        self.opts = opts

    def run(self) -> None:
        try:
            self.loaded.emit(build_engine(self.opts))
        except (Exception, SystemExit) as e:
            self.failed.emit(str(e))


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxShift")
        self.pipeline = None
        self.loader = None
        self.created_mic = False

        self.engine = QComboBox()
        self.engine.addItems(ENGINES)
        self.voice = QComboBox()
        self.voice.addItems([v.name for v in list_voices()])
        self.mic = QComboBox()
        self.mic.addItem("System default", None)
        self.output = QComboBox()
        self.output.addItem("System default", None)
        for d in audio.list_devices():
            if d.name != audio.VIRTUAL_MIC:
                (self.mic if d.kind == "source" else self.output).addItem(d.description, d.name)

        self.pitch = QDoubleSpinBox(minimum=-24, maximum=24, singleStep=0.5, value=12, suffix=" st")
        self.pitch.valueChanged.connect(self.pitch_changed)
        self.gate = QDoubleSpinBox(minimum=-80, maximum=0, singleStep=1, value=-45, suffix=" dB")

        self.monitor = QCheckBox("Play through output (use headphones)")
        self.virtual_mic = QCheckBox("Virtual microphone for other apps")
        self.virtual_mic.setChecked(True)
        self.record = QCheckBox("Record to file")
        self.record_path = QLineEdit(str(Path.home() / "voxshift.flac"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self.choose_file)

        record_row = QHBoxLayout()
        record_row.addWidget(self.record)
        record_row.addWidget(self.record_path)
        record_row.addWidget(browse)

        self.start = QPushButton("Start")
        self.start.clicked.connect(self.toggle)
        self.status = QLabel("Stopped")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        form = QFormLayout()
        form.addRow("Engine", self.engine)
        form.addRow("Voice", self.voice)
        form.addRow("Microphone", self.mic)
        form.addRow("Output", self.output)
        form.addRow("Pitch", self.pitch)
        form.addRow("Noise gate", self.gate)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.monitor)
        layout.addWidget(self.virtual_mic)
        layout.addLayout(record_row)
        layout.addWidget(self.start)
        layout.addWidget(self.status)

        self.timer = QTimer(interval=500)
        self.timer.timeout.connect(self.refresh)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Record to", self.record_path.text(),
                                              "Audio (*.flac *.wav *.ogg)")
        if path:
            self.record_path.setText(path)

    def options(self) -> Options:
        return Options(engine=self.engine.currentText(), voice=self.voice.currentText() or None,
                       pitch=self.pitch.value(), mic=self.mic.currentData(),
                       monitor=self.monitor.isChecked(), monitor_device=self.output.currentData(),
                       virtual_mic=self.virtual_mic.isChecked(),
                       record=Path(self.record_path.text()) if self.record.isChecked() else None,
                       gate_db=self.gate.value())

    def toggle(self) -> None:
        if self.pipeline:
            self.stop()
            return
        self.start.setEnabled(False)
        self.status.setText("Loading engine...")
        self.loader = Loader(self.options())
        self.loader.loaded.connect(self.engine_ready)
        self.loader.failed.connect(self.show_error)
        self.loader.start()

    def engine_ready(self, engine) -> None:
        try:
            self.pipeline, self.created_mic = start_session(engine, self.options())
        except (Exception, SystemExit) as e:
            self.show_error(str(e))
            return
        self.start.setEnabled(True)
        self.start.setText("Stop")
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        pipeline, self.pipeline = self.pipeline, None
        pipeline.stop()
        if self.created_mic:
            audio.remove_virtual_mic()
        self.start.setText("Start")
        self.status.setText("Stopped")
        if pipeline.error:
            self.show_error(str(pipeline.error))

    def show_error(self, message: str) -> None:
        self.start.setEnabled(True)
        self.status.setText("Stopped")
        QMessageBox.critical(self, "VoxShift", message)

    def pitch_changed(self, value: float) -> None:
        if self.pipeline:
            self.pipeline.engine.set_pitch(value)

    def refresh(self) -> None:
        if not self.pipeline:
            return
        if not self.pipeline.running:
            self.stop()
            return
        s = self.pipeline.stats
        engine = self.pipeline.engine
        self.status.setText(f"{engine.name}: {engine.algorithmic_latency_ms:.0f} ms + inference "
                            f"{s.last_ms:.0f} ms · overruns {s.overruns} · "
                            f"in {s.input_db:.0f} dB · out {s.output_db:.0f} dB")

    def closeEvent(self, event) -> None:
        if self.pipeline:
            self.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()
