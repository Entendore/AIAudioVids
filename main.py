import sys
import os
import json
import time
import shutil
import logging
import math
import gc
import threading
import inspect
import signal
import wave
import random
import traceback
import faulthandler
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from collections import deque
from io import BytesIO

CWD = os.getcwd()
FAULT_LOG_FILE = os.path.join(CWD, "AudioLDM2_Studio.log")
try:
    fault_log_file = open(FAULT_LOG_FILE, "w", buffering=1)
    faulthandler.enable(file=fault_log_file)
except Exception as e:
    print(f"Could not enable faulthandler: {e}")

import numpy as np

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(FAULT_LOG_FILE, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("AudioLDM2_Studio")

# Audio Visualization Imports
try:
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# Qt Imports
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QUrl, QTimer, QPointF
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSlider, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter,
    QDialog, QFormLayout, QMenu, QPlainTextEdit, QSizePolicy, QFrame,
    QStatusBar, QScrollArea, QListWidget, QListWidgetItem, QTabWidget
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import (
    QColor, QPalette, QFont, QPainter, QPen, QPixmap,
    QMouseEvent, QKeySequence, QShortcut, QLinearGradient, QPolygonF, QAction
)

# ML Imports
try:
    import torch
    from diffusers import AudioLDM2Pipeline
    import transformers
    import diffusers
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    torch = None
    AudioLDM2Pipeline = None

# --- EXPERT PATCH 2: Force language_model to be GPT2LMHeadModel ---
try:
    if ML_AVAILABLE:
        from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
        import torch.nn as nn

        _orig_fp = AudioLDM2Pipeline.from_pretrained

        @classmethod
        def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            cache_dir = kwargs.get("cache_dir")
            torch_dtype = kwargs.get("torch_dtype")
            use_safetensors = kwargs.get("use_safetensors", True)

            pipe = _orig_fp.__func__(
                cls, pretrained_model_name_or_path, *args, **kwargs
            )

            lm = getattr(pipe, "language_model", None)
            if lm is not None and isinstance(lm, GPT2Model) and not isinstance(lm, GPT2LMHeadModel):
                logger.info(
                    "AudioLDM2: language_model is GPT2Model (base). "
                    "Reloading as GPT2LMHeadModel to restore .generate()."
                )
                try:
                    reload_kwargs = {
                        "subfolder": "language_model",
                        "cache_dir": cache_dir,
                    }
                    if torch_dtype is not None:
                        reload_kwargs["torch_dtype"] = torch_dtype
                    new_lm = GPT2LMHeadModel.from_pretrained(
                        pretrained_model_name_or_path, **reload_kwargs
                    )
                    if hasattr(lm, "device"):
                        new_lm = new_lm.to(lm.device)
                    pipe.language_model = new_lm
                    logger.info("AudioLDM2: GPT2LMHeadModel loaded successfully with original weights.")
                    return pipe
                except Exception as e_reload:
                    logger.warning(
                        f"AudioLDM2: Could not reload GPT2LMHeadModel from checkpoint ({e_reload}). "
                        f"Falling back to in-memory wrapper (lm_head will be random)."
                    )
                try:
                    new_lm = GPT2LMHeadModel(lm.config)
                    new_lm.transformer = lm
                    new_lm = new_lm.to(lm.device)
                    if torch_dtype is not None:
                        new_lm = new_lm.to(torch_dtype)
                    pipe.language_model = new_lm
                    logger.warning(
                        "AudioLDM2: Using GPT2LMHeadModel wrapper. "
                        "lm_head is random — text conditioning may be degraded, "
                        "but generation will not crash."
                    )
                except Exception as e_wrap:
                    logger.error(f"AudioLDM2: Wrapper fallback failed: {e_wrap}")

            return pipe

        AudioLDM2Pipeline.from_pretrained = _patched_from_pretrained
        logger.info("✅ Expert patch 2 applied: GPT2Model → GPT2LMHeadModel conversion.")
except Exception as e:
    logger.error(f"⚠️ Could not apply GPT2LMHeadModel conversion patch: {e}")
# ------------------------------------------------------------------------

# --- Configuration & Constants ---
APP_NAME = "AudioLDM2 Studio"
APP_VERSION = "6.1.0-unified-visualizer"
SETTINGS_ORG = "AudioLDM2"
SETTINGS_APP = "Studio"
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), "model_cache")
SETTINGS_FILE = os.path.join(CWD, "config.json")

DEFAULT_MODELS = [
    "cvssp/audioldm2",
    "cvssp/audioldm2-music",
    "cvssp/audioldm2-gigaspeech",
    "cvssp/audioldm2-ljspeech",
]
DEFAULT_PROMPT_PRESETS = [
    "Cinematic riser and massive impact, sub bass drop",
    "Sci-fi UI confirmation beep, clean digital tone",
    "Distant thunder rolling across a valley, rain ambience",
    "Continuous ambient forest soundscape, birds, wind",
    "Evolving dark drone, low frequency rumble",
]

MAX_DURATION_S = 86400.0
LONG_FORM_THRESHOLD_S = 60.0
AUTO_CHUNK_SIZE_S = 30.0
OVERLAP_S = 2.0
MIN_CHUNK_S = 5.0

MAX_HISTORY = 500
MAX_PROMPT_HISTORY = 50
FADE_DURATION_S = 0.05
TRIM_THRESHOLD = 0.01
ETA_SMOOTHING_ALPHA = 0.3
PROGRESS_THROTTLE_S = 0.033  # ~30 FPS cap on UI updates

STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QGroupBox { border: 1px solid #3c3c3c; border-radius: 4px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #cccccc; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; background-color: #2b2b2b; }
QPushButton { background-color: #3c3c3c; border: 1px solid #4a4a4a; color: #e0e0e0; padding: 6px 12px; border-radius: 3px; min-height: 20px; }
QPushButton:hover { background-color: #4a4a4a; border-color: #2A82DA; }
QPushButton:pressed { background-color: #2b2b2b; }
QPushButton:disabled { background-color: #333333; color: #666666; border-color: #333333; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background-color: #333333; border: 1px solid #3c3c3c; padding: 4px 6px; border-radius: 2px; color: #e0e0e0; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus { border: 1px solid #2A82DA; }
QComboBox QAbstractItemView { background-color: #333333; border: 1px solid #3c3c3c; selection-background-color: #2A82DA; color: #e0e0e0; }
QTableWidget { background-color: #2b2b2b; alternate-background-color: #333333; border: 1px solid #3c3c3c; gridline-color: #3c3c3c; }
QHeaderView::section { background-color: #3c3c3c; padding: 6px; border: none; font-weight: bold; color: #cccccc; }
QProgressBar { border: 1px solid #3c3c3c; border-radius: 2px; text-align: center; background-color: #333333; color: white; min-height: 22px; }
QProgressBar::chunk { background-color: #2A82DA; border-radius: 1px; }
QScrollBar:vertical { background: #2b2b2b; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 20px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #2A82DA; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox { color: #e0e0e0; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 2px; border: 1px solid #4a4a4a; background: #333333; }
QCheckBox::indicator:checked { background: #2A82DA; border-color: #2A82DA; }
QLabel { color: #d0d0d0; }
QStatusBar { background-color: #2b2b2b; color: #aaaaaa; }
QSlider::groove:horizontal { border: 1px solid #3c3c3c; height: 4px; background: #333333; border-radius: 2px; }
QSlider::handle:horizontal { background: #2A82DA; border: 1px solid #2A82DA; width: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #3a92ea; }
QSplitter::handle { background-color: #3c3c3c; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QListWidget { background-color: #2b2b2b; border: 1px solid #3c3c3c; color: #e0e0e0; }
QListWidget::item { padding: 4px 6px; }
QListWidget::item:selected { background-color: #2A82DA; color: white; }
QTabWidget::pane { border: 1px solid #3c3c3c; background-color: #2b2b2b; }
QTabBar::tab { background-color: #333333; color: #aaaaaa; padding: 8px 16px; border: 1px solid #3c3c3c; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background-color: #2b2b2b; color: #2A82DA; border-bottom: 2px solid #2A82DA; }
QTabBar::tab:hover { background-color: #3c3c3c; }
"""


# --- Helpers ---
def sanitize_filename(text: str, max_length: int = 30) -> str:
    safe = "".join(c for c in (text or "") if c.isalnum() or c in (' ', '_', '-')).rstrip()
    return safe[:max_length].replace(' ', '_') if safe else "audio"


def format_time(seconds) -> str:
    try:
        total_secs = int(float(seconds))
        if total_secs < 0:
            total_secs = 0
        minutes, secs = divmod(total_secs, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"
    except Exception:
        return "00:00"


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    if audio.size == 0:
        return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak == 0:
        return audio
    target_peak = 10 ** (target_db / 20.0)
    return audio * (target_peak / peak)


def apply_fade(audio: np.ndarray, sample_rate: int, fade_in_s: float = 0.05, fade_out_s: float = 0.05) -> np.ndarray:
    audio = audio.astype(np.float32, copy=True)
    if audio.size == 0:
        return audio
    fade_in_samples = int(fade_in_s * sample_rate)
    fade_out_samples = int(fade_out_s * sample_rate)
    if fade_in_samples > 0:
        fade_in_samples = min(fade_in_samples, audio.size)
        audio[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
    if fade_out_samples > 0:
        fade_out_samples = min(fade_out_samples, audio.size)
        audio[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
    return audio


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if audio.size == 0:
        return audio.copy()
    audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    above_threshold = np.where(np.abs(audio) > threshold)[0]
    return audio[above_threshold[0]:above_threshold[-1] + 1] if above_threshold.size > 0 else audio


SEED_ALGORITHMS = {"Sequential", "Random", "Fixed", "Golden Ratio"}


def parse_multi_prompt_line(line: str, default_seed_algo: str, default_duration: float) -> dict:
    parts = [p.strip() for p in line.split('|')]
    prompt = parts[0].strip()
    seed_algo = default_seed_algo
    duration = default_duration

    for idx in range(1, min(len(parts), 3)):
        val = parts[idx]
        if not val:
            continue
        if val in SEED_ALGORITHMS:
            seed_algo = val
        else:
            try:
                d = float(val)
                if d > 0:
                    duration = min(d, MAX_DURATION_S)
            except ValueError:
                pass
    return {"prompt": prompt, "seed_algo": seed_algo, "duration": duration}


def compute_chunks_for_duration(duration: float) -> list:
    if duration <= LONG_FORM_THRESHOLD_S:
        return [max(MIN_CHUNK_S, duration)]

    effective = AUTO_CHUNK_SIZE_S - OVERLAP_S
    n = math.ceil((duration - OVERLAP_S) / effective)
    last_len = duration - (n - 1) * effective
    if last_len < MIN_CHUNK_S and n > 1:
        n -= 1

    chunks = []
    for i in range(n):
        if i == n - 1:
            cl = max(MIN_CHUNK_S, duration - (n - 1) * effective)
        else:
            cl = AUTO_CHUNK_SIZE_S
        chunks.append(cl)
    return chunks


@dataclass(slots=True)
class GenerationParams:
    prompt: str
    negative_prompt: str
    model_name: str
    duration: float
    steps: int
    guidance: float
    seed: int
    device: str
    cache_dir: str
    output_dir: str
    num_variations: int = 1
    normalize: bool = True
    fade_in: bool = True
    fade_out: bool = True
    trim: bool = True
    use_cpu_offload: bool = False
    section: str = ""
    use_multi_prompt: bool = False
    multi_prompts: str = ""
    seed_algo: str = "Sequential"


# =============================================================================
# Audio Generation Thread
# =============================================================================

class AudioGenerationThread(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    eta_updated = Signal(str)
    generation_completed = Signal(str, dict)
    error_occurred = Signal(str)
    cancelled = Signal()

    _shared_pipe = None
    _shared_pipe_model = None
    _shared_pipe_device = None
    _shared_pipe_offload = None
    _pipe_lock = threading.Lock()
    _uses_new_callback: bool = True

    def __init__(self, params: GenerationParams, parent=None) -> None:
        super().__init__(parent)
        self.params = params
        self._cancel_event = threading.Event()
        self.ema_step_time = None
        self.last_step_time = None
        self._last_progress_ts = 0.0
        self.pipe = None
        self.setObjectName("AudioGenerationThread")

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self):
        self._cancel_event.set()

    def _load_pipeline(self, actual_device: str):
        if self.is_cancelled:
            raise InterruptedError("User cancelled")

        with AudioGenerationThread._pipe_lock:
            if (AudioGenerationThread._shared_pipe is not None
                and AudioGenerationThread._shared_pipe_model == self.params.model_name
                and AudioGenerationThread._shared_pipe_device == actual_device
                and AudioGenerationThread._shared_pipe_offload == self.params.use_cpu_offload):
                self.pipe = AudioGenerationThread._shared_pipe
                return

            self._unload_pipeline_unsafe()
            self.status_updated.emit("Loading model weights...")
            self.progress_updated.emit(5)

            if self.is_cancelled:
                raise InterruptedError("User cancelled")

            dtype = torch.float16 if actual_device != "cpu" else torch.float32
            load_kwargs = {"torch_dtype": dtype, "cache_dir": self.params.cache_dir}
            if dtype == torch.float16:
                load_kwargs["variant"] = "fp16"

            pipe = None
            try:
                pipe = AudioLDM2Pipeline.from_pretrained(
                    self.params.model_name, **load_kwargs, use_safetensors=True
                )
            except Exception as e_outer:
                if self.is_cancelled:
                    raise InterruptedError("User cancelled") from e_outer
                logger.warning(f"fp16/safetensors load failed, retrying default: {e_outer}")
                load_kwargs.pop("variant", None)
                if self.is_cancelled:
                    raise InterruptedError("User cancelled")
                try:
                    pipe = AudioLDM2Pipeline.from_pretrained(
                        self.params.model_name, **load_kwargs, use_safetensors=False
                    )
                except Exception as e_inner:
                    raise RuntimeError(f"Failed to load model {self.params.model_name}: {e_inner}") from e_inner

            if self.is_cancelled:
                del pipe
                raise InterruptedError("User cancelled")

            if self.params.use_cpu_offload and actual_device != "cpu":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(actual_device)

            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()

            try:
                sig = inspect.signature(pipe.__call__)
                AudioGenerationThread._uses_new_callback = "callback_on_step_end" in sig.parameters
            except Exception:
                AudioGenerationThread._uses_new_callback = True

            AudioGenerationThread._shared_pipe = pipe
            AudioGenerationThread._shared_pipe_model = self.params.model_name
            AudioGenerationThread._shared_pipe_device = actual_device
            AudioGenerationThread._shared_pipe_offload = self.params.use_cpu_offload
            self.pipe = pipe

    def _on_step(self, *args, chunk_idx: int = 0, **kwargs):
        if self.is_cancelled:
            raise InterruptedError("User cancelled")

        if len(args) >= 4:
            step = int(args[1])
            callback_kwargs = args[3]
            self._update_progress_and_eta(step, chunk_idx)
            return callback_kwargs
        elif len(args) >= 3:
            step = int(args[0])
            self._update_progress_and_eta(step, chunk_idx)
            return None
        return None

    def _update_progress_and_eta(self, step: int, chunk_idx: int):
        p = self.params
        num_chunks_int = int(self._num_chunks)
        steps_done = self._steps_done_before_variation + (chunk_idx * p.steps) + (step + 1)
        total_progress = 15 + int(80 * steps_done / max(self._total_steps_all, 1))

        now = time.time()
        is_final = (step + 1 >= p.steps and chunk_idx + 1 >= num_chunks_int)

        if is_final or (now - self._last_progress_ts) >= PROGRESS_THROTTLE_S:
            self._last_progress_ts = now
            self.progress_updated.emit(min(total_progress, 95))

        if self.last_step_time is not None:
            dt = now - self.last_step_time
            if dt > 0:
                self.ema_step_time = (
                    dt if self.ema_step_time is None
                    else ETA_SMOOTHING_ALPHA * dt + (1 - ETA_SMOOTHING_ALPHA) * self.ema_step_time
                )
                remaining_steps = max(0, p.steps - step - 1)
                remaining_chunks = max(0, num_chunks_int - chunk_idx - 1)
                remaining_vars = max(0, self._total_variations - self._variation - 1)
                eta_seconds = (
                    remaining_steps + remaining_chunks * p.steps
                    + remaining_vars * num_chunks_int * p.steps
                ) * self.ema_step_time
                if is_final or (now - self._last_progress_ts) >= PROGRESS_THROTTLE_S:
                    self.eta_updated.emit(format_time(eta_seconds))
        self.last_step_time = now
        self.status_updated.emit(
            f"Var {self._variation + 1}/{self._total_variations} | "
            f"Chunk {chunk_idx + 1}/{num_chunks_int} | Step {step + 1}/{p.steps}"
        )

    def run(self):
        wav_writer = None
        gen_start_time = time.time()
        try:
            p = self.params
            cache_dir = os.path.normpath(p.cache_dir or DEFAULT_CACHE_DIR)
            os.makedirs(cache_dir, exist_ok=True)
            final_out_dir = p.output_dir
            if p.section:
                final_out_dir = os.path.join(p.output_dir, sanitize_filename(p.section, 50))
                os.makedirs(final_out_dir, exist_ok=True)

            os.environ["HF_HOME"] = cache_dir
            actual_device = p.device
            if actual_device == "auto":
                if torch.cuda.is_available():
                    actual_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    actual_device = "mps"
                else:
                    actual_device = "cpu"

            self._load_pipeline(actual_device)
            self.status_updated.emit("Generating audio...")
            self.progress_updated.emit(15)

            use_multi = p.use_multi_prompt
            prompt_configs = []
            if use_multi and p.multi_prompts:
                for line in p.multi_prompts.split('\n'):
                    line = line.strip()
                    if line:
                        cfg = parse_multi_prompt_line(line, p.seed_algo, p.duration)
                        prompt_configs.append(cfg)
            if not prompt_configs:
                prompt_configs = [{"prompt": p.prompt, "seed_algo": p.seed_algo, "duration": p.duration}]

            has_custom_durations = any(
                abs(cfg["duration"] - p.duration) > 0.01 for cfg in prompt_configs
            )

            chunk_plan = []

            if has_custom_durations:
                for cfg in prompt_configs:
                    prompt_dur = cfg["duration"]
                    chunk_lens = compute_chunks_for_duration(prompt_dur)
                    for cl in chunk_lens:
                        chunk_plan.append({
                            "prompt": cfg["prompt"],
                            "seed_algo": cfg["seed_algo"],
                            "chunk_len": cl,
                        })
                total_duration = sum(cfg["duration"] for cfg in prompt_configs)
            else:
                total_duration = p.duration
                use_chunking_global = total_duration > LONG_FORM_THRESHOLD_S
                MAX_CHUNK_S = AUTO_CHUNK_SIZE_S if use_chunking_global else total_duration

                if use_chunking_global:
                    OVERLAP_S_TMP = OVERLAP_S
                    num_chunks = math.ceil((total_duration - OVERLAP_S_TMP) / (MAX_CHUNK_S - OVERLAP_S_TMP))
                    last_chunk_len = total_duration - (num_chunks - 1) * (MAX_CHUNK_S - OVERLAP_S_TMP)
                    if last_chunk_len < MIN_CHUNK_S and num_chunks > 1:
                        num_chunks -= 1
                else:
                    OVERLAP_S_TMP = 0.0
                    num_chunks = 1

                if len(prompt_configs) > 1:
                    chunks_per_prompt = max(1, num_chunks // len(prompt_configs))
                    remainder = num_chunks % len(prompt_configs)
                    for i, cfg in enumerate(prompt_configs):
                        count = chunks_per_prompt + (1 if i < remainder else 0)
                        for j in range(count):
                            is_last = (i == len(prompt_configs) - 1 and j == count - 1)
                            if is_last:
                                cl = max(MIN_CHUNK_S, total_duration - (num_chunks - 1) * (MAX_CHUNK_S - OVERLAP_S_TMP))
                            else:
                                cl = MAX_CHUNK_S
                            chunk_plan.append({
                                "prompt": cfg["prompt"],
                                "seed_algo": cfg["seed_algo"],
                                "chunk_len": cl,
                            })
                    while len(chunk_plan) < num_chunks:
                        cfg = prompt_configs[-1]
                        chunk_plan.append({
                            "prompt": cfg["prompt"],
                            "seed_algo": cfg["seed_algo"],
                            "chunk_len": MAX_CHUNK_S,
                        })
                else:
                    cfg = prompt_configs[0]
                    for i in range(num_chunks):
                        is_last = (i == num_chunks - 1)
                        if is_last and use_chunking_global:
                            cl = max(MIN_CHUNK_S, total_duration - (num_chunks - 1) * (MAX_CHUNK_S - OVERLAP_S_TMP))
                        elif use_chunking_global:
                            cl = MAX_CHUNK_S
                        else:
                            cl = total_duration
                        chunk_plan.append({
                            "prompt": cfg["prompt"],
                            "seed_algo": cfg["seed_algo"],
                            "chunk_len": cl,
                        })

            num_chunks = len(chunk_plan)
            use_chunking = num_chunks > 1
            OVERLAP_S_LOCAL = OVERLAP_S if use_chunking else 0.0

            if use_chunking:
                self.status_updated.emit(f"Long-form mode: Streaming {num_chunks} chunks to disk...")

            sample_rate = getattr(
                getattr(getattr(self.pipe, "vae", None), "config", None),
                "sample_rate", 16000
            )

            saved_paths, metadata_list = [], []
            for variation in range(p.num_variations):
                if self.is_cancelled:
                    raise InterruptedError("User cancelled")
                base_seed = (
                    p.seed + variation if p.seed >= 0
                    else int(time.time() * 1000) % (2 ** 31) + variation
                )

                self.ema_step_time, self.last_step_time = None, None
                self._steps_done_before_variation = variation * num_chunks * p.steps
                self._total_steps_all = p.num_variations * num_chunks * p.steps
                self._num_chunks = num_chunks
                self._variation = variation
                self._total_variations = p.num_variations

                safe_prompt = sanitize_filename(p.prompt)
                suffix = f"_v{variation + 1}" if p.num_variations > 1 else ""
                filename = f"{int(time.time())}_{safe_prompt}{suffix}.wav"
                out_path = os.path.join(final_out_dir, filename)

                overlap_buffer = None
                wav_writer = wave.open(out_path, 'wb')
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(sample_rate)

                try:
                    for chunk_idx in range(num_chunks):
                        if self.is_cancelled:
                            raise InterruptedError("User cancelled")

                        is_last_chunk = (chunk_idx == num_chunks - 1)
                        chunk_cfg = chunk_plan[chunk_idx]
                        chunk_len = chunk_cfg["chunk_len"]
                        current_prompt = chunk_cfg["prompt"]
                        chunk_algo = chunk_cfg["seed_algo"]

                        if chunk_algo == "Random":
                            chunk_seed = random.randint(0, 2 ** 31 - 1)
                        elif chunk_algo == "Fixed":
                            chunk_seed = base_seed
                        elif chunk_algo == "Golden Ratio":
                            phi = 1.61803398875
                            chunk_seed = int((base_seed + chunk_idx * phi * 1000000) % (2 ** 31))
                        else:  # Sequential
                            chunk_seed = (base_seed + chunk_idx * 7919) % (2 ** 31)

                        gen_device = "cpu" if p.use_cpu_offload else actual_device
                        generator = torch.Generator(device=gen_device).manual_seed(chunk_seed)
                        cb = partial(self._on_step, chunk_idx=chunk_idx)

                        kwargs = dict(
                            prompt=current_prompt,
                            negative_prompt=p.negative_prompt if p.negative_prompt else None,
                            num_inference_steps=p.steps,
                            audio_length_in_s=chunk_len,
                            guidance_scale=p.guidance,
                            generator=generator,
                        )
                        if AudioGenerationThread._uses_new_callback:
                            kwargs["callback_on_step_end"] = cb
                            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                        else:
                            kwargs["callback"] = cb
                            kwargs["callback_steps"] = 1

                        with torch.inference_mode():
                            audio_chunk = self.pipe(**kwargs).audios[0]
                        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                        if num_chunks > 1:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        if p.trim:
                            audio_chunk = trim_silence(audio_chunk)
                        if p.normalize:
                            audio_chunk = normalize_audio(audio_chunk)
                        if p.fade_in or p.fade_out:
                            audio_chunk = apply_fade(
                                audio_chunk, sample_rate,
                                FADE_DURATION_S if p.fade_in else 0,
                                FADE_DURATION_S if p.fade_out else 0,
                            )

                        chunk_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)

                        overlap_samples_target = int(OVERLAP_S_LOCAL * sample_rate)
                        if (overlap_buffer is not None and use_chunking
                                and overlap_samples_target > 0
                                and len(chunk_int16) >= overlap_samples_target):
                            overlap_samples = min(len(overlap_buffer), overlap_samples_target)
                            t = np.linspace(0, np.pi / 2, overlap_samples, dtype=np.float32)
                            fade_out_arr, fade_in_arr = np.cos(t), np.sin(t)
                            prev_float = overlap_buffer[-overlap_samples:].astype(np.float32) / 32768.0
                            curr_float = chunk_int16[:overlap_samples].astype(np.float32) / 32768.0
                            crossfaded = prev_float * fade_out_arr + curr_float * fade_in_arr
                            chunk_int16[:overlap_samples] = (
                                np.clip(crossfaded, -1.0, 1.0) * 32767
                            ).astype(np.int16)

                        if (use_chunking and not is_last_chunk
                                and len(chunk_int16) > overlap_samples_target
                                and overlap_samples_target > 0):
                            overlap_buffer = chunk_int16[-overlap_samples_target:].copy()
                            wav_writer.writeframes(
                                chunk_int16[:-overlap_samples_target].tobytes()
                            )
                        else:
                            if not is_last_chunk:
                                overlap_buffer = None
                            wav_writer.writeframes(chunk_int16.tobytes())
                finally:
                    if wav_writer is not None:
                        try:
                            wav_writer.close()
                        except Exception:
                            pass
                    wav_writer = None

                metadata = {
                    'prompt': p.prompt, 'negative_prompt': p.negative_prompt, 'model': p.model_name,
                    'duration': total_duration, 'steps': p.steps, 'guidance': p.guidance, 'seed': base_seed,
                    'sample_rate': sample_rate, 'device': actual_device, 'variation': variation + 1,
                    'total_variations': p.num_variations, 'generation_time': time.time() - gen_start_time,
                    'timestamp': datetime.now().isoformat(), 'chunked_generation': num_chunks > 1,
                    'num_chunks': num_chunks, 'path': out_path, 'filename': filename,
                    'tags': [], 'favorite': False,
                    'multi_prompt': p.use_multi_prompt, 'seed_algo': p.seed_algo,
                    'prompt_configs': prompt_configs if use_multi else None,
                }
                saved_paths.append(out_path)
                metadata_list.append(metadata)

            self.progress_updated.emit(100)
            self.status_updated.emit("Generation complete")
            self.eta_updated.emit("")
            self.generation_completed.emit(
                saved_paths[0],
                {'paths': saved_paths, 'metadata': metadata_list, 'gen_time': time.time() - gen_start_time}
            )

        except InterruptedError:
            self.status_updated.emit("Cancelled")
            self.progress_updated.emit(0)
            self.eta_updated.emit("")
            self.cancelled.emit()
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"Generation failed: {err_trace}")
            if "out of memory" in str(e).lower():
                self.error_occurred.emit("GPU Out of Memory!\nTry enabling CPU Offload or lowering Steps.")
            else:
                self.error_occurred.emit(f"Generation Failed:\n{err_trace}")
        finally:
            if wav_writer is not None:
                try:
                    wav_writer.close()
                except Exception:
                    pass
            self.clear_cache()

    def clear_cache(self):
        self.pipe = None
        gc.collect()
        if ML_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @classmethod
    def unload_pipeline(cls):
        with cls._pipe_lock:
            cls._unload_pipeline_unsafe()

    @classmethod
    def _unload_pipeline_unsafe(cls):
        if cls._shared_pipe is not None:
            pipe = cls._shared_pipe
            if hasattr(pipe, "remove_hooks"):
                try:
                    pipe.remove_hooks()
                except Exception:
                    pass
            try:
                if ML_AVAILABLE and torch.cuda.is_available():
                    pipe.to("cpu")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
            del pipe
            cls._shared_pipe = None
            gc.collect()


# =============================================================================
# Model Preload Thread
# =============================================================================

class ModelPreloadThread(QThread):
    status_updated = Signal(str)
    finished_preload = Signal(str, bool)

    def __init__(self, model_name: str, cache_dir: str, device: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.setObjectName("ModelPreloadThread")

    def run(self):
        try:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create cache dir {self.cache_dir}: {e}")
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["HF_HUB_CACHE"] = self.cache_dir

            with AudioGenerationThread._pipe_lock:
                if (AudioGenerationThread._shared_pipe is not None
                    and AudioGenerationThread._shared_pipe_model == self.model_name
                    and AudioGenerationThread._shared_pipe_device == self.device
                    and AudioGenerationThread._shared_pipe_offload == False):
                    self.status_updated.emit(f"Model '{self.model_name}' already loaded.")
                    self.finished_preload.emit(self.model_name, True)
                    return

                AudioGenerationThread._unload_pipeline_unsafe()
                self.status_updated.emit(f"Preloading {self.model_name} from {self.cache_dir}…")

                dtype = torch.float16 if self.device != "cpu" else torch.float32
                load_kwargs = {"torch_dtype": dtype, "cache_dir": self.cache_dir}
                if dtype == torch.float16:
                    load_kwargs["variant"] = "fp16"

                try:
                    pipe = AudioLDM2Pipeline.from_pretrained(
                        self.model_name, **load_kwargs, use_safetensors=True
                    )
                except Exception as e_outer:
                    logger.warning(f"Preload fp16 failed, retrying default: {e_outer}")
                    load_kwargs.pop("variant", None)
                    pipe = AudioLDM2Pipeline.from_pretrained(
                        self.model_name, **load_kwargs, use_safetensors=False
                    )

                pipe = pipe.to(self.device)
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_tiling"):
                    pipe.enable_vae_tiling()

                try:
                    sig = inspect.signature(pipe.__call__)
                    AudioGenerationThread._uses_new_callback = "callback_on_step_end" in sig.parameters
                except Exception:
                    AudioGenerationThread._uses_new_callback = True

                AudioGenerationThread._shared_pipe = pipe
                AudioGenerationThread._shared_pipe_model = self.model_name
                AudioGenerationThread._shared_pipe_device = self.device
                AudioGenerationThread._shared_pipe_offload = False

            self.finished_preload.emit(self.model_name, True)
        except Exception as e:
            logger.error(f"Preload failed for {self.model_name}: {e}")
            self.finished_preload.emit(self.model_name, False)


# =============================================================================
# Video Generation Threads
# =============================================================================

class PreviewRenderThread(QThread):
    preview_ready = Signal(bytes)
    error_occurred = Signal(str)

    def __init__(self, audio_path: str, vis_type: str, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.vis_type = vis_type
        self.setObjectName("PreviewRenderThread")

    def run(self):
        try:
            # Only load up to 30 seconds to keep the preview fast
            y, sr = librosa.load(self.audio_path, sr=None, duration=30.0)
            
            if self.vis_type == "Spectrogram":
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                feature = librosa.power_to_db(S, ref=np.max)
            elif self.vis_type == "Chromagram":
                feature = librosa.feature.chroma_stft(y=y, sr=sr)
            elif self.vis_type == "Harmonics":
                y_harmonic = librosa.effects.harmonic(y)
                S = librosa.feature.melspectrogram(y=y_harmonic, sr=sr)
                feature = librosa.power_to_db(S, ref=np.max)
            else:
                feature = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

            plt.style.use('dark_background')
            fig = plt.figure(figsize=(6, 4), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
            ax.set_axis_off()               # No labels, ticks, or borders
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Full extent for the static preview
            ax.imshow(feature, aspect='auto', origin='lower', 
                      extent=[0, len(y)/sr, 0, feature.shape[0]], cmap='magma')
            
            buf = BytesIO()
            # facecolor=fig.get_facecolor() ensures the background is black, not white
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
            plt.close(fig)
            
            buf.seek(0)
            self.preview_ready.emit(buf.getvalue())
        except Exception as e:
            self.error_occurred.emit(str(e))


class VideoGenerationThread(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    completed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, audio_path: str, output_path: str, vis_type: str, video_format: str, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.output_path = output_path
        self.vis_type = vis_type
        self.video_format = video_format  # "16:9" or "9:16"
        self._cancel = threading.Event()
        self.setObjectName("VideoGenerationThread")

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            self.status_updated.emit("Loading audio for visualization...")
            y, sr = librosa.load(self.audio_path, sr=None)
            duration = len(y) / sr

            self.status_updated.emit("Computing audio features...")
            if self.vis_type == "Spectrogram":
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                feature = librosa.power_to_db(S, ref=np.max)
            elif self.vis_type == "Chromagram":
                feature = librosa.feature.chroma_stft(y=y, sr=sr)
            elif self.vis_type == "Harmonics":
                y_harmonic = librosa.effects.harmonic(y)
                S = librosa.feature.melspectrogram(y=y_harmonic, sr=sr)
                feature = librosa.power_to_db(S, ref=np.max)
            else:
                feature = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

            plt.style.use('dark_background')
            
            # Set dimensions based on chosen YouTube format
            if "9:16" in self.video_format:
                # 1080x1920 (Shorts/TikTok)
                fig = plt.figure(figsize=(9, 16), dpi=120)
            else:
                # 1920x1080 (Standard YouTube)
                fig = plt.figure(figsize=(16, 9), dpi=120)
                
            # Fill the entire frame to remove all padding/borders
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()  # Turn off all labels, ticks, and spines
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            fps = 30
            if duration > 60:
                fps = 20
            if duration > 180:
                fps = 15
            total_frames = max(1, int(duration * fps))
            
            # Lock axes limits to prevent jittering/rescaling during animation
            ax.set_xlim(0, duration)
            ax.set_ylim(0, feature.shape[0])
            
            img = ax.imshow(feature[:, :1], aspect='auto', origin='lower', 
                           extent=[0, duration, 0, feature.shape[0]], cmap='magma', animated=True)

            self.status_updated.emit("Rendering video frames...")
            temp_video = self.output_path + "_temp.mp4"
            writer = FFMpegWriter(fps=fps)

            with writer.saving(fig, temp_video, dpi=120):
                for i in range(total_frames):
                    if self._cancel.is_set():
                        break
                    current_t = (i + 1) / fps
                    current_frame = int((i + 1) / total_frames * feature.shape[1])
                    current_frame = max(1, min(current_frame, feature.shape[1]))
                    
                    img.set_array(feature[:, :current_frame])
                    # Update extent to grow the visualizer over time
                    img.set_extent([0, max(current_t, 1.0/fps), 0, feature.shape[0]])
                    writer.grab_frame()
                    
                    if i % 10 == 0:
                        self.progress_updated.emit(int(i / total_frames * 100))
                        self.status_updated.emit(f"Rendering frame {i}/{total_frames}")

            plt.close(fig)

            if self._cancel.is_set():
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                self.status_updated.emit("Video generation cancelled")
                return

            self.progress_updated.emit(90)
            self.status_updated.emit("Merging audio with video...")
            
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', temp_video, '-i', self.audio_path, 
                    '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', 
                    '-shortest', self.output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            except Exception:
                if os.path.exists(temp_video):
                    os.rename(temp_video, self.output_path)
                self.status_updated.emit("Saved video without audio (ffmpeg missing?).")

            self.progress_updated.emit(100)
            self.completed.emit(self.output_path)

        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"Video generation failed: {err_trace}")
            self.error_occurred.emit(f"Video Generation Failed:\n{err_trace}")


# =============================================================================
# UI Widgets
# =============================================================================

class SFXPromptBuilderTab(QWidget):
    insertPromptRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; color: #ccc; } QLabel { color: #ddd; }")

        layout = QVBoxLayout(self)
        info_lbl = QLabel("Formula: [Category]: [Action], [Environment], [Quality]")
        info_lbl.setStyleSheet("color: #2A82DA; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(info_lbl)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.category_combo = QComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(["Foley", "Cinematic", "Ambient", "Game UI", "Nature", "Sci-Fi", "Mechanical", "Weapon", "Vehicle", "Creature"])
        form_layout.addRow("1. Category / Foley Term:", self.category_combo)

        self.action_combo = QComboBox()
        self.action_combo.setEditable(True)
        self.action_combo.addItems(["footsteps on gravel", "heavy door slamming", "rain on a tin roof", "sword draw and metallic scrape", "massive sub-bass impact", "clean digital confirmation beep", "distant thunder rolling", "fire crackling in a fireplace", "glass breaking", "engine revving"])
        form_layout.addRow("2. Specific Action / Sound:", self.action_combo)

        self.env_combo = QComboBox()
        self.env_combo.setEditable(True)
        self.env_combo.addItems(["close microphone", "distant", "large reverberant hall", "small dry room", "dense forest", "busy city street", "professional studio", "underwater", "vast open space", "inside a car"])
        form_layout.addRow("3. Acoustic Environment:", self.env_combo)

        self.quality_combo = QComboBox()
        self.quality_combo.setEditable(True)
        self.quality_combo.addItems(["high quality, professional recording", "ASMR, highly detailed", "cinematic, epic", "crisp transients, clean", "deep bass response", "no reverb, dry", "immersive, 3D audio", "lo-fi, vintage"])
        form_layout.addRow("4. Quality Modifiers:", self.quality_combo)

        layout.addLayout(form_layout)

        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("color: #2A82DA; font-style: italic; padding: 10px; background-color: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 4px; min-height: 40px;")
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        self.category_combo.currentTextChanged.connect(self.update_preview)
        self.action_combo.currentTextChanged.connect(self.update_preview)
        self.env_combo.currentTextChanged.connect(self.update_preview)
        self.quality_combo.currentTextChanged.connect(self.update_preview)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.insert_btn = QPushButton("➕ Insert into Prompt")
        self.insert_btn.setStyleSheet("QPushButton { background-color: #2A82DA; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; } QPushButton:hover { background-color: #3A92EA; }")
        self.insert_btn.clicked.connect(self._on_insert)
        btn_row.addWidget(self.insert_btn)
        layout.addLayout(btn_row)
        self.update_preview()

    def _on_insert(self):
        text = self.get_formatted_prompt()
        if text:
            self.insertPromptRequested.emit(text)

    def update_preview(self):
        cat = self.category_combo.currentText().strip()
        act = self.action_combo.currentText().strip()
        env = self.env_combo.currentText().strip()
        qual = self.quality_combo.currentText().strip()
        parts = []
        if cat and act:
            parts.append(f"{cat}: {act}")
        elif act:
            parts.append(act)
        if env:
            parts.append(env)
        if qual:
            parts.append(qual)
        self.preview_label.setText(", ".join(parts) if parts else "Fill in the fields above to build your prompt...")

    def get_formatted_prompt(self):
        text = self.preview_label.text()
        return text if text != "Fill in the fields above to build your prompt..." else ""


class MultiPromptEditorTab(QWidget):
    applyRequested = Signal(str, float)

    SEED_ALGOS = ["Sequential", "Random", "Fixed", "Golden Ratio"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_seed_algo = "Sequential"
        self.default_duration = 10.0
        self._loading = True

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            "Each prompt can have its own seed algorithm and duration.\n"
            "Format: <b>prompt | seed_algo | duration</b>\n"
            "Example: <i>Cinematic impact | Random | 30</i>\n"
            "Leave seed_algo or duration blank to use global defaults."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #2A82DA; font-style: italic; background-color: #1e1e1e; padding: 10px; border: 1px solid #3c3c3c; border-radius: 4px;")
        layout.addWidget(info)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #2ecc71; font-weight: bold; padding: 4px;")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#", "Prompt", "Seed Algorithm", "Duration (s)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellChanged.connect(self._on_cell_changed)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("➕ Add Prompt")
        self.add_btn.clicked.connect(lambda: self._add_row())
        btn_row.addWidget(self.add_btn)

        self.duplicate_btn = QPushButton("📋 Duplicate")
        self.duplicate_btn.clicked.connect(self._duplicate_row)
        btn_row.addWidget(self.duplicate_btn)

        self.remove_btn = QPushButton("➖ Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.remove_btn)

        btn_row.addStretch()

        self.move_up_btn = QPushButton("⬆ Up")
        self.move_up_btn.clicked.connect(lambda: self._move_row(-1))
        btn_row.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("⬇ Down")
        self.move_down_btn.clicked.connect(lambda: self._move_row(1))
        btn_row.addWidget(self.move_down_btn)
        layout.addLayout(btn_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick fill:"))
        self.preset_duration_spin = QDoubleSpinBox()
        self.preset_duration_spin.setRange(1.0, MAX_DURATION_S)
        self.preset_duration_spin.setValue(self.default_duration)
        self.preset_duration_spin.setSuffix(" s")
        preset_row.addWidget(self.preset_duration_spin)

        self.apply_dur_btn = QPushButton("Apply Dur to All")
        self.apply_dur_btn.clicked.connect(self._apply_duration_all)
        preset_row.addWidget(self.apply_dur_btn)

        self.preset_algo_combo = QComboBox()
        self.preset_algo_combo.addItems(self.SEED_ALGOS)
        self.preset_algo_combo.setCurrentText(self.default_seed_algo)
        preset_row.addWidget(self.preset_algo_combo)

        self.apply_algo_btn = QPushButton("Apply Algo to All")
        self.apply_algo_btn.clicked.connect(self._apply_algo_all)
        preset_row.addWidget(self.apply_algo_btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        ok_row = QHBoxLayout()
        ok_row.addStretch()
        self.apply_btn = QPushButton("✓ Apply to Generate Tab")
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #2A82DA; color: white; font-weight: bold; padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3A92EA; }"
        )
        self.apply_btn.clicked.connect(self._on_apply)
        ok_row.addWidget(self.apply_btn)
        layout.addLayout(ok_row)

    def update_defaults(self, algo: str, dur: float):
        self.default_seed_algo = algo
        self.default_duration = dur
        self.preset_algo_combo.setCurrentText(algo)
        self.preset_duration_spin.setValue(dur)

    def load_from_text(self, text: str):
        self._loading = True
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        self.table.setRowCount(0)
        for line in lines:
            cfg = parse_multi_prompt_line(line, self.default_seed_algo, self.default_duration)
            self._add_row(cfg["prompt"], cfg["seed_algo"], cfg["duration"])
        self._loading = False
        self._update_summary()

    def _add_row(self, prompt: str = "", seed_algo: str = None, duration: float = None):
        self._loading = True
        row = self.table.rowCount()
        self.table.insertRow(row)

        num_item = QTableWidgetItem(str(row + 1))
        num_item.setFlags(Qt.ItemIsEnabled)
        num_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, num_item)

        self.table.setItem(row, 1, QTableWidgetItem(prompt))

        combo = QComboBox()
        combo.addItems(self.SEED_ALGOS)
        combo.setCurrentText(seed_algo or self.default_seed_algo)
        self.table.setCellWidget(row, 2, combo)

        spin = QDoubleSpinBox()
        spin.setRange(1.0, MAX_DURATION_S)
        spin.setDecimals(1)
        spin.setValue(duration if duration is not None else self.default_duration)
        spin.valueChanged.connect(self._update_summary)
        self.table.setCellWidget(row, 3, spin)

        combo.currentTextChanged.connect(self._update_summary)
        self._loading = False
        self._renumber()
        self._update_summary()

    def _duplicate_row(self):
        row = self.table.currentRow()
        if row < 0: return
        prompt = self.table.item(row, 1).text()
        algo = self.table.cellWidget(row, 2).currentText()
        dur = self.table.cellWidget(row, 3).value()
        self._add_row(prompt, algo, dur)

    def _remove_selected(self):
        rows = sorted(set(item.row() for item in self.table.selectedItems()), reverse=True)
        if not rows: return
        for row in rows:
            self.table.removeRow(row)
        self._renumber()
        self._update_summary()

    def _move_row(self, direction: int):
        row = self.table.currentRow()
        if row < 0: return
        target = row + direction
        if target < 0 or target >= self.table.rowCount(): return
        prompt_a = self.table.item(row, 1).text()
        prompt_b = self.table.item(target, 1).text()
        self.table.item(row, 1).setText(prompt_b)
        self.table.item(target, 1).setText(prompt_a)
        
        combo_a = self.table.cellWidget(row, 2)
        combo_b = self.table.cellWidget(target, 2)
        algo_a, algo_b = combo_a.currentText(), combo_b.currentText()
        combo_a.setCurrentText(algo_b)
        combo_b.setCurrentText(algo_a)
        
        spin_a = self.table.cellWidget(row, 3)
        spin_b = self.table.cellWidget(target, 3)
        dur_a, dur_b = spin_a.value(), spin_b.value()
        spin_a.setValue(dur_b)
        spin_b.setValue(dur_a)
        self.table.setCurrentCell(target, 1)

    def _renumber(self):
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setText(str(i + 1))

    def _on_cell_changed(self, row, col):
        if not self._loading and col == 1:
            self._update_summary()

    def _apply_duration_all(self):
        val = self.preset_duration_spin.value()
        for row in range(self.table.rowCount()):
            spin = self.table.cellWidget(row, 3)
            if spin: spin.setValue(val)

    def _apply_algo_all(self):
        algo = self.preset_algo_combo.currentText()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 2)
            if combo: combo.setCurrentText(algo)

    def _update_summary(self):
        total = 0.0
        count = self.table.rowCount()
        chunks_est = 0
        for row in range(count):
            spin = self.table.cellWidget(row, 3)
            if spin:
                total += spin.value()
                chunks_est += len(compute_chunks_for_duration(spin.value()))
        self.summary_label.setText(
            f"📊 {count} prompt(s) | Total duration: {format_time(total)} | Estimated chunks: {chunks_est}"
        )

    def _on_apply(self):
        self.applyRequested.emit(self.get_serialized_text(), self.get_total_duration())

    def get_total_duration(self):
        total = 0.0
        for row in range(self.table.rowCount()):
            spin = self.table.cellWidget(row, 3)
            if spin: total += spin.value()
        return total

    def get_serialized_text(self) -> str:
        lines = []
        for row in range(self.table.rowCount()):
            prompt_item = self.table.item(row, 1)
            if not prompt_item: continue
            prompt = prompt_item.text().strip()
            if not prompt: continue
            combo = self.table.cellWidget(row, 2)
            spin = self.table.cellWidget(row, 3)
            algo = combo.currentText() if combo else self.default_seed_algo
            dur = spin.value() if spin else self.default_duration

            has_custom_algo = algo != self.default_seed_algo
            has_custom_dur = abs(dur - self.default_duration) > 0.01

            if has_custom_algo and has_custom_dur:
                lines.append(f"{prompt} | {algo} | {dur}")
            elif has_custom_algo:
                lines.append(f"{prompt} | {algo}")
            elif has_custom_dur:
                lines.append(f"{prompt} | {dur}")
            else:
                lines.append(prompt)
        return '\n'.join(lines)


class VideoVisualizationTab(QWidget):
    """Dedicated tab for generating MP4 visualizations from audio files."""
    video_completed = Signal(str)
    audio_path_changed = Signal(str)
    preview_viz_requested = Signal(str, str)  # path, vis_type

    VIS_TYPES = ["Spectrogram", "Chromagram", "Harmonics"]
    FORMATS = ["YouTube Video (16:9)", "YouTube Shorts (9:16)"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; color: #ccc; } QLabel { color: #ddd; }")

        self.current_audio_path = None
        self.video_thread = None
        self.last_output_path = None
        self._history_paths = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        info = QLabel(
            "🎬 Generate an MP4 visualization of an audio file.\n"
            "Pick a file from your history, or browse for any audio file.\n"
            "The main visualizer will update to preview the selected style."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "color: #2A82DA; font-style: italic; background-color: #1e1e1e; "
            "padding: 10px; border: 1px solid #3c3c3c; border-radius: 4px;"
        )
        layout.addWidget(info)

        src_group = QGroupBox("Audio Source")
        src_layout = QVBoxLayout(src_group)

        hist_row = QHBoxLayout()
        hist_row.addWidget(QLabel("From History:"))
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(250)
        self.history_combo.currentIndexChanged.connect(self._on_history_selected)
        hist_row.addWidget(self.history_combo, 1)

        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setFixedWidth(100)
        self.browse_btn.clicked.connect(self._browse_audio)
        hist_row.addWidget(self.browse_btn)
        src_layout.addLayout(hist_row)

        self.selected_file_label = QLabel("No audio file selected.")
        self.selected_file_label.setWordWrap(True)
        self.selected_file_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 6px; "
            "background-color: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 3px;"
        )
        src_layout.addWidget(self.selected_file_label)
        layout.addWidget(src_group)

        vis_group = QGroupBox("Visualization Settings")
        vis_layout = QFormLayout(vis_group)
        vis_layout.setSpacing(8)

        self.vis_type_combo = QComboBox()
        self.vis_type_combo.addItems(self.VIS_TYPES)
        self.vis_type_combo.setItemData(0, "Mel-scaled spectrogram showing frequency energy over time", Qt.ToolTipRole)
        self.vis_type_combo.setItemData(1, "Chromagram showing the 12 pitch classes (musical notes)", Qt.ToolTipRole)
        self.vis_type_combo.setItemData(2, "Mel spectrogram of the harmonic (pitch) component only", Qt.ToolTipRole)
        vis_layout.addRow("Visualization:", self.vis_type_combo)

        self.format_combo = QComboBox()
        self.format_combo.addItems(self.FORMATS)
        self.format_combo.setItemData(0, "1920x1080 resolution. Standard horizontal format for regular YouTube videos.", Qt.ToolTipRole)
        self.format_combo.setItemData(1, "1080x1920 resolution. Vertical format for YouTube Shorts and TikTok.", Qt.ToolTipRole)
        vis_layout.addRow("Video Format:", self.format_combo)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("./generated_video")
        out_browse = QPushButton("…")
        out_browse.setFixedWidth(30)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_dir_edit)
        out_row.addWidget(out_browse)
        vis_layout.addRow("Output Dir:", out_row)
        layout.addWidget(vis_group)

        btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("🎬  Generate MP4")
        self.generate_btn.setMinimumHeight(48)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2A82DA; color: white; font-size: 14px; "
            "font-weight: bold; border: none; border-radius: 4px; } "
            "QPushButton:hover { background-color: #3A92EA; } "
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.generate_btn.clicked.connect(self._generate)
        btn_row.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(48)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 14px; "
            "font-weight: bold; border: none; border-radius: 4px; } "
            "QPushButton:hover { background-color: #d0493b; } "
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #2ecc71; padding: 4px;")
        self.result_label.setVisible(False)
        status_layout.addWidget(self.result_label)

        open_row = QHBoxLayout()
        open_row.addStretch()
        self.open_folder_btn = QPushButton("📂 Open Output Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        open_row.addWidget(self.open_folder_btn)
        status_layout.addLayout(open_row)

        layout.addWidget(status_group)
        layout.addStretch()

        # Connect dropdown change to request preview update
        self.vis_type_combo.currentTextChanged.connect(self._trigger_preview)

    def set_output_dir(self, path: str):
        if path and not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(path)

    def update_history(self, history_list):
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self._history_paths = []
        self.history_combo.addItem("-- Select an audio file from history --", None)

        seen = set()
        for meta in reversed(history_list):
            path = meta.get('path', '')
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            prompt = meta.get('prompt', '(no prompt)') or '(no prompt)'
            dur = format_time(meta.get('duration', 0.0))
            ts = meta.get('timestamp', '')[:10] or '????-??-??'
            label = f"[{ts}] {prompt[:55]}  ({dur})"
            self.history_combo.addItem(label, path)
            self._history_paths.append(path)

        if self.current_audio_path:
            idx = self.history_combo.findData(self.current_audio_path)
            if idx >= 0:
                self.history_combo.setCurrentIndex(idx)
            else:
                self.history_combo.setCurrentIndex(0)
        self.history_combo.blockSignals(False)

    def _on_history_selected(self, idx):
        if idx <= 0:
            return
        path = self.history_combo.currentData()
        if path and os.path.exists(path):
            self.set_audio_path(path)

    def _browse_audio(self):
        start = os.path.dirname(self.current_audio_path) if self.current_audio_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", start,
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*.*)"
        )
        if path:
            self.set_audio_path(path)
            self.history_combo.blockSignals(True)
            self.history_combo.setCurrentIndex(0)
            self.history_combo.blockSignals(False)

    def set_audio_path(self, path, emit_signal=True):
        self.current_audio_path = path
        size_mb = os.path.getsize(path) / (1024 * 1024)
        self.selected_file_label.setText(
            f"📁 <b>{os.path.basename(path)}</b><br>"
            f"<span style='color:#888'>{path}</span><br>"
            f"<span style='color:#888'>Size: {size_mb:.2f} MB</span>"
        )
        self.selected_file_label.setStyleSheet(
            "color: #2ecc71; padding: 8px; background-color: #1e1e1e; "
            "border: 1px solid #2ecc71; border-radius: 3px;"
        )
        self.generate_btn.setEnabled(True)
        self.result_label.setVisible(False)

        self.history_combo.blockSignals(True)
        idx = self.history_combo.findData(path)
        if idx >= 0:
            self.history_combo.setCurrentIndex(idx)
        else:
            self.history_combo.setCurrentIndex(0)
        self.history_combo.blockSignals(False)

        if emit_signal:
            self.audio_path_changed.emit(path)

        self._trigger_preview()

    def _trigger_preview(self):
        if self.current_audio_path:
            self.preview_viz_requested.emit(self.current_audio_path, self.vis_type_combo.currentText())

    def _browse_output(self):
        cur = self.output_dir_edit.text().strip() or os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", cur)
        if path:
            self.output_dir_edit.setText(path)

    def _generate(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "No Audio", "Please select an audio file first.")
            return

        if not VIZ_AVAILABLE:
            QMessageBox.warning(
                self, "Missing Libraries",
                "librosa and matplotlib are required for video generation.\n\n"
                "Please install them:\n   pip install librosa matplotlib"
            )
            return

        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_video")
        os.makedirs(out_dir, exist_ok=True)

        vis_type = self.vis_type_combo.currentText()
        video_format = self.format_combo.currentText()

        base_name = os.path.splitext(os.path.basename(self.current_audio_path))[0]
        safe_base = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)

        if "9:16" in video_format:
            default_name = f"{safe_base}_{vis_type.lower()}_shorts.mp4"
        else:
            default_name = f"{safe_base}_{vis_type.lower()}_yt.mp4"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", os.path.join(out_dir, default_name), "MP4 Video (*.mp4)"
        )
        if not path:
            return

        self.last_output_path = path
        self.video_thread = VideoGenerationThread(self.current_audio_path, path, vis_type, video_format, parent=self)
        self.video_thread.progress_updated.connect(self._on_progress)
        self.video_thread.status_updated.connect(self._on_status)
        self.video_thread.completed.connect(self._on_completed)
        self.video_thread.error_occurred.connect(self._on_error)
        self.video_thread.start()

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting…")
        self.result_label.setVisible(False)
        self.open_folder_btn.setEnabled(False)

    def _cancel(self):
        if self.video_thread:
            self.video_thread.cancel()
            self.status_label.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, val):
        self.progress_bar.setValue(val)

    def _on_status(self, msg):
        self.status_label.setText(msg)

    def _on_completed(self, path):
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ Complete!")
        self.result_label.setText(f"Video saved to:\n{path}")
        self.result_label.setStyleSheet("color: #2ecc71; padding: 4px;")
        self.result_label.setVisible(True)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(True)
        self.video_completed.emit(path)

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Failed")
        self.result_label.setText(msg)
        self.result_label.setStyleSheet("color: #c0392b; padding: 4px;")
        self.result_label.setVisible(True)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "Video Generation Error", msg)

    def _open_output_folder(self):
        if not self.last_output_path:
            return
        folder = os.path.dirname(os.path.abspath(self.last_output_path))
        try:
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
        except Exception as e:
            logger.warning(f"Could not open folder: {e}")

    def is_running(self) -> bool:
        return self.video_thread is not None and self.video_thread.isRunning()

    def cleanup(self):
        if self.is_running():
            self.video_thread.cancel()
            self.video_thread.wait(2000)
            if self.video_thread.isRunning():
                self.video_thread.terminate()
                self.video_thread.wait(1000)


class WaveformLoaderThread(QThread):
    """Offloads audio file reading and peak calculation from the GUI thread."""
    audio_loaded = Signal(np.ndarray, float, int)
    error_occurred = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setObjectName("WaveformLoaderThread")

    def run(self):
        try:
            with wave.open(self.path, 'rb') as wf:
                self.sample_rate = wf.getframerate()
                frames_to_read = min(wf.getnframes(), self.sample_rate * 300)
                raw = wf.readframes(frames_to_read)
                duration_s = wf.getnframes() / self.sample_rate
                
            raw_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_loaded.emit(raw_audio, duration_s, self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to load waveform in thread: {e}")
            self.error_occurred.emit(str(e))


class WaveformWidget(QWidget):
    seekRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_audio = None
        self._peaks = None
        self._peaks_width = -1
        self._poly_peak = None
        self.sample_rate = 16000
        self.duration_s = 0.0
        self._audio_loaded = False
        self._loader_thread = None
        
        # Video Preview Mode Vars
        self.viz_mode = "Waveform"
        self.viz_pixmap = None
        self._preview_thread = None
        
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.playback_position = 0.0
        self.setMouseTracking(True)

    def set_audio(self, path: str):
        self.viz_mode = "Waveform"
        self.viz_pixmap = None
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(1000)

        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(1000)

        if not path or not os.path.exists(path):
            self._raw_audio = None
            self._audio_loaded = False
            self._peaks = None
            self._poly_peak = None
            self.duration_s = 0.0
            self.update()
            return

        self._audio_loaded = False
        self.update()

        self._loader_thread = WaveformLoaderThread(path, self)
        self._loader_thread.audio_loaded.connect(self._on_audio_loaded)
        self._loader_thread.start()

    def set_video_preview(self, path: str, vis_type: str):
        self.viz_mode = vis_type
        self.viz_pixmap = None
        
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(1000)

        self._audio_loaded = False
        self._peaks = None
        self._poly_peak = None
        self.update()

        if not path or not os.path.exists(path):
            return
            
        if not VIZ_AVAILABLE:
            self.update()
            return

        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            self._preview_thread.wait(1000)

        self._preview_thread = PreviewRenderThread(path, vis_type, self)
        self._preview_thread.preview_ready.connect(self._on_preview_ready)
        self._preview_thread.error_occurred.connect(lambda e: self.viz_mode = "Error") # Fallback
        self._preview_thread.start()

    def _on_preview_ready(self, img_bytes):
        pixmap = QPixmap()
        if pixmap.loadFromData(img_bytes, 'PNG'):
            self.viz_pixmap = pixmap
            self.update()

    def _on_audio_loaded(self, raw_audio: np.ndarray, duration_s: float, sample_rate: int):
        self._raw_audio = raw_audio
        self.duration_s = duration_s
        self.sample_rate = sample_rate
        self._audio_loaded = True
        self._peaks = None
        self._peaks_width = -1
        self._ensure_peaks(self.width())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._peaks = None
        self._peaks_width = -1
        if self.viz_mode == "Waveform":
            self._ensure_peaks(self.width())

    def _ensure_peaks(self, width: int):
        if width <= 0:
            return
        if self._peaks is not None and self._peaks_width == width:
            return
        if self._raw_audio is None:
            self._peaks = None
            self._poly_peak = None
            return
        n = len(self._raw_audio)
        abs_audio = np.abs(self._raw_audio)
        if n <= width:
            self._peaks = np.pad(abs_audio, (0, max(0, width - n)), 'constant').astype(np.float32)
        else:
            chunk_size = max(1, n // width)
            n_full = (n // chunk_size) * chunk_size
            trimmed = abs_audio[:n_full].reshape(-1, chunk_size)
            self._peaks = trimmed.max(axis=1).astype(np.float32)
            if len(self._peaks) < width:
                pad = width - len(self._peaks)
                self._peaks = np.pad(self._peaks, (0, pad), 'constant')
        self._peaks_width = width
        self._poly_peak = self._build_poly(self._peaks, width)
        self.update()

    def _build_poly(self, heights, width):
        mid = self.height() / 2.0
        max_h = self.height() * 0.42
        poly = QPolygonF()
        if width <= 0:
            return poly
        poly << QPointF(0, mid)
        for x, h in enumerate(heights):
            poly << QPointF(x, mid - (h * max_h))
        poly << QPointF(width, mid)
        for x in range(width - 1, -1, -1):
            poly << QPointF(x, mid + (heights[x] * max_h))
        return poly

    def set_playback_position(self, position: float):
        self.playback_position = max(0.0, min(1.0, position))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.viz_mode == "Waveform" and self._audio_loaded and event.button() == Qt.LeftButton:
            pos = event.position().x() / max(self.width(), 1)
            self.seekRequested.emit(pos)
            self.set_playback_position(pos)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(22, 22, 26))

        # --- Video Preview Mode ---
        if self.viz_mode != "Waveform":
            if self.viz_pixmap is not None:
                scaled_pixmap = self.viz_pixmap.scaled(
                    width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                x = (width - scaled_pixmap.width()) // 2
                y = (height - scaled_pixmap.height()) // 2
                painter.drawPixmap(x, y, scaled_pixmap)
            else:
                painter.setPen(QColor(100, 100, 100))
                painter.drawText(
                    self.rect(), Qt.AlignCenter,
                    "Computing visualization preview..." if self._preview_thread and self._preview_thread.isRunning() else "Visualization unavailable"
                )
            return

        # --- Standard Waveform Mode ---
        if self._audio_loaded and (self._poly_peak is None or self._peaks_width != width):
            self._ensure_peaks(width)

        if not self._audio_loaded:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "Loading audio..." if self._loader_thread and self._loader_thread.isRunning() else "No audio loaded"
            )
            return

        if self._poly_peak is None or width <= 0:
            return

        mid = height / 2.0
        play_x = int(self.playback_position * (width - 1))

        painter.setPen(QPen(QColor(50, 50, 55), 1, Qt.DashLine))
        painter.setFont(QFont("Segoe UI", 8))
        if self.duration_s > 0:
            step_s = 60.0 if self.duration_s > 600 else (15.0 if self.duration_s > 60 else 2.0)
            t = 0.0
            while t < self.duration_s:
                gx = int((t / self.duration_s) * width)
                painter.drawLine(gx, 10, gx, height - 10)
                painter.drawText(gx + 3, 12, 40, 15, Qt.AlignLeft, format_time(t))
                t += step_s

        painter.setClipRect(play_x, 0, width - play_x, height)
        painter.setBrush(QColor(60, 65, 75))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(self._poly_peak)

        painter.setClipRect(0, 0, play_x + 1, height)
        grad = QLinearGradient(0, 0, 0, height)
        grad.setColorAt(0.0, QColor(40, 160, 240))
        grad.setColorAt(0.5, QColor(80, 220, 255))
        grad.setColorAt(1.0, QColor(40, 160, 240))
        painter.setBrush(grad)
        painter.drawPolygon(self._poly_peak)
        painter.setClipping(False)

        if self.playback_position >= 0:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(play_x, 0, play_x, height)


class PromptHistoryLineEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.setPlaceholderText("Describe the audio... (Ctrl+Enter to generate)")
        self.setMaximumHeight(70)
        self.setTabChangesFocus(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return and (event.modifiers() & Qt.ControlModifier):
            self.window().start_generation()
            return
        super().keyPressEvent(event)

    def add_to_history(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > MAX_PROMPT_HISTORY:
                self.history.pop(0)


# =============================================================================
# Main Window
# =============================================================================

class AudioLDM2Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        if not ML_AVAILABLE:
            logger.error("ML libraries not installed.")
            QMessageBox.critical(
                self, "Missing Dependencies",
                "Required ML libraries (torch/diffusers/transformers) are not installed."
            )
            sys.exit(1)

        self.settings_file = os.path.join(CWD, "config.json")
        self.settings = self.load_json_settings()

        _cache_dir = self.settings.get("cache_dir", "")
        if _cache_dir:
            try:
                os.makedirs(_cache_dir, exist_ok=True)
            except Exception:
                pass
            os.environ["HF_HOME"] = _cache_dir
            os.environ["HF_HUB_CACHE"] = _cache_dir

        self.worker_thread: AudioGenerationThread | None = None
        self._preload_thread: ModelPreloadThread | None = None
        self._gen_queue: deque[GenerationParams] = deque()
        self.base_audio_path = None
        self.current_audio_path = None
        self.setAcceptDrops(True)

        self.setup_ui()
        self.setup_audio()
        self.setup_statusbar()
        self.load_settings()
        self.setup_shortcuts()

        QTimer.singleShot(500, self._maybe_preload_model)

    def load_json_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_json_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.create_left_panel(splitter)
        self.create_right_panel(splitter)
        splitter.setSizes([500, 900])

        # Connect tab changes to switch visualizer modes
        self.video_tab.preview_viz_requested.connect(self.waveform_widget.set_video_preview)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        if index == 3:  # MP4 Video tab
            if not self.video_tab.current_audio_path and self.current_audio_path:
                self.video_tab.set_audio_path(self.current_audio_path, emit_signal=False)
            elif self.video_tab.current_audio_path:
                self.waveform_widget.set_video_preview(self.video_tab.current_audio_path, self.video_tab.vis_type_combo.currentText())
        else:
            if self.current_audio_path and os.path.exists(self.current_audio_path):
                self.waveform_widget.set_audio(self.current_audio_path)

    def create_left_panel(self, parent):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        left_layout.addWidget(self.tab_widget)

        gen_tab = QWidget()
        layout = QVBoxLayout(gen_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        prompt_group = QGroupBox("Prompt")
        p_layout = QVBoxLayout(prompt_group)
        self.prompt_input = PromptHistoryLineEdit()
        p_layout.addWidget(self.prompt_input)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([""] + DEFAULT_PROMPT_PRESETS)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)

        self.sfx_builder_btn = QPushButton("🛠️ SFX Builder")
        self.sfx_builder_btn.setFixedWidth(110)
        self.sfx_builder_btn.setToolTip("Switch to the SFX Builder tab")
        self.sfx_builder_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        preset_row.addWidget(self.sfx_builder_btn)
        p_layout.addLayout(preset_row)

        self.multi_prompt_check = QCheckBox("Use Prompt Sequence (One per line for long-form)")
        self.multi_prompt_check.toggled.connect(self.toggle_multi_prompt)
        p_layout.addWidget(self.multi_prompt_check)

        editor_row = QHBoxLayout()
        self.multi_prompt_edit_btn = QPushButton("📝 Open Sequence Editor")
        self.multi_prompt_edit_btn.setFixedHeight(30)
        self.multi_prompt_edit_btn.setToolTip("Switch to the structured sequence editor tab")
        self.multi_prompt_edit_btn.clicked.connect(self._focus_multi_prompt_tab)
        editor_row.addWidget(self.multi_prompt_edit_btn)
        editor_row.addStretch()
        p_layout.addLayout(editor_row)

        self.multi_prompt_edit = QPlainTextEdit()
        self.multi_prompt_edit.setPlaceholderText(
            "Enter one prompt per line for long-form evolution...\n"
            "Optional per-prompt config: prompt | seed_algo | duration\n\n"
            "Examples:\n"
            "Cinematic riser and impact | Random | 30\n"
            "Evolving dark drone | Golden Ratio | 45\n"
            "Ambient forest, wind and birds | Sequential | 60\n"
            "Outro fade to silence | Fixed | 15"
        )
        self.multi_prompt_edit.setVisible(False)
        self.multi_prompt_edit.setMaximumHeight(140)
        p_layout.addWidget(self.multi_prompt_edit)

        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setPlaceholderText("Negative prompt (optional)...")
        p_layout.addWidget(self.neg_prompt_input)
        layout.addWidget(prompt_group)

        param_group = QGroupBox("Parameters")
        f_layout = QFormLayout(param_group)
        f_layout.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(DEFAULT_MODELS)
        self.model_combo.currentTextChanged.connect(lambda _: self._maybe_preload_model())
        f_layout.addRow("Model:", self.model_combo)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, MAX_DURATION_S)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setDecimals(1)
        f_layout.addRow("Duration:", self.duration_spin)

        self.duration_info = QLabel("→ Will generate as a single chunk.")
        self.duration_info.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        f_layout.addRow("", self.duration_info)
        self.duration_spin.valueChanged.connect(self.update_duration_info)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 500)
        self.steps_spin.setValue(200)
        f_layout.addRow("Steps:", self.steps_spin)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setValue(3.5)
        self.guidance_spin.setSingleStep(0.5)
        f_layout.addRow("Guidance:", self.guidance_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 999999999)
        self.seed_spin.setValue(-1)
        f_layout.addRow("Seed:", self.seed_spin)

        self.seed_algo_combo = QComboBox()
        self.seed_algo_combo.addItems(["Sequential", "Random", "Fixed", "Golden Ratio"])
        f_layout.addRow("Seed Algo:", self.seed_algo_combo)

        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(1, 10)
        self.variations_spin.setValue(1)
        f_layout.addRow("Variations:", self.variations_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_combo.setCurrentText("mps")
        self.device_combo.currentTextChanged.connect(lambda _: self._maybe_preload_model())
        f_layout.addRow("Device:", self.device_combo)

        self.cpu_offload_check = QCheckBox("Enable CPU Offload (Low VRAM)")
        f_layout.addRow("", self.cpu_offload_check)

        cache_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setPlaceholderText(DEFAULT_CACHE_DIR)
        cache_browse_btn = QPushButton("…")
        cache_browse_btn.setFixedWidth(30)
        cache_browse_btn.clicked.connect(self.browse_cache_dir)
        cache_row.addWidget(self.cache_dir_edit)
        cache_row.addWidget(cache_browse_btn)
        f_layout.addRow("Cache Dir:", cache_row)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("./generated_audio")
        out_browse_btn = QPushButton("…")
        out_browse_btn.setFixedWidth(30)
        out_browse_btn.clicked.connect(self.browse_output_dir)
        out_row.addWidget(self.output_dir_edit)
        out_row.addWidget(out_browse_btn)
        f_layout.addRow("Output:", out_row)

        self.section_edit = QLineEdit()
        self.section_edit.setPlaceholderText("Subfolder (optional)")
        f_layout.addRow("Section:", self.section_edit)
        layout.addWidget(param_group)

        post_group = QGroupBox("Post-Processing")
        post_layout = QVBoxLayout(post_group)
        self.normalize_check = QCheckBox("Normalize audio volume")
        self.normalize_check.setChecked(True)
        post_layout.addWidget(self.normalize_check)
        self.trim_check = QCheckBox("Trim silence")
        self.trim_check.setChecked(True)
        post_layout.addWidget(self.trim_check)
        fade_row = QHBoxLayout()
        self.fade_in_check = QCheckBox("Fade in")
        self.fade_in_check.setChecked(True)
        fade_row.addWidget(self.fade_in_check)
        self.fade_out_check = QCheckBox("Fade out")
        self.fade_out_check.setChecked(True)
        fade_row.addWidget(self.fade_out_check)
        post_layout.addLayout(fade_row)
        layout.addWidget(post_group)

        gen_btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.setMinimumHeight(45)
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2A82DA; color: white; font-size: 14px; "
            "font-weight: bold; border: none; border-radius: 4px; } "
            "QPushButton:hover { background-color: #3A92EA; }"
        )
        self.generate_btn.clicked.connect(self.start_generation)
        gen_btn_row.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(45)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-size: 14px; "
            "font-weight: bold; border: none; border-radius: 4px; } "
            "QPushButton:hover { background-color: #d0493b; }"
        )
        self.cancel_btn.clicked.connect(self.cancel_generation)
        gen_btn_row.addWidget(self.cancel_btn)
        layout.addLayout(gen_btn_row)

        prog_group = QGroupBox("Status")
        prog_layout = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        prog_layout.addWidget(self.status_label)
        self.eta_label = QLabel("")
        self.eta_label.setAlignment(Qt.AlignRight)
        prog_layout.addWidget(self.eta_label)
        layout.addWidget(prog_group)

        queue_group = QGroupBox("Queue")
        queue_layout = QVBoxLayout(queue_group)
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(80)
        self.queue_list.setAlternatingRowColors(True)
        queue_layout.addWidget(self.queue_list)
        self.queue_clear_btn = QPushButton("Clear Queue")
        self.queue_clear_btn.clicked.connect(self.clear_queue)
        queue_layout.addWidget(self.queue_clear_btn)
        layout.addWidget(queue_group)

        layout.addStretch()
        
        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setFrameShape(QFrame.NoFrame)
        gen_scroll.setWidget(gen_tab)
        
        self.tab_widget.addTab(gen_scroll, "Generate")

        self.sfx_builder_tab = SFXPromptBuilderTab()
        self.sfx_builder_tab.insertPromptRequested.connect(self._insert_sfx_prompt)
        self.tab_widget.addTab(self.sfx_builder_tab, "SFX Builder")

        self.multi_prompt_tab = MultiPromptEditorTab()
        self.multi_prompt_tab.applyRequested.connect(self._apply_multi_prompt_tab)
        self.tab_widget.addTab(self.multi_prompt_tab, "Sequence Editor")

        self.video_tab = VideoVisualizationTab()
        self.video_tab.video_completed.connect(self._on_video_tab_completed)
        self.video_tab.audio_path_changed.connect(self._load_audio_file)
        self.tab_widget.addTab(self.video_tab, "🎬 MP4 Video")

        parent.addWidget(left_panel)

    def create_right_panel(self, parent):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        v_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(v_splitter)

        # --- Top Pane: Visualizer + Playback Controls ---
        top_pane = QWidget()
        top_layout = QVBoxLayout(top_pane)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        vis_group = QGroupBox("Visualizer")
        vis_layout = QVBoxLayout(vis_group)
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.seekRequested.connect(self._seek_to)
        vis_layout.addWidget(self.waveform_widget)
        top_layout.addWidget(vis_group, 1)

        play_group = QGroupBox("Playback & Export")
        play_layout = QVBoxLayout(play_group)
        play_row = QHBoxLayout()
        self.load_btn = QPushButton("Load WAV")
        self.load_btn.setFixedHeight(36)
        self.load_btn.clicked.connect(self.open_audio)
        play_row.addWidget(self.load_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedHeight(36)
        self.play_btn.setMinimumWidth(80)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_playback)
        play_row.addWidget(self.play_btn)

        self.save_btn = QPushButton("Save As...")
        self.save_btn.setFixedHeight(36)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_audio)
        play_row.addWidget(self.save_btn)

        play_layout.addLayout(play_row)

        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Volume:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_slider.valueChanged.connect(self.set_volume)
        vol_layout.addWidget(self.vol_slider)
        self.vol_label = QLabel("50%")
        self.vol_label.setFixedWidth(35)
        vol_layout.addWidget(self.vol_label)
        play_layout.addLayout(vol_layout)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2A82DA;")
        play_layout.addWidget(self.time_label)
        
        top_layout.addWidget(play_group)
        v_splitter.addWidget(top_pane)

        # --- Bottom Pane: History List ---
        hist_group = QGroupBox("History")
        hist_layout = QVBoxLayout(hist_group)
        self.playlist_table = QTableWidget()
        self.playlist_table.setColumnCount(4)
        self.playlist_table.setHorizontalHeaderLabels(["Prompt", "Duration", "Date", "Seed"])
        self.playlist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.playlist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.playlist_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.playlist_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.playlist_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.playlist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.playlist_table.setAlternatingRowColors(True)
        self.playlist_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlist_table.customContextMenuRequested.connect(self.show_history_context_menu)
        self.playlist_table.itemDoubleClicked.connect(self.play_from_playlist)
        hist_layout.addWidget(self.playlist_table)
        
        v_splitter.addWidget(hist_group)
        v_splitter.setSizes([500, 300])

        parent.addWidget(right_panel)

    def _on_preset_changed(self, text: str):
        if not text:
            return
        if self.multi_prompt_check.isChecked():
            self.multi_prompt_edit.setPlainText(text)
        else:
            self.prompt_input.setPlainText(text)

    def show_history_context_menu(self, pos):
        item = self.playlist_table.itemAt(pos)
        if not item:
            return
        row = item.row()
        prompt_item = self.playlist_table.item(row, 0)
        if not prompt_item:
            return
        prompt = prompt_item.text()

        menu = QMenu(self)
        copy_action = QAction("Copy Prompt to Clipboard", self)
        copy_action.triggered.connect(lambda checked=False, p=prompt: QApplication.clipboard().setText(p))
        menu.addAction(copy_action)

        use_action = QAction("Use this Prompt", self)
        use_action.triggered.connect(lambda checked=False, p=prompt: self.use_prompt_from_history(p))
        menu.addAction(use_action)

        menu.exec(self.playlist_table.viewport().mapToGlobal(pos))

    def use_prompt_from_history(self, prompt):
        if self.multi_prompt_check.isChecked():
            self.multi_prompt_edit.setPlainText(prompt)
        else:
            self.prompt_input.setPlainText(prompt)

    def _insert_sfx_prompt(self, text: str):
        if not text:
            return
        if self.multi_prompt_check.isChecked():
            current = self.multi_prompt_edit.toPlainText().strip()
            new_text = f"{current}, {text}" if current else text
            self.multi_prompt_edit.setPlainText(new_text)
        else:
            current = self.prompt_input.toPlainText().strip()
            new_text = f"{current}, {text}" if current else text
            self.prompt_input.setPlainText(new_text)
        
        self.tab_widget.setCurrentIndex(0)

    def _focus_multi_prompt_tab(self):
        default_algo = self.seed_algo_combo.currentText()
        default_dur = self.duration_spin.value()
        self.multi_prompt_tab.update_defaults(default_algo, default_dur)
        
        current_text = self.multi_prompt_edit.toPlainText()
        self.multi_prompt_tab.load_from_text(current_text)
        
        self.tab_widget.setCurrentIndex(2)

    def _apply_multi_prompt_tab(self, text: str, total_duration: float):
        self.multi_prompt_edit.setPlainText(text)
        self.multi_prompt_check.setChecked(True)
        self.duration_spin.setValue(min(total_duration, MAX_DURATION_S))
        self.tab_widget.setCurrentIndex(0)

    def toggle_multi_prompt(self, checked):
        self.multi_prompt_edit.setVisible(checked)
        self.multi_prompt_edit_btn.setVisible(checked)
        self.prompt_input.setVisible(not checked)
        self.preset_combo.setVisible(not checked)
        self.sfx_builder_btn.setVisible(not checked)

    def update_duration_info(self):
        duration = self.duration_spin.value()
        if duration > LONG_FORM_THRESHOLD_S:
            self.duration_info.setText(
                f"⏳ Long-form mode: Will stream to disk in {int(AUTO_CHUNK_SIZE_S)}s "
                f"chunks with {OVERLAP_S}s crossfade."
            )
            self.duration_info.setStyleSheet("color: #2ecc71; font-style: italic; font-size: 10px;")
        else:
            self.duration_info.setText("→ Will generate as a single chunk.")
            self.duration_info.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")

    def setup_audio(self):
        self.audio_output = QAudioOutput(self)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.mediaStatusChanged.connect(self.on_media_status)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.set_volume(50)

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.start_generation)
        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_playback)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_generation)

    def browse_cache_dir(self):
        cur = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        start = cur
        while start and not os.path.exists(start):
            parent = os.path.dirname(start)
            if parent == start:
                break
            start = parent
        path = QFileDialog.getExistingDirectory(self, "Select Model Cache Folder", start or "")
        if path:
            self.cache_dir_edit.setText(path)
            self.settings["cache_dir"] = path
            self.save_json_settings()
            self._maybe_preload_model()

    def browse_output_dir(self):
        cur = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", cur)
        if path:
            self.output_dir_edit.setText(path)

    def _resolve_device(self) -> str:
        device = self.device_combo.currentText()
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _maybe_preload_model(self):
        if not ML_AVAILABLE:
            return
        if self._preload_thread and self._preload_thread.isRunning():
            return
        if (self.worker_thread and self.worker_thread.isRunning()
            and self.worker_thread.params.model_name != self.model_combo.currentText().strip()):
            return
        model = self.model_combo.currentText().strip()
        if not model or '/' not in model:
            return
        device = self._resolve_device()
        cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self._preload_thread = ModelPreloadThread(model, cache_dir, device, parent=self)
        self._preload_thread.status_updated.connect(self.status_bar.showMessage, Qt.QueuedConnection)
        self._preload_thread.finished_preload.connect(self._on_preload_done, Qt.QueuedConnection)
        self._preload_thread.start()

    def _on_preload_done(self, model_name: str, success: bool):
        if success:
            self.status_bar.showMessage(f"Model '{model_name}' ready", 5000)
        else:
            self.status_bar.showMessage(f"Could not preload '{model_name}'", 5000)
        thread = self._preload_thread
        self._preload_thread = None
        if thread is not None:
            thread.deleteLater()

    def _refresh_queue_view(self):
        self.queue_list.setUpdatesEnabled(False)
        self.queue_list.clear()
        for i, p in enumerate(self._gen_queue, 1):
            self.queue_list.addItem(f"{i}. [{sanitize_filename(p.model_name, 20)}] {p.prompt[:60]}")
        self.queue_list.setUpdatesEnabled(True)
        if self._gen_queue:
            self.status_bar.showMessage(f"Queue: {len(self._gen_queue)} pending", 3000)

    def clear_queue(self):
        n = len(self._gen_queue)
        self._gen_queue.clear()
        self._refresh_queue_view()
        if n:
            self.status_bar.showMessage(f"Cleared {n} queued item(s)", 3000)

    def _build_params(self) -> GenerationParams | None:
        use_multi = self.multi_prompt_check.isChecked()
        multi_text = self.multi_prompt_edit.toPlainText().strip() if use_multi else ""

        if use_multi:
            if not multi_text:
                QMessageBox.warning(self, "Input Error", "Please enter at least one prompt in the sequence.")
                return None
            prompt = multi_text.split('\n')[0].strip()
        else:
            prompt = self.prompt_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
                return None

        model = self.model_combo.currentText().strip()
        if not model or '/' not in model:
            QMessageBox.warning(self, "Invalid Model", "Model must be in 'org/model' format.")
            return None

        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        os.makedirs(out_dir, exist_ok=True)
        cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        if not use_multi:
            self.prompt_input.add_to_history(prompt)

        return GenerationParams(
            prompt=prompt,
            negative_prompt=self.neg_prompt_input.text().strip(),
            model_name=model,
            duration=self.duration_spin.value(),
            steps=self.steps_spin.value(),
            guidance=self.guidance_spin.value(),
            seed=self.seed_spin.value(),
            device=self.device_combo.currentText(),
            cache_dir=cache_dir,
            output_dir=out_dir,
            num_variations=self.variations_spin.value(),
            normalize=self.normalize_check.isChecked(),
            fade_in=self.fade_in_check.isChecked(),
            fade_out=self.fade_out_check.isChecked(),
            trim=self.trim_check.isChecked(),
            use_cpu_offload=self.cpu_offload_check.isChecked(),
            section=self.section_edit.text().strip(),
            use_multi_prompt=use_multi,
            multi_prompts=multi_text,
            seed_algo=self.seed_algo_combo.currentText(),
        )

    def start_generation(self) -> bool:
        if self.worker_thread and self.worker_thread.isRunning():
            params = self._build_params()
            if params is None:
                return False
            self._gen_queue.append(params)
            self._refresh_queue_view()
            self.status_bar.showMessage(f"Queued (position {len(self._gen_queue)})", 3000)
            return True

        params = self._build_params()
        if params is None:
            return False
        return self._start_with_params(params)

    def _start_with_params(self, params: GenerationParams) -> bool:
        self.worker_thread = AudioGenerationThread(params, parent=self)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.worker_thread.status_updated.connect(self.status_label.setText, Qt.QueuedConnection)
        self.worker_thread.eta_updated.connect(self.eta_label.setText, Qt.QueuedConnection)
        self.worker_thread.generation_completed.connect(self.generation_finished, Qt.QueuedConnection)
        self.worker_thread.error_occurred.connect(self.generation_error, Qt.QueuedConnection)
        self.worker_thread.cancelled.connect(self.generation_cancelled, Qt.QueuedConnection)
        self.worker_thread.finished.connect(self.on_worker_thread_finished, Qt.QueuedConnection)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater, Qt.QueuedConnection)

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.worker_thread.start()
        return True

    def cancel_generation(self):
        if self.worker_thread:
            self.worker_thread.cancel()
            self.status_label.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)

    def generation_cancelled(self):
        self.status_bar.showMessage("Generation cancelled", 3000)

    def generation_finished(self, path, metadata):
        self.base_audio_path = path
        self.current_audio_path = path
        self.progress_bar.setVisible(False)
        self.eta_label.setText("")
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.play_btn.setText("Play")

        meta_list = metadata['metadata']
        history_data = self.settings.get("history", [])
        
        self.playlist_table.setUpdatesEnabled(False)
        row = self.playlist_table.rowCount()
        for meta in meta_list:
            history_data.append(meta)
            self.playlist_table.insertRow(row)
            prompt_item = QTableWidgetItem(meta.get('prompt', ''))
            prompt_item.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 0, prompt_item)

            duration_item = QTableWidgetItem(format_time(meta.get('duration', 0.0)))
            duration_item.setTextAlignment(Qt.AlignCenter)
            duration_item.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 1, duration_item)

            self.playlist_table.setItem(row, 2, QTableWidgetItem(meta.get('timestamp', '')[:10]))
            self.playlist_table.setItem(row, 3, QTableWidgetItem(str(meta.get('seed', ''))))
            row += 1
        self.playlist_table.setUpdatesEnabled(True)
        self.playlist_table.viewport().update()

        if len(history_data) > MAX_HISTORY:
            history_data = history_data[-MAX_HISTORY:]
        self.settings["history"] = history_data
        self.save_json_settings()
        
        self.video_tab.update_history(history_data)
        self._load_audio_file(path)
        self.status_bar.showMessage(f"Generated {len(metadata['paths'])} file(s)", 5000)

    def generation_error(self, msg):
        self.progress_bar.setVisible(False)
        self.eta_label.setText("")
        QMessageBox.critical(self, "Generation Error", msg)

    def on_worker_thread_finished(self):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Ready")
        thread = self.worker_thread
        self.worker_thread = None
        if thread is not None:
            thread.deleteLater()
        gc.collect()

        if self._gen_queue:
            next_params = self._gen_queue.popleft()
            self._refresh_queue_view()
            self.status_bar.showMessage("Starting next queued job...", 3000)
            self._start_with_params(next_params)

    def _on_video_tab_completed(self, path: str):
        self.status_bar.showMessage(f"Video saved to {path}", 5000)

    def toggle_playback(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.stop()
        else:
            current_source = self.media_player.source().toLocalFile()
            target_source = os.path.abspath(self.current_audio_path)
            if current_source != target_source:
                self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))
            self.media_player.play()

    def on_playback_state_changed(self, state):
        self.play_btn.setText("Stop" if state == QMediaPlayer.PlayingState else "Play")

    def play_from_playlist(self, item):
        path_item = self.playlist_table.item(item.row(), 1)
        path = path_item.toolTip() if path_item else ""
        if path and os.path.exists(path):
            self._load_audio_file(path)
            self.toggle_playback()

    def _load_audio_file(self, path):
        if not path or not os.path.exists(path):
            return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.stop()
        self.base_audio_path = path
        self.current_audio_path = path
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.play_btn.setText("Play")
        
        if self.tab_widget.currentIndex() == 3:
            self.video_tab.set_audio_path(path, emit_signal=False)
        else:
            self.waveform_widget.set_audio(path)
            
        self.media_player.setSource(QUrl.fromLocalFile(path))
        
        if hasattr(self, 'video_tab') and self.video_tab and self.video_tab.current_audio_path != path:
            self.video_tab.set_audio_path(path, emit_signal=False)

    def on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.play_btn.setText("Play")
            self.waveform_widget.set_playback_position(0.0)

    def on_position_changed(self, position):
        duration = self.media_player.duration()
        if duration > 0:
            self.time_label.setText(f"{format_time(position / 1000)} / {format_time(duration / 1000)}")
            self.waveform_widget.set_playback_position(position / duration)

    def on_duration_changed(self, duration):
        if duration > 0:
            self.time_label.setText(f"00:00 / {format_time(duration / 1000)}")

    def _seek_to(self, pos: float):
        if self.media_player.duration() > 0:
            self.media_player.setPosition(int(pos * self.media_player.duration()))

    def set_volume(self, val):
        self.audio_output.setVolume(val / 100.0)
        self.vol_label.setText(f"{val}%")

    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "WAV Files (*.wav)")
        if path:
            self._load_audio_file(path)

    def save_audio(self):
        if not self.base_audio_path:
            return
        default_name = os.path.basename(self.base_audio_path)
        path, _ = QFileDialog.getSaveFileName(self, "Save Audio", default_name, "WAV Audio (*.wav)")
        if path:
            try:
                shutil.copy2(self.base_audio_path, path)
                self.status_bar.showMessage("Audio saved successfully!", 3000)
            except Exception as e:
                logger.error(f"Failed to save audio: {e}")
                QMessageBox.warning(self, "Save Failed", f"Could not save file:\n{e}")

    def load_settings(self):
        self.prompt_input.setPlainText(self.settings.get("prompt", ""))
        self.neg_prompt_input.setText(self.settings.get("negative_prompt", ""))
        self.model_combo.setCurrentText(self.settings.get("model", "cvssp/audioldm2"))
        self.duration_spin.setValue(float(self.settings.get("duration", 10.0)))
        self.steps_spin.setValue(int(self.settings.get("steps", 200)))
        self.guidance_spin.setValue(float(self.settings.get("guidance", 3.5)))
        self.seed_spin.setValue(int(self.settings.get("seed", -1)))
        self.variations_spin.setValue(int(self.settings.get("variations", 1)))
        self.output_dir_edit.setText(self.settings.get("output_dir", os.path.join(os.getcwd(), "generated_audio")))
        self.cache_dir_edit.setText(self.settings.get("cache_dir", ""))
        self.cpu_offload_check.setChecked(self.settings.get("cpu_offload", False))

        self.multi_prompt_check.setChecked(self.settings.get("multi_prompt_check", False))
        self.multi_prompt_edit.setPlainText(self.settings.get("multi_prompts", ""))
        self.toggle_multi_prompt(self.multi_prompt_check.isChecked())
        self.seed_algo_combo.setCurrentText(self.settings.get("seed_algo", "Sequential"))
        self.update_duration_info()

        history_data = self.settings.get("history", [])
        
        self.playlist_table.setUpdatesEnabled(False)
        for meta in history_data:
            row = self.playlist_table.rowCount()
            self.playlist_table.insertRow(row)
            prompt_item = QTableWidgetItem(meta.get('prompt', ''))
            prompt_item.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 0, prompt_item)

            duration_item = QTableWidgetItem(format_time(meta.get('duration', 0.0)))
            duration_item.setTextAlignment(Qt.AlignCenter)
            duration_item.setToolTip(meta.get('path', ''))
            self.playlist_table.setItem(row, 1, duration_item)

            self.playlist_table.setItem(row, 2, QTableWidgetItem(meta.get('timestamp', '')[:10]))
            self.playlist_table.setItem(row, 3, QTableWidgetItem(str(meta.get('seed', ''))))
        self.playlist_table.setUpdatesEnabled(True)
        self.playlist_table.viewport().update()

        self.video_tab.update_history(history_data)
        self.video_tab.set_output_dir(self.settings.get("output_dir", ""))

    def closeEvent(self, event):
        # 1. Audio Generation Thread
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.status_bar.showMessage("Waiting for generation to stop...")
            self.worker_thread.wait(5000)
            if self.worker_thread.isRunning():
                logger.warning("Worker thread did not exit cleanly; terminating.")
                self.worker_thread.terminate()
                self.worker_thread.wait(1000)

        # 2. Preload Thread
        if self._preload_thread and self._preload_thread.isRunning():
            self._preload_thread.wait(5000)
            if self._preload_thread.isRunning():
                self._preload_thread.terminate()
                self._preload_thread.wait(1000)

        # 3. Video Tab Thread
        if hasattr(self, 'video_tab') and self.video_tab and self.video_tab.is_running():
            self.video_tab.cleanup()

        # 4. Waveform Loader Thread
        if hasattr(self, 'waveform_widget') and self.waveform_widget._loader_thread and self.waveform_widget._loader_thread.isRunning():
            self.waveform_widget._loader_thread.quit()
            self.waveform_widget._loader_thread.wait(2000)
            
        # 5. Preview Render Thread
        if hasattr(self, 'waveform_widget') and self.waveform_widget._preview_thread and self.waveform_widget._preview_thread.isRunning():
            self.waveform_widget._preview_thread.quit()
            self.waveform_widget._preview_thread.wait(2000)

        # 6. Unload shared PyTorch pipeline
        AudioGenerationThread.unload_pipeline()

        self.settings["prompt"] = self.prompt_input.toPlainText()
        self.settings["negative_prompt"] = self.neg_prompt_input.text()
        self.settings["model"] = self.model_combo.currentText()
        self.settings["duration"] = self.duration_spin.value()
        self.settings["steps"] = self.steps_spin.value()
        self.settings["guidance"] = self.guidance_spin.value()
        self.settings["seed"] = self.seed_spin.value()
        self.settings["variations"] = self.variations_spin.value()
        self.settings["output_dir"] = self.output_dir_edit.text().strip()
        self.settings["cpu_offload"] = self.cpu_offload_check.isChecked()
        self.settings["multi_prompt_check"] = self.multi_prompt_check.isChecked()
        self.settings["multi_prompts"] = self.multi_prompt_edit.toPlainText()
        self.settings["seed_algo"] = self.seed_algo_combo.currentText()
        self.settings["cache_dir"] = self.cache_dir_edit.text().strip()
        self.save_json_settings()
        event.accept()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setApplicationName(APP_NAME)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
        app.setStyleSheet(STYLESHEET)
        app.setFont(QFont("Segoe UI", 9))

        window = AudioLDM2Studio()
        window.show()

        def signal_handler(sig, frame):
            logger.info("SIGINT received, quitting...")
            if window:
                window.close()

        signal.signal(signal.SIGINT, signal_handler)
        sig_timer = QTimer()
        sig_timer.start(250)
        sig_timer.timeout.connect(lambda: None)

        sys.exit(app.exec())
    except Exception:
        logger.critical("Fatal Startup Error", exc_info=True)
        sys.exit(1)