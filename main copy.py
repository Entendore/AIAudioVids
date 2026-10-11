#!/usr/bin/env python3
"""
AudioLDM2 Studio Pro v4.0.0
Complete ASMR video creation suite with:
- 13 audio-reactive visualizers (Electric Sheep, Chaos Game, Reactive Diffusion,
  Rain, Zen Mandala, Soft Orbs, Breathing Wave, Fireplace, Water Ripples,
  Frequency Bars, Radial Spectrum, Spectrogram Waterfall, Waveform Landscape)
- YouTube-optimized MP4 export (H.264 + AAC + faststart)
- ASMR audio processing (EQ, compression, noise floor, LUFS normalization)
- Audio looping (generate 10min, loop to 8hrs)
- Sound layering (mix multiple ASMR sounds)
- Batch ASMR queue with crash recovery
- Custom theme editor
- System tray + desktop notifications
- Segment-based encoding with resume
- Thumbnail capture
- YouTube metadata generation (title, description, tags, chapters)
"""
import sys, os, json, time, shutil, logging, math, csv, gc, copy
import subprocess, threading, inspect, platform, base64, zipfile, re
import itertools, colorsys, struct, tempfile
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from logging.handlers import RotatingFileHandler
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile
import scipy.signal
from scipy.ndimage import convolve as ndimage_convolve

from PySide6.QtCore import (Qt, QThread, Signal, QObject, QUrl, QTimer, QPoint,
    QPointF, QSize, QEvent, QByteArray, QRectF, QRect)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QProgressBar,
    QFileDialog, QMessageBox, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QDialog, QFormLayout, QMenu, QStyle,
    QInputDialog, QPlainTextEdit, QSizePolicy, QFrame, QToolButton, QStatusBar,
    QListWidget, QListWidgetItem, QScrollArea, QToolTip, QColorDialog,
    QSystemTrayIcon)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import (QAction, QIcon, QPixmap, QColor, QPalette, QFont,
    QPainter, QPen, QMouseEvent, QKeySequence, QShortcut, QDesktopServices,
    QLinearGradient, QPainterPath, QPolygon, QBrush, QImage)

try:
    import torch
    from diffusers import AudioLDM2Pipeline
    import transformers, diffusers
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════
APP_NAME = "AudioLDM2 Studio Pro"
APP_VERSION = "4.0.0"
SETTINGS_ORG = "AudioLDM2"; SETTINGS_APP = "Studio"
DEFAULT_CACHE_DIR = "E:\\cacheAI"
CWD = os.getcwd()
BATCH_STATE_FILE = os.path.join(CWD, "batch_state.json")
LOG_FILE = os.path.join(CWD, f"{APP_NAME}.log")

DEFAULT_MODELS = ["cvssp/audioldm2", "cvssp/audioldm2-music",
    "cvssp/audioldm2-gigaspeech", "cvssp/audioldm2-ljspeech"]

DEFAULT_PROMPT_PRESETS = [
    "Musical constellations twinkling in the night sky",
    "Rain drops falling on a tin roof with distant thunder",
    "A cinematic orchestral swell with epic drums",
    "Birds chirping in a peaceful forest at dawn",
    "Electronic synth wave with retro 80s vibe",
    "Ocean waves crashing on a rocky shore",
    "A jazz saxophone playing in a smoky bar",
    "Wind howling through mountain peaks",
]
NEGATIVE_PROMPT_PRESETS = [
    "noise, distortion, artifacts, low quality",
    "static, hum, buzzing, interference",
    "echo, reverb, muffled sound",
    "clipping, peaking, distortion",
]

OVERLAP_S = 1.0; MIN_CHUNK_S = 2.0; MAX_HISTORY = 500
MAX_PROMPT_HISTORY = 50; FADE_DURATION_S = 0.05; TRIM_THRESHOLD = 0.01
ETA_SMOOTHING_ALPHA = 0.3; AUTO_SAVE_INTERVAL_MS = 60_000
GPU_MEM_POLL_MS = 2_000; PROMPT_MATRIX_MAX = 200
THUMB_WIDTH = 120; THUMB_HEIGHT = 28; THUMB_BARS = 60
THUMB_QUEUE_MAXLEN = 256; AUDIO_CACHE_MAX_ENTRIES = 8

VIDEO_RESOLUTIONS = {
    "480p (854×480)": (854, 480), "720p (1280×720)": (1280, 720),
    "1080p (1920×1080)": (1920, 1080), "1440p (2560×1440)": (2560, 1440),
    "4K (3840×2160)": (3840, 2160),
}
VIDEO_FPS_OPTIONS = [24, 30, 60]
VIDEO_STYLES = [
    "Electric Sheep", "Frequency Bars", "Radial Spectrum", "Spectrogram Waterfall",
    "Waveform Landscape", "Chaos Game", "Reactive Diffusion",
    "Rain Particles", "Zen Mandala", "Soft Orbs", "Breathing Wave",
    "Fireplace", "Water Ripples",
]
VIDEO_COLOR_THEMES = {
    "Cosmic":  {"hue_base": 260, "hue_range": 120, "sat": 220, "val": 255},
    "Fire":    {"hue_base": 0,   "hue_range": 60,  "sat": 240, "val": 255},
    "Ocean":   {"hue_base": 180, "hue_range": 80,  "sat": 200, "val": 255},
    "Forest":  {"hue_base": 100, "hue_range": 80,  "sat": 200, "val": 255},
    "Sunset":  {"hue_base": 20,  "hue_range": 50,  "sat": 230, "val": 255},
    "Neon":    {"hue_base": 300, "hue_range": 120, "sat": 255, "val": 255},
    "Rainbow": {"hue_base": 0,   "hue_range": 360, "sat": 220, "val": 255},
}
DIFFUSION_PRESETS = {
    "Solitons": {"feed": 0.030, "kill": 0.062}, "Maze": {"feed": 0.029, "kill": 0.058},
    "Spots": {"feed": 0.025, "kill": 0.060}, "Bubbles": {"feed": 0.040, "kill": 0.060},
    "Worms": {"feed": 0.055, "kill": 0.062}, "Holes": {"feed": 0.060, "kill": 0.062},
    "Mazes": {"feed": 0.039, "kill": 0.058},
}
YOUTUBE_ENCODING_PRESETS = {
    "ASMR 1080p 24fps (Calm)": {
        "resolution": "1080p (1920×1080)", "fps": 24, "preset": "slow",
        "crf": 20, "audio_bitrate": "320k",
        "extra_args": ["-profile:v","high","-level","4.0","-ar","48000",
            "-bf","2","-g","48","-keyint_min","48"],
    },
    "ASMR 1080p 30fps (Standard)": {
        "resolution": "1080p (1920×1080)", "fps": 30, "preset": "slow",
        "crf": 19, "audio_bitrate": "320k",
        "extra_args": ["-profile:v","high","-level","4.0","-ar","48000",
            "-bf","2","-g","60","-keyint_min","60"],
    },
    "ASMR 4K 24fps (Ultra Calm)": {
        "resolution": "4K (3840×2160)", "fps": 24, "preset": "slow",
        "crf": 18, "audio_bitrate": "320k",
        "extra_args": ["-profile:v","high","-level","5.1","-ar","48000",
            "-bf","2","-g","48","-keyint_min","48"],
    },
    "YouTube 1080p 30fps (General)": {
        "resolution": "1080p (1920×1080)", "fps": 30, "preset": "medium",
        "crf": 18, "audio_bitrate": "192k",
        "extra_args": ["-profile:v","high","-level","4.0"],
    },
    "YouTube 4K 30fps (Ultra HD)": {
        "resolution": "4K (3840×2160)", "fps": 30, "preset": "medium",
        "crf": 18, "audio_bitrate": "256k",
        "extra_args": ["-profile:v","high","-level","5.1"],
    },
}
RENDER_QUALITY_PRESETS = {
    "Fast (Low Particles)": {"particles_mult": 0.5, "sim_scale": 8, "glow": 0.15, "steps": 4},
    "Quality (Balanced)": {"particles_mult": 1.0, "sim_scale": 6, "glow": 0.22, "steps": 8},
    "Ultra (High Density)": {"particles_mult": 2.0, "sim_scale": 4, "glow": 0.35, "steps": 12},
}
ASMR_PRESETS = {
    "Rain for Sleep (1 Hour)": {
        "prompt": "Gentle steady rain falling on leaves and grass with occasional distant thunder rumble, continuous, calming, sleep aid, ASMR, high quality recording",
        "negative_prompt": "sudden loud noises, distortion, harsh sounds, music, speech, abrupt changes",
        "duration": 600.0, "target_duration": 3600.0, "loop": True, "crossfade": 3.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Rain Particles", "color_theme": "Ocean",
        "motion_speed": 0.3, "softness": 0.6, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -50,
    },
    "Ocean Waves (2 Hours)": {
        "prompt": "Rhythmic ocean waves gently crashing on a sandy beach, receding water sounds, seagulls distant, continuous, meditative, sleep aid, ASMR",
        "negative_prompt": "sudden loud noises, distortion, harsh sounds, music, speech",
        "duration": 600.0, "target_duration": 7200.0, "loop": True, "crossfade": 4.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Breathing Wave", "color_theme": "Ocean",
        "motion_speed": 0.2, "softness": 0.7, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -50,
    },
    "Fireplace Ambience (3 Hours)": {
        "prompt": "Crackling fireplace with wood popping sounds, ember hissing, warm crackling, continuous, cozy, sleep aid, ASMR",
        "negative_prompt": "sudden loud noises, distortion, harsh sounds, music, speech",
        "duration": 600.0, "target_duration": 10800.0, "loop": True, "crossfade": 3.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Fireplace", "color_theme": "Fire",
        "motion_speed": 0.5, "softness": 0.4, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 6000, "noise_floor_db": -45,
    },
    "Zen Meditation Bowl (1 Hour)": {
        "prompt": "Tibetan singing bowl meditation tone, resonant, peaceful, gentle striking, continuous, meditative, ASMR",
        "negative_prompt": "noise, distortion, harsh sounds, sudden changes, music melody",
        "duration": 600.0, "target_duration": 3600.0, "loop": True, "crossfade": 5.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Zen Mandala", "color_theme": "Cosmic",
        "motion_speed": 0.15, "softness": 0.5, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -55,
    },
    "Deep Space Sleep (2 Hours)": {
        "prompt": "Deep space ambient drone, cosmic resonance, ethereal pad, very slow evolution, continuous, sleep aid, ASMR",
        "negative_prompt": "noise, distortion, harsh sounds, sudden changes, percussion, drums",
        "duration": 600.0, "target_duration": 7200.0, "loop": True, "crossfade": 5.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Soft Orbs", "color_theme": "Cosmic",
        "motion_speed": 0.1, "softness": 0.8, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -50,
    },
    "Water Drops in Cave (1 Hour)": {
        "prompt": "Water drops falling in a large cave, echoing, rhythmic, dripping, cave ambience, continuous, calming, ASMR",
        "negative_prompt": "noise, distortion, harsh sounds, music, speech",
        "duration": 600.0, "target_duration": 3600.0, "loop": True, "crossfade": 3.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Water Ripples", "color_theme": "Ocean",
        "motion_speed": 0.25, "softness": 0.6, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -50,
    },
    "White Noise Sleep (8 Hours)": {
        "prompt": "Smooth white noise, gentle hiss, consistent, continuous, sleep aid, ASMR, no variation",
        "negative_prompt": "sudden changes, distortion, harsh sounds, music, speech, percussion",
        "duration": 600.0, "target_duration": 28800.0, "loop": True, "crossfade": 5.0,
        "steps": 200, "guidance": 2.0, "chunk_size": 10.0,
        "visual_style": "Soft Orbs", "color_theme": "Neon",
        "motion_speed": 0.05, "softness": 0.9, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -45,
    },
    "Forest Rain & Birds (2 Hours)": {
        "prompt": "Gentle rain in a forest with birds chirping, leaves rustling, distant stream, continuous, nature, calming, ASMR",
        "negative_prompt": "sudden loud noises, distortion, harsh sounds, music, speech",
        "duration": 600.0, "target_duration": 7200.0, "loop": True, "crossfade": 4.0,
        "steps": 200, "guidance": 3.5, "chunk_size": 10.0,
        "visual_style": "Rain Particles", "color_theme": "Forest",
        "motion_speed": 0.25, "softness": 0.5, "encoding": "ASMR 1080p 24fps (Calm)",
        "process_asmr": True, "lowpass_hz": 8000, "noise_floor_db": -50,
    },
}
ASMR_VISUAL_STYLES = ["Rain Particles", "Zen Mandala", "Soft Orbs",
    "Breathing Wave", "Fireplace", "Water Ripples"]

STYLESHEET = """
QMainWindow { background-color: #1a1a2e; }
QWidget { color: #d0d0e0; }
QGroupBox { border: 1px solid #2a2a4a; border-radius: 6px; margin-top: 12px;
    padding-top: 12px; font-weight: bold; color: #8888bb; background-color: #16213e; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; background-color: #1a1a2e; }
QTabWidget::pane { border: 1px solid #2a2a4a; border-radius: 4px;
    background-color: #16213e; top: -1px; }
QTabBar::tab { background: #1a1a2e; color: #8888bb; padding: 8px 20px; margin-right: 2px;
    border-top-left-radius: 4px; border-top-right-radius: 4px; min-width: 80px;
    font-weight: bold; border: 1px solid #2a2a4a; border-bottom: none; }
QTabBar::tab:selected { background: #0f3460; color: #e0e0ff; border-color: #e94560; }
QTabBar::tab:hover:!selected { background: #2a2a4a; color: #e0e0ff; }
QPushButton { background-color: #0f3460; border: 1px solid #1a1a4a; color: #e0e0e0;
    padding: 8px 16px; border-radius: 4px; min-height: 22px; font-weight: 500; }
QPushButton:hover { background-color: #1a4a80; border-color: #e94560; }
QPushButton:pressed { background-color: #16213e; }
QPushButton:disabled { background-color: #1a1a2e; color: #555570; border-color: #2a2a4a; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background-color: #0d1b2a; border: 1px solid #2a2a4a; padding: 6px 8px;
    border-radius: 4px; color: #e0e0e0; selection-background-color: #e94560; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #e94560; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background-color: #0d1b2a; border: 1px solid #2a2a4a;
    selection-background-color: #e94560; color: #e0e0e0; }
QTableWidget { gridline-color: #2a2a4a; background-color: #16213e;
    alternate-background-color: #0d1b2a; border: 1px solid #2a2a4a; border-radius: 4px;
    color: #d0d0e0; }
QHeaderView::section { background-color: #0f3460; padding: 8px; border: none;
    font-weight: bold; color: #8888bb; }
QProgressBar { border: 1px solid #2a2a4a; border-radius: 4px; text-align: center;
    background-color: #0d1b2a; color: white; min-height: 24px; }
QProgressBar::chunk { background-color: #e94560; border-radius: 2px; }
QScrollBar:vertical { background: #1a1a2e; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #0f3460; min-height: 24px; border-radius: 6px; }
QScrollBar::handle:vertical:hover { background: #e94560; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1a1a2e; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #0f3460; min-width: 24px; border-radius: 6px; }
QScrollBar::handle:horizontal:hover { background: #e94560; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollArea { border: none; background-color: transparent; }
QCheckBox { color: #d0d0e0; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #2a2a4a; background: #0d1b2a; }
QCheckBox::indicator:checked { background: #e94560; border-color: #e94560; }
QLabel { color: #c0c0d0; }
QToolButton { background-color: #0f3460; border: 1px solid #1a1a4a; border-radius: 4px;
    padding: 4px; color: #e0e0e0; }
QToolButton:hover { background-color: #1a4a80; border-color: #e94560; }
QStatusBar { background-color: #0d1b2a; color: #8888bb; } QStatusBar::item { border: none; }
QMenuBar { background-color: #0d1b2a; color: #c0c0d0; border-bottom: 1px solid #2a2a4a; }
QMenuBar::item:selected { background-color: #0f3460; }
QMenu { background-color: #16213e; border: 1px solid #2a2a4a; color: #d0d0d0; }
QMenu::item:selected { background-color: #e94560; }
QSlider::groove:horizontal { border: 1px solid #2a2a4a; height: 6px;
    background: #0d1b2a; border-radius: 3px; }
QSlider::handle:horizontal { background: #e94560; border: 1px solid #e94560;
    width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #ff5577; }
QSplitter::handle { background-color: #2a2a4a; } QSplitter::handle:horizontal { width: 3px; }
"""

# ═══════════════════════════════════════════════════════════════════════
# JSON Settings
# ═══════════════════════════════════════════════════════════════════════
class JsonSettings:
    def __init__(self, path):
        self.path = path; self.data = {}; self._dirty = False
        self._lock = threading.Lock(); self.load()
    def load(self):
        with self._lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'r', encoding='utf-8') as f: self.data = json.load(f)
                except Exception as e: logging.error(f"Failed to load settings: {e}"); self.data = {}
    def save(self):
        with self._lock:
            try:
                tmp = self.path + ".tmp"
                with open(tmp, 'w', encoding='utf-8') as f: json.dump(self.data, f, indent=4)
                os.replace(tmp, self.path); self._dirty = False
            except Exception as e: logging.error(f"Failed to save settings: {e}")
    def value(self, key, default=None, type=None):
        with self._lock: val = self.data.get(key, default)
        if val is None:
            return copy.deepcopy(default) if isinstance(default, (list, dict)) else default
        if isinstance(val, str) and val.startswith("base64:"):
            return QByteArray.fromBase64(val[7:].encode('utf-8'))
        if type == bool:
            return val.lower() in ('true','1','yes') if isinstance(val, str) else bool(val)
        if type == int:
            try: return int(val)
            except: return default
        if type == float:
            try: return float(val)
            except: return default
        return copy.deepcopy(val) if isinstance(val, (list, dict)) else val
    def setValue(self, key, value):
        with self._lock:
            if isinstance(value, QByteArray):
                value = "base64:" + base64.b64encode(value.data()).decode('utf-8')
            elif isinstance(value, (list, dict)): value = copy.deepcopy(value)
            if self.data.get(key) != value: self.data[key] = value; self._dirty = True
    def remove(self, key):
        with self._lock:
            if key in self.data: del self.data[key]; self._dirty = True
        self.save()
    def sync(self):
        if self._dirty: self.save()
    def clear(self):
        with self._lock: self.data = {}; self._dirty = True
        self.save()

# ═══════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(APP_NAME)
try:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, mode='a', encoding='utf-8')
except Exception:
    handler = logging.FileHandler("audioldm2_studio.log", mode='a', encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logger.addHandler(handler)

def log_system_info():
    logger.info("=" * 60)
    logger.info(f"{APP_NAME} v{APP_VERSION} starting")
    logger.info(f"Python: {sys.version}"); logger.info(f"OS: {platform.platform()}")
    if ML_AVAILABLE:
        logger.info(f"PyTorch: {torch.__version__}; Diffusers: {diffusers.__version__}; Transformers: {transformers.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════
def sanitize_filename(text, max_length=30):
    safe = "".join(c for c in text if c.isalnum() or c in (' ','_','-')).rstrip()
    return safe[:max_length].replace(' ', '_') if safe else "audio"

def format_time(seconds):
    try:
        t = max(0, int(float(seconds))); m, s = divmod(t, 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
    except: return "00:00"

def normalize_audio_data(data):
    if data is None or data.size == 0: return np.zeros(0, dtype=np.float32)
    if data.ndim > 1: data = data.mean(axis=1)
    if data.dtype == np.int16: return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32: return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8: return (data.astype(np.float32) - 128.0) / 128.0
    if data.dtype == np.float32: return np.ascontiguousarray(data, dtype=np.float32)
    return data.astype(np.float32)

def normalize_audio(audio, target_db=-3.0):
    if audio.size == 0: return audio.copy()
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    else: audio = np.ascontiguousarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak == 0: return audio
    return audio * (10 ** (target_db / 20.0) / peak)

def apply_fade(audio, sample_rate, fade_in_s=0.05, fade_out_s=0.05):
    if audio.dtype != np.float32: audio = audio.astype(np.float32)
    else: audio = np.ascontiguousarray(audio)
    if audio.size == 0: return audio
    fi, fo = int(fade_in_s * sample_rate), int(fade_out_s * sample_rate)
    if fi + fo > len(audio): fi = len(audio) // 2; fo = len(audio) - fi
    if fi > 0: audio[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0: audio[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return audio

def trim_silence(audio, threshold=0.01):
    if audio.size == 0: return audio.copy()
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    else: audio = np.ascontiguousarray(audio, dtype=np.float32)
    above = np.where(np.abs(audio) > threshold)[0]
    if above.size == 0: return audio
    return np.ascontiguousarray(audio[above[0]:above[-1]+1])

def hsv_to_rgb_array(h, s, v):
    h = (np.asarray(h, dtype=np.float32) % 360.0) / 60.0
    c = v / 255.0 * s / 255.0; x = c * (1.0 - np.abs(h % 2.0 - 1.0)); m = v / 255.0 - c
    sector = h.astype(np.int32) % 6
    r1 = np.choose(sector, [c, x, 0.0, 0.0, x, c])
    g1 = np.choose(sector, [x, c, c, x, 0.0, 0.0])
    b1 = np.choose(sector, [0.0, 0.0, x, c, c, x])
    return np.clip(np.stack([r1+m, g1+m, b1+m], axis=-1) * 255.0, 0, 255).astype(np.uint8)

def create_seamless_loop(audio, sample_rate, target_duration_s, crossfade_s=3.0):
    source_duration = len(audio) / sample_rate
    if source_duration >= target_duration_s:
        return audio[:int(target_duration_s * sample_rate)]
    cf = int(min(crossfade_s, source_duration / 4) * sample_rate)
    if cf < 2: cf = 2
    fo = np.cos(np.linspace(0, np.pi/2, cf, dtype=np.float32))**2
    fi = np.sin(np.linspace(0, np.pi/2, cf, dtype=np.float32))**2
    main, end, begin = audio[:-cf], audio[-cf:], audio[:cf]
    cross = end * fo + begin * fi
    loop_unit = np.concatenate([main, cross])
    n_loops = math.ceil(target_duration_s / (len(loop_unit) / sample_rate))
    looped = np.tile(loop_unit, n_loops)[:int(target_duration_s * sample_rate)]
    edge = int(min(2.0, target_duration_s / 20) * sample_rate)
    if edge > 0 and edge * 2 < len(looped):
        looped[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)
        looped[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    return np.ascontiguousarray(looped, dtype=np.float32)

def process_asmr_audio(audio, sample_rate, lowpass_hz=8000.0, highpass_hz=20.0,
                       compress_ratio=3.0, noise_floor_db=-50.0, target_lufs=-16.0,
                       add_pink_noise=True, gentle_compression=True):
    from scipy.signal import butter, sosfilt, lfilter
    audio = np.ascontiguousarray(audio, dtype=np.float32); nyq = 0.5 * sample_rate
    if 0 < highpass_hz < nyq:
        audio = sosfilt(butter(2, highpass_hz/nyq, 'hp', output='sos'), audio).astype(np.float32)
    if 0 < lowpass_hz < nyq:
        audio = sosfilt(butter(4, min(lowpass_hz/nyq, 0.95), 'lp', output='sos'), audio).astype(np.float32)
    if gentle_compression:
        threshold = 0.5; above = np.abs(audio) > threshold
        if np.any(above):
            excess = np.abs(audio[above]) - threshold
            audio[above] = np.sign(audio[above]) * (threshold + excess / compress_ratio)
    if add_pink_noise and noise_floor_db > -80:
        nl = 10 ** (noise_floor_db / 20.0)
        white = np.random.randn(len(audio)).astype(np.float32)
        b = [0.049922035, -0.095074837, 0.050897763]; a = [1.0, -1.9131741, 0.91182514]
        pink = lfilter(b, a, white).astype(np.float32); pk = np.max(np.abs(pink))
        if pk > 0: audio = audio + pink / pk * nl
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 0:
        gain = 10 ** ((target_lufs - (20 * math.log10(rms + 1e-10) - 0.691)) / 20.0)
        audio *= gain
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98: audio *= 0.98 / peak
    return audio.astype(np.float32)

def check_disk_space(path, required_gb):
    try:
        u = shutil.disk_usage(os.path.dirname(path) or '.'); return u.free / (1024**3) >= required_gb, u.free / (1024**3)
    except: return True, 0.0

def estimate_video_file_size(duration_s, width, height, fps, crf):
    base_mbps = (width * height) / (1920*1080) * 12 * (2 ** ((18 - crf) / 6))
    return (base_mbps + 0.32) * duration_s / 8 / 1024

def generate_youtube_chapters(duration_s, interval_s=900):
    chapters = []; t = 0.0
    while t < duration_s:
        h, m, s = int(t//3600), int((t%3600)//60), int(t%60)
        ts = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        pct = t / duration_s if duration_s > 0 else 0
        name = "Intro" if t == 0 else ("Beginning" if pct < 0.1 else
              "Settling In" if pct < 0.25 else ("Deep Relaxation" if pct < 0.5 else
              "Drifting Off" if pct < 0.75 else ("Deep Sleep" if pct < 0.9 else "Final Moments")))
        chapters.append(f"{ts} {name}"); t += interval_s
    return chapters

def get_audio_features(audio, sr, time_s, window_ms=50, n_bands=8):
    ws = min(int(sr * window_ms / 1000), len(audio)); si = int(time_s * sr)
    if si + ws > len(audio):
        window = audio[si:] if si < len(audio) else np.zeros(ws, dtype=np.float32)
        if len(window) < ws: window = np.pad(window, (0, ws - len(window)))
    else: window = audio[si:si+ws]
    amp = float(np.sqrt(np.mean(window**2)))
    if ws >= 64:
        fft = np.fft.rfft(window * np.hanning(len(window)))
        mag = np.abs(fft[:len(fft)//2]); bs = max(1, len(mag)//n_bands)
        bands = np.array([np.mean(mag[i*bs:(i+1)*bs]) for i in range(n_bands)], dtype=np.float32)
        mx = bands.max()
        if mx > 0: bands /= mx
    else: bands = np.zeros(n_bands, dtype=np.float32)
    return amp, bands, window

# ═══════════════════════════════════════════════════════════════════════
# Audio Data Cache
# ═══════════════════════════════════════════════════════════════════════
class AudioDataCache:
    def __init__(self, max_entries=8):
        self._max = max_entries; self._cache = deque(); self._data = {}; self._lock = threading.Lock()
    def get(self, path):
        if not path: return None
        with self._lock:
            if path in self._data:
                self._cache.remove(path); self._cache.append(path); return self._data[path]
        return None
    def put(self, path, sr, data):
        if not path: return
        with self._lock:
            if path in self._data: self._cache.remove(path)
            self._cache.append(path); self._data[path] = (sr, data)
            while len(self._cache) > self._max: self._data.pop(self._cache.popleft(), None)
    def invalidate(self, path):
        with self._lock:
            if path in self._data: self._cache.remove(path); self._data.pop(path, None)
    def clear(self):
        with self._lock: self._cache.clear(); self._data.clear()
    @staticmethod
    def load_from_disk(path):
        if not path or not os.path.exists(path): return None
        try:
            sr, data = wavfile.read(path); data = normalize_audio_data(data)
            return (sr, np.ascontiguousarray(data, dtype=np.float32)) if data.size > 0 else None
        except Exception as e:
            logger.error(f"Failed to load audio from {path}: {e}"); return None

# ═══════════════════════════════════════════════════════════════════════
# State Machine & Data Models
# ═══════════════════════════════════════════════════════════════════════
class BatchState(Enum):
    IDLE = 0; RUNNING = 1; CANCELLED = 2; COMPLETED = 3

@dataclass
class GenerationParams:
    prompt: str; negative_prompt: str; model_name: str; duration: float
    steps: int; guidance: float; seed: int; device: str; cache_dir: str; output_dir: str
    num_variations: int = 1; normalize: bool = True; fade_in: bool = True
    fade_out: bool = True; trim: bool = True; chunk_size: float = 10.0
    use_cpu_offload: bool = False; section: str = ""
    high_pass_freq: float = 0.0; low_pass_freq: float = 0.0

# ═══════════════════════════════════════════════════════════════════════
# Audio Generation Worker
# ═══════════════════════════════════════════════════════════════════════
class AudioGenerationWorker(QObject):
    progress_updated = Signal(int); status_updated = Signal(str); eta_updated = Signal(str)
    generation_completed = Signal(str, dict); error_occurred = Signal(str); cancelled = Signal()
    _shared_pipe = None; _shared_pipe_model = None; _shared_pipe_device = None
    _shared_pipe_offload = None; _pipe_lock = threading.Lock()

    def __init__(self, params):
        super().__init__(); self.params = params; self._cancel_event = threading.Event()
        self.pipe = None; self.ema_step_time = None; self.last_step_time = None

    @property
    def is_cancelled(self): return self._cancel_event.is_set()
    @is_cancelled.setter
    def is_cancelled(self, v): self._cancel_event.set() if v else self._cancel_event.clear()

    def _load_pipeline(self, actual_device):
        with self._pipe_lock:
            if (AudioGenerationWorker._shared_pipe is not None and
                AudioGenerationWorker._shared_pipe_model == self.params.model_name and
                AudioGenerationWorker._shared_pipe_device == actual_device and
                AudioGenerationWorker._shared_pipe_offload == self.params.use_cpu_offload and
                not self.is_cancelled):
                self.pipe = AudioGenerationWorker._shared_pipe; return
            self._unload_unsafe(); self.status_updated.emit("Loading model..."); self.progress_updated.emit(5)
            dtype = torch.float16 if actual_device != "cpu" else torch.float32
            kw = {"torch_dtype": dtype, "cache_dir": self.params.cache_dir}
            if dtype == torch.float16: kw["variant"] = "fp16"
            pipe = None; last_err = None
            for attempt in [{"use_safetensors": True}, {"use_safetensors": False}]:
                if self.is_cancelled: raise InterruptedError("Cancelled")
                try:
                    pipe = AudioLDM2Pipeline.from_pretrained(self.params.model_name, **kw, **attempt); break
                except Exception as e:
                    if "variant" in str(e).lower() or "fp16" in str(e).lower():
                        kw.pop("variant", None)
                        try: pipe = AudioLDM2Pipeline.from_pretrained(self.params.model_name, **kw, **attempt); break
                        except Exception as e2: last_err = e2
                    else: last_err = e
                    pipe = None; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            if pipe is None: raise RuntimeError(f"Failed to load model: {last_err}")
            if self.is_cancelled: del pipe; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None; raise InterruptedError("Cancelled")
            self.status_updated.emit("Configuring model..."); self.progress_updated.emit(10)
            try:
                if self.params.use_cpu_offload and actual_device != "cpu":
                    if not hasattr(pipe, 'enable_model_cpu_offload'): raise RuntimeError("Install accelerate: pip install accelerate")
                    pipe.enable_model_cpu_offload()
                else: pipe = pipe.to(actual_device)
                if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_tiling"): pipe.enable_vae_tiling()
            except RuntimeError as e:
                del pipe; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if "out of memory" in str(e).lower(): raise RuntimeError("GPU OOM")
                raise
            AudioGenerationWorker._shared_pipe = pipe
            AudioGenerationWorker._shared_pipe_model = self.params.model_name
            AudioGenerationWorker._shared_pipe_device = actual_device
            AudioGenerationWorker._shared_pipe_offload = self.params.use_cpu_offload
            self.pipe = pipe

    def run(self):
        gen_start = time.time()
        try:
            p = self.params; cache_dir = os.path.normpath(p.cache_dir or DEFAULT_CACHE_DIR)
            os.makedirs(cache_dir, exist_ok=True)
            final_out = os.path.join(p.output_dir, sanitize_filename(p.section, 50)) if p.section else p.output_dir
            os.makedirs(final_out, exist_ok=True)
            for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE"]: os.environ[k] = cache_dir
            dev = p.device
            if dev == "auto": dev = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            self._load_pipeline(dev); self.status_updated.emit("Generating..."); self.progress_updated.emit(15)
            saved = []; meta_list = []; MAX_C = max(MIN_CHUNK_S, p.chunk_size); chunking = p.duration > MAX_C
            for v in range(p.num_variations):
                if self.is_cancelled: raise InterruptedError("Cancelled")
                base_seed = p.seed + v if p.seed >= 0 else int(time.time() * 1000) % (2**31) + v
                vs = time.time(); parts = []
                sr = getattr(getattr(getattr(self.pipe, "vae", None), "config", None), "sample_rate", 16000)
                self.ema_step_time = None; self.last_step_time = None
                nc = math.ceil((p.duration - OVERLAP_S) / (MAX_C - OVERLAP_S)) if chunking else 1
                def cb(step, ts, lat, c=0):
                    if self.is_cancelled: raise InterruptedError("Cancelled")
                    step, c, nc_i, ps, vi, nv = int(step), int(c), int(nc), int(p.steps), int(v), int(p.num_variations)
                    ct = time.time(); bp = (step+1)/ps; cp = (c+bp)/nc_i
                    tp = 15 + int(75 * cp / max(nv,1)) + int(vi / max(nv,1) * 75)
                    self.progress_updated.emit(min(tp, 95))
                    if self.last_step_time:
                        dt = ct - self.last_step_time
                        if dt > 0:
                            self.ema_step_time = dt if self.ema_step_time is None else ETA_SMOOTHING_ALPHA * dt + (1-ETA_SMOOTHING_ALPHA) * self.ema_step_time
                            self.eta_updated.emit(format_time((max(0,ps-step-1) + max(0,nc_i-c-1)*ps + max(0,nv-vi-1)*nc_i*ps) * self.ema_step_time))
                    self.last_step_time = ct
                    self.status_updated.emit(f"V{vi+1}/{nv} C{c+1}/{nc_i} S{step+1}/{ps}")
                for ci in range(nc):
                    if self.is_cancelled: raise InterruptedError("Cancelled")
                    cl = MAX_C if chunking and ci < nc-1 else (max(1.0, p.duration - (nc-1)*(MAX_C-OVERLAP_S)) if chunking else p.duration)
                    cseed = (base_seed + ci * 7919) % (2**31)
                    gen_dev = "cpu" if p.use_cpu_offload else dev
                    gen = torch.Generator(device=gen_dev).manual_seed(cseed)
                    def cb_old(step, ts, lat, c=ci): cb(step, ts, lat, c)
                    def cb_new(pipe_, step, ts, kw, c=ci):
                        cb(step, ts, kw.get("latents") if isinstance(kw, dict) else None, c)
                    kwargs = dict(prompt=p.prompt, negative_prompt=p.negative_prompt if p.negative_prompt else None,
                        num_inference_steps=p.steps, audio_length_in_s=cl, guidance_scale=p.guidance, generator=gen)
                    sig = inspect.signature(self.pipe.__call__)
                    if "callback_on_step_end" in sig.parameters:
                        kwargs["callback_on_step_end"] = cb_new; kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                    else: kwargs["callback"] = cb_old; kwargs["callback_steps"] = 1
                    with torch.inference_mode():
                        ac = np.asarray(self.pipe(**kwargs).audios[0], dtype=np.float32)
                    if nc > 1: gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if not parts: parts.append(ac)
                    else:
                        ov = min(int(OVERLAP_S * sr), len(parts[-1])//2, len(ac)//2)
                        if ov <= 0: parts.append(ac)
                        else:
                            fo = np.sqrt(np.linspace(1.0, 0.0, ov, dtype=np.float32))
                            fi = np.sqrt(np.linspace(0.0, 1.0, ov, dtype=np.float32))
                            parts[-1][-ov:] = (parts[-1][-ov:]*fo + ac[:ov]*fi).astype(np.float32, copy=False)
                            parts.append(ac[ov:])
                if self.is_cancelled: raise InterruptedError("Cancelled")
                audio = np.concatenate(parts) if len(parts) > 1 else (parts[0] if parts else np.zeros(0, dtype=np.float32))
                parts.clear(); del parts; gc.collect()
                if p.high_pass_freq > 0 or p.low_pass_freq > 0:
                    from scipy.signal import butter, sosfilt; nyq = 0.5 * sr
                    if p.high_pass_freq > 0: audio = sosfilt(butter(4, min(p.high_pass_freq/nyq, 0.99), 'hp', output='sos'), audio).astype(np.float32)
                    if p.low_pass_freq > 0: audio = sosfilt(butter(4, max(p.low_pass_freq/nyq, 0.01), 'lp', output='sos'), audio).astype(np.float32)
                if p.trim: audio = trim_silence(audio)
                if p.normalize: audio = normalize_audio(audio)
                if p.fade_in or p.fade_out: audio = apply_fade(audio, sr, FADE_DURATION_S if p.fade_in else 0, FADE_DURATION_S if p.fade_out else 0)
                if audio.size == 0: raise RuntimeError("Empty audio")
                fn = f"{int(time.time())}_{sanitize_filename(p.prompt)}{'_v'+str(v+1) if p.num_variations>1 else ''}.wav"
                op = os.path.join(final_out, fn)
                wavfile.write(op, sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
                del audio; gc.collect()
                m = {'prompt':p.prompt,'negative_prompt':p.negative_prompt,'model':p.model_name,'duration':p.duration,
                     'steps':p.steps,'guidance':p.guidance,'seed':base_seed,'sample_rate':sr,'device':dev,
                     'variation':v+1,'total_variations':p.num_variations,'generation_time':time.time()-vs,
                     'timestamp':datetime.now().isoformat(),'chunked':nc>1,'num_chunks':nc,'path':op,'filename':fn}
                saved.append(op); meta_list.append(m)
            mp = os.path.join(final_out, "manifest.json"); md = []
            if os.path.exists(mp):
                try:
                    with open(mp, 'r', encoding='utf-8') as f: md = json.load(f)
                except: md = []
            md.extend(meta_list)
            try:
                with open(mp, 'w', encoding='utf-8') as f: json.dump(md, f, indent=4)
            except: pass
            self.progress_updated.emit(100); self.status_updated.emit("Complete"); self.eta_updated.emit("")
            self.generation_completed.emit(saved[0], {'paths':saved, 'metadata':meta_list, 'gen_time':time.time()-gen_start})
        except InterruptedError:
            self.status_updated.emit("Cancelled"); self.progress_updated.emit(0); self.eta_updated.emit(""); self.cancelled.emit()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.error_occurred.emit("GPU OOM! Lower chunk size/steps or enable CPU offload")
            else: self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally: self.clear_cache()

    def clear_cache(self):
        self.pipe = None; self.ema_step_time = None; self.last_step_time = None; gc.collect()
        if ML_AVAILABLE and torch.cuda.is_available():
            try: torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            except: pass

    @classmethod
    def unload_pipeline(cls):
        with cls._pipe_lock: cls._unload_unsafe()
    @classmethod
    def _unload_unsafe(cls):
        if cls._shared_pipe is not None:
            try:
                p = cls._shared_pipe
                if hasattr(p, "remove_hooks"):
                    try: p.remove_hooks()
                    except: pass
                del p; cls._shared_pipe = None; cls._shared_pipe_model = None
                cls._shared_pipe_device = None; cls._shared_pipe_offload = None; gc.collect()
                if ML_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            except: pass

# ═══════════════════════════════════════════════════════════════════════
# RendererBase — shared optimizations for all visualizers
# ═══════════════════════════════════════════════════════════════════════
class RendererBase:
    _GLOW = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]], dtype=np.int32)

    def _init_buffers(self, w, h):
        self._w = max(w, 2); self._h = max(h, 2); self._motion_speed = 1.0; self._softness = 0.0
        self._backbuf = np.zeros((self._h, self._w, 3), dtype=np.float32)
        self._backbuf_flat = self._backbuf.reshape(-1)
        self._outbuf = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        self._qimg = QImage(self._outbuf.data, self._w, self._h, self._w*3, QImage.Format_RGB888)
        self._bi = None; self._bw = None; self._bs = 0

    def _resize_buffers(self, w, h):
        w, h = max(w, 2), max(h, 2)
        if w == self._w and h == self._h: return
        self._init_buffers(w, h)

    def _fade(self, f=0.87): np.multiply(self._backbuf, f, out=self._backbuf, casting='unsafe')

    def _scatter(self, xs, ys, rgb, glow=0.22):
        w, h = self._w, self._h; n = len(xs)
        if n == 0: return
        xi = np.clip(xs.astype(np.int32), 0, w-1).astype(np.int64)
        yi = np.clip(ys.astype(np.int32), 0, h-1).astype(np.int64)
        pi = yi * w + xi; nb = 1 + (8 if glow > 0 else 0); te = n*3*nb
        if self._bs < te: self._bi = np.empty(te, dtype=np.int64); self._bw = np.empty(te, dtype=np.float64); self._bs = te
        idx, wt = self._bi, self._bw; s = n*3; base = 0
        idx[base:base+s:3] = pi*3; idx[base+1:base+s:3] = pi*3+1; idx[base+2:base+s:3] = pi*3+2
        wt[base:base+s:3] = rgb[:,0].astype(np.float64); wt[base+1:base+s:3] = rgb[:,1].astype(np.float64); wt[base+2:base+s:3] = rgb[:,2].astype(np.float64)
        if glow > 0:
            gw = wt[:s] * glow
            for gi, (dx, dy) in enumerate(self._GLOW):
                b = (gi+1)*s; gx = np.clip(xi+dx, 0, w-1); gy = np.clip(yi+dy, 0, h-1); gp = gy*w+gx
                idx[b:b+s:3] = gp*3; idx[b+1:b+s:3] = gp*3+1; idx[b+2:b+s:3] = gp*3+2
                wt[b:b+s:3] = gw[0::3]; wt[b+1:b+s:3] = gw[1::3]; wt[b+2:b+s:3] = gw[2::3]
        used = s*nb; bs = w*h*3
        self._backbuf_flat += np.bincount(idx[:used], weights=wt[:used], minlength=bs)[:bs].astype(np.float32)

    def _finalize(self):
        np.clip(self._backbuf, 0, 255, out=self._backbuf); self._outbuf[:] = self._backbuf; return self._qimg
    def get_raw_rgb(self):
        np.clip(self._backbuf, 0, 255, out=self._backbuf); self._outbuf[:] = self._backbuf; return self._outbuf.tobytes()
    def set_motion_speed(self, s): self._motion_speed = max(0.01, min(2.0, s))
    def set_softness(self, s): self._softness = max(0, min(1, s))
    def _sat(self): return max(0, int(self._theme["sat"] * (1 - self._softness * 0.4)))
    def _val(self): return min(255, int(self._theme["val"] * (1 - self._softness * 0.1) + self._softness * 30))

# ═══════════════════════════════════════════════════════════════════════
# FastLaplacian
# ═══════════════════════════════════════════════════════════════════════
class FastLaplacian:
    def __init__(self, h, w):
        self._h, self._w = h, w; self._p = np.empty((h+2, w+2), dtype=np.float32)
        self._out = np.empty((h, w), dtype=np.float32)
    def resize(self, h, w):
        if h == self._h and w == self._w: return
        self._h, self._w = h, w; self._p = np.empty((h+2, w+2), dtype=np.float32); self._out = np.empty((h, w), dtype=np.float32)
    def compute(self, U):
        p = self._p; p[1:-1,1:-1] = U; p[0,1:-1]=U[-1,:]; p[-1,1:-1]=U[0,:]; p[1:-1,0]=U[:,-1]; p[1:-1,-1]=U[:,0]
        p[0,0]=U[-1,-1]; p[0,-1]=U[-1,0]; p[-1,0]=U[0,-1]; p[-1,-1]=U[0,0]
        np.multiply(U, -1.0, out=self._out)
        self._out += 0.20*(p[2:,1:-1]+p[:-2,1:-1]+p[1:-1,2:]+p[1:-1,:-2])
        self._out += 0.05*(p[2:,2:]+p[2:,:-2]+p[:-2,2:]+p[:-2,:-2])
        return self._out

# ═══════════════════════════════════════════════════════════════════════
# Visualizer Renderers (13 styles)
# ═══════════════════════════════════════════════════════════════════════
class ElectricSheepRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic", num_particles=1200):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._n = max(100, num_particles); self._time = 0.0; self._hue_shift = 0.0
        self._smooth_amp = 0.0; self._smooth_bands = np.zeros(8, dtype=np.float32)
        self._px = np.random.rand(self._n).astype(np.float32)*self._w; self._py = np.random.rand(self._n).astype(np.float32)*self._h
        self._pvx = np.zeros(self._n, dtype=np.float32); self._pvy = np.zeros(self._n, dtype=np.float32)
        self._plife = np.random.rand(self._n).astype(np.float32)*150+50; self._pmax = self._plife.copy()
        self._phue = np.random.rand(self._n).astype(np.float32)*360
        self._attractors = [{'phase': i*1.0472, 'freq': 0.25+i*0.08, 'x':0, 'y':0, 'strength':1.0} for i in range(6)]
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h; cx, cy = w/2, h/2
        self._smooth_amp = 0.7*self._smooth_amp + 0.3*amp
        if len(bands) >= 8: self._smooth_bands = 0.8*self._smooth_bands + 0.2*bands[:8]
        amp = self._smooth_amp; bands = self._smooth_bands
        for i, a in enumerate(self._attractors):
            b = bands[i%len(bands)] if len(bands)>0 else 0
            r = min(w,h)*0.15*(1+amp*2.5+b*0.8)
            a['x'] = cx + math.cos(self._time*a['freq']+a['phase'])*r
            a['y'] = cy + math.sin(self._time*a['freq']*1.3+a['phase'])*r; a['strength'] = 0.3+b*4.0
        ang = (self._px*0.008+self._py*0.006+self._time*0.4+amp*3.0)
        ang2 = (self._px*0.02+self._py*0.015+self._time*0.6)*0.5; fs = 0.4+amp*12.0
        self._pvx += (np.cos(ang)+np.cos(ang2)*0.5).astype(np.float32)*fs
        self._pvy += (np.sin(ang)+np.sin(ang2)*0.5).astype(np.float32)*fs
        for a in self._attractors:
            dx, dy = a['x']-self._px, a['y']-self._py; d2 = dx*dx+dy*dy+100; d = np.sqrt(d2)
            f = a['strength']/(d2*0.008); self._pvx += (dx/d*f).astype(np.float32); self._pvy += (dy/d*f).astype(np.float32)
        self._pvx *= 0.92; self._pvy *= 0.92; self._px += self._pvx; self._py += self._pvy; self._plife -= 1
        dead = (self._plife<=0)|(self._px<-20)|(self._px>w+20)|(self._py<-20)|(self._py>h+20)
        nd = int(np.sum(dead))
        if nd > 0:
            a = self._attractors[np.random.randint(len(self._attractors))]; sp = 25+amp*50
            self._px[dead] = a['x']+(np.random.rand(nd)-0.5)*sp; self._py[dead] = a['y']+(np.random.rand(nd)-0.5)*sp
            self._pvx[dead]=0; self._pvy[dead]=0; self._plife[dead]=np.random.rand(nd)*120+60; self._pmax[dead]=self._plife[dead]
            self._phue[dead] = (self._theme["hue_base"]+np.random.rand(nd)*self._theme["hue_range"]+self._hue_shift)%360
        self._hue_shift += 0.4+amp*8.0; self._fade(0.85)
        lr = np.clip(self._plife/self._pmax, 0, 1).astype(np.float32)
        hue = (self._phue+self._hue_shift)%360; rgb = hsv_to_rgb_array(hue, self._sat(), self._val()).astype(np.float32)
        rgb *= lr * (0.5+amp*4.0)[:, np.newaxis]; self._scatter(self._px, self._py, rgb, 0.22); return self._finalize()

class FrequencyBarsRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic"):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._nb = 64; self._peaks = np.zeros(self._nb, dtype=np.float32); self._smooth = np.zeros(self._nb, dtype=np.float32); self._time = 0.0
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        self._time += dt * self._motion_speed; w, h = self._w, self._h
        if len(bands) > 0: self._smooth = 0.6*self._smooth + 0.4*np.interp(np.linspace(0,1,self._nb), np.linspace(0,1,len(bands)), bands).astype(np.float32)
        self._peaks = np.maximum(self._peaks*0.98, self._smooth); self._fade(0.88)
        bw = w / self._nb
        for i in range(self._nb):
            bh = int(self._smooth[i] * h * 0.85); x = int(i*bw); bwd = max(1, int(bw*0.8))
            hue = (self._theme["hue_base"] + (i/self._nb)*self._theme["hue_range"] + self._time*10) % 360
            rgb = hsv_to_rgb_array(np.array([hue]), self._sat(), self._val())[0].astype(np.float32)
            if bh > 0: self._backbuf[h-bh:h, x:x+bwd] += rgb * 0.8
            ph = int(self._peaks[i] * h * 0.85)
            if ph > 0 and 0 <= h-ph < h: self._backbuf[h-ph:h-ph+2, x:x+bwd] += rgb
        self._backbuf[h//2:] *= 0.3; return self._finalize()

class RadialSpectrumRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic"):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._nb = 128; self._smooth = np.zeros(self._nb, dtype=np.float32); self._peaks = np.zeros(self._nb, dtype=np.float32)
        self._time = 0.0; self._rot = 0.0
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        self._time += dt*self._motion_speed; self._rot += dt*self._motion_speed*0.3; w, h = self._w, self._h; cx, cy = w/2, h/2
        if len(bands) > 0: self._smooth = 0.6*self._smooth + 0.4*np.interp(np.linspace(0,1,self._nb), np.linspace(0,1,len(bands)), bands).astype(np.float32)
        self._peaks = np.maximum(self._peaks*0.97, self._smooth); self._fade(0.82)
        ir = min(w,h)*0.12; mr = min(w,h)*0.45; n = self._nb
        angles = np.linspace(0, 2*math.pi, n, endpoint=False) + self._rot
        all_xs, all_ys, all_rgb = [], [], []
        for i in range(n):
            bl = self._smooth[i]*(mr-ir); steps = max(2, int(bl))
            a = angles[i]; xs = cx + np.linspace(ir, ir+bl, steps)*math.cos(a); ys = cy + np.linspace(ir, ir+bl, steps)*math.sin(a)
            hue = (self._theme["hue_base"]+(i/n)*self._theme["hue_range"]+self._time*20)%360
            rgb = np.tile(hsv_to_rgb_array(np.array([hue]), self._sat(), self._val())[0].astype(np.float32), (steps, 1))
            all_xs.append(xs); all_ys.append(ys); all_rgb.append(rgb)
        if all_xs:
            self._scatter(np.concatenate(all_xs), np.concatenate(all_ys), np.concatenate(all_rgb), 0.0)
        cr = int(ir*(1+amp*0.5))
        for a in range(0, 360, 2):
            ang = math.radians(a + self._rot*57.3); x, y = int(cx+cr*math.cos(ang)), int(cy+cr*math.sin(ang))
            if 0 <= x < w and 0 <= y < h: self._backbuf[y, x] += [255, 255, 255]
        return self._finalize()

class SpectrogramWaterfallRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic"):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"]); self._time = 0.0
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        self._time += dt*self._motion_speed; w, h = self._w, self._h
        self._backbuf[1:] = self._backbuf[:-1]; self._backbuf[0] = 0
        if len(bands) > 0:
            line = np.interp(np.linspace(0, 1, w), np.linspace(0, 1, len(bands)), bands).astype(np.float32)
            hues = (self._theme["hue_base"] + line * self._theme["hue_range"] + self._time*5) % 360
            rgb = hsv_to_rgb_array(hues, self._sat(), self._val()).astype(np.float32) * (line[:, np.newaxis] * 2.0 + 0.1)
            self._backbuf[0] = rgb
        if amp > 0.01:
            g = int(amp * 50); self._backbuf[:g, :3] += [100, 50, 200]; self._backbuf[:g, -3:] += [100, 50, 200]
        return self._finalize()

class WaveformLandscapeRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic"):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._history = deque(maxlen=80); self._time = 0.0
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands, waveform=None):
        self._time += dt*self._motion_speed; w, h = self._w, self._h
        if waveform is not None and len(waveform) > 0:
            n = len(waveform)
            if n >= w: wf = np.abs(waveform[:w*(n//w)].reshape(w, n//w)).mean(axis=1)
            else: wf = np.interp(np.linspace(0, n-1, w), np.arange(n), np.abs(waveform))
        else: wf = np.interp(np.linspace(0, 1, w), np.linspace(0, 1, max(1, len(bands))), bands) if len(bands) > 0 else np.zeros(w)
        self._history.appendleft(wf); self._fade(0.85)
        for ri, rd in enumerate(self._history):
            persp = 1.0 - ri/max(len(self._history), 1); yb = int(h*0.5 + ri*(h/max(len(self._history),1))*0.4)
            sc = persp * h * 0.3; hue = (self._theme["hue_base"]+ri*5+self._time*10)%360
            rgb = hsv_to_rgb_array(np.array([hue]), self._sat(), self._val())[0].astype(np.float32)
            for x in range(0, w, max(1, w//400)):
                y = yb - int(rd[x]*sc)
                if 0 <= y < h:
                    self._backbuf[y, x] += rgb * persp * 1.5
                    if y+1 < h: self._backbuf[y+1, x] += rgb * persp * 0.5
        return self._finalize()

class ChaosGameRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic", num_vertices=5, num_points=5000):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._nv = max(3, num_vertices); self._np = max(100, num_points)
        self._verts = np.zeros((self._nv, 2), dtype=np.float32); self._update_verts()
        self._px = np.random.rand(self._np).astype(np.float32)*self._w; self._py = np.random.rand(self._np).astype(np.float32)*self._h
        self._pv = np.random.randint(0, self._nv, self._np).astype(np.int32); self._phue = np.random.rand(self._np).astype(np.float32)*360
        self._time = 0.0; self._hs = 0.0; self._jf = 0.5; self._sa = 0.0; self._sw = np.ones(self._nv, dtype=np.float32)/self._nv; self._rot = 0.0
    def _update_verts(self):
        cx, cy = self._w/2, self._h/2; r = min(self._w, self._h)*0.35
        for i in range(self._nv):
            a = -math.pi/2 + i*2*math.pi/self._nv; self._verts[i, 0] = cx+r*math.cos(a); self._verts[i, 1] = cy+r*math.sin(a)
    def resize(self, w, h): self._resize_buffers(w, h); self._update_verts()
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h; cx, cy = w/2, h/2
        self._sa = 0.8*self._sa + 0.2*amp; amp = self._sa; self._jf = 0.42 + amp*0.18
        self._rot += dt*(0.05 + (bands[3] if len(bands)>3 else 0)*0.5)
        rb = min(w, h)*0.35
        for i in range(self._nv):
            b = bands[i%len(bands)] if len(bands)>0 else 0; r = rb*(1.0+b*0.35+amp*0.25)
            a = -math.pi/2 + i*2*math.pi/self._nv + self._rot
            self._verts[i, 0] = cx+r*math.cos(a); self._verts[i, 1] = cy+r*math.sin(a)
        wt = np.array([bands[i%len(bands)] if len(bands)>0 else 1.0 for i in range(self._nv)], dtype=np.float32)+0.05; wt /= wt.sum()
        self._sw = 0.85*self._sw + 0.15*wt; cw = np.cumsum(self._sw)
        ch = np.searchsorted(cw, np.random.rand(self._np).astype(np.float32)).astype(np.int32); ch = np.clip(ch, 0, self._nv-1)
        rep = ch == self._pv; nr = int(np.sum(rep))
        if nr > 0: ch[rep] = (ch[rep] + np.random.randint(1, self._nv, nr)) % self._nv
        self._pv = ch
        self._px += (self._verts[ch, 0] - self._px) * self._jf; self._py += (self._verts[ch, 1] - self._py) * self._jf
        if amp > 0.25 and np.random.rand() < 0.1:
            nr2 = min(int(amp*self._np*0.3), self._np//4); rm = np.random.choice(self._np, nr2, replace=False)
            vi = np.random.randint(0, self._nv, nr2)
            self._px[rm] = self._verts[vi, 0] + (np.random.rand(nr2)-0.5)*20; self._py[rm] = self._verts[vi, 1] + (np.random.rand(nr2)-0.5)*20
        self._hs += 0.25 + amp*7.0; self._fade(0.86)
        vh = self._pv.astype(np.float32) * (360.0/self._nv); hue = (self._phue+vh+self._hs+self._theme["hue_base"])%360
        rgb = hsv_to_rgb_array(hue, self._sat(), self._val()).astype(np.float32) * (0.25+amp*3.5)
        self._scatter(self._px, self._py, rgb, 0.22); return self._finalize()

class ReactiveDiffusionRenderer(RendererBase):
    PRESETS = DIFFUSION_PRESETS
    def __init__(self, w=800, h=600, theme="Cosmic", sim_scale=6):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._ss = max(2, sim_scale); self._sw = max(64, self._w//self._ss); self._sh = max(36, self._h//self._ss)
        self._lap = FastLaplacian(self._sh, self._sw)
        self._U = np.ones((self._sh, self._sw), dtype=np.float32); self._V = np.zeros((self._sh, self._sw), dtype=np.float32); self._seed()
        self._tU = np.empty_like(self._U); self._tV = np.empty_like(self._V); self._uvv = np.empty_like(self._U); self._tmp = np.empty_like(self._U)
        self._sr = np.empty((self._sh, self._sw, 3), dtype=np.float32)
        yy = np.arange(self._sh, dtype=np.float32).reshape(-1, 1); xx = np.arange(self._sw, dtype=np.float32).reshape(1, -1)
        self._spatial = ((yy/self._sh*120 + xx/self._sw*120) % 360)
        self._du=1.0; self._dv=0.5; self._feed=0.055; self._kill=0.062; self._steps=8; self._time=0.0; self._sa=0.0; self._sb=np.zeros(8,dtype=np.float32); self._pt=0.0
    def _seed(self):
        self._U.fill(1.0); self._V.fill(0.0)
        for _ in range(6):
            cx, cy, r = np.random.randint(0, self._sw), np.random.randint(0, self._sh), np.random.randint(4, 12)
            yy, xx = np.ogrid[:self._sh, :self._sw]; self._V[(yy-cy)**2+(xx-cx)**2 < r*r] = 0.5+np.random.rand()*0.5
    def resize(self, w, h):
        self._resize_buffers(w, h); self._sw = max(64, w//self._ss); self._sh = max(36, h//self._ss)
        self._lap = FastLaplacian(self._sh, self._sw); self._U = np.ones((self._sh, self._sw), dtype=np.float32)
        self._V = np.zeros((self._sh, self._sw), dtype=np.float32); self._seed()
        self._tU=np.empty_like(self._U); self._tV=np.empty_like(self._V); self._uvv=np.empty_like(self._U); self._tmp=np.empty_like(self._U)
        self._sr = np.empty((self._sh, self._sw, 3), dtype=np.float32)
        yy = np.arange(self._sh, dtype=np.float32).reshape(-1, 1); xx = np.arange(self._sw, dtype=np.float32).reshape(1, -1)
        self._spatial = ((yy/self._sh*120 + xx/self._sw*120) % 360)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def _pick(self, n): 
        if n in self.PRESETS: self._feed = self.PRESETS[n]["feed"]; self._kill = self.PRESETS[n]["kill"]
    def _inj(self, cx, cy, r, a):
        y0,y1,x0,x1 = max(0,cy-r),min(self._sh,cy+r+1),max(0,cx-r),min(self._sw,cx+r+1)
        if y0>=y1 or x0>=x1: return
        yy,xx = np.ogrid[y0:y1,x0:x1]; m = (yy-cy)**2+(xx-cx)**2 < r*r
        np.maximum(self._V[y0:y1,x0:x1], self._V[y0:y1,x0:x1]+a*m, out=self._V[y0:y1,x0:x1])
        np.clip(self._V[y0:y1,x0:x1], 0, 1, out=self._V[y0:y1,x0:x1])
    def _sim(self):
        lU, lV = self._lap.compute(self._U), self._lap.compute(self._V)
        np.multiply(self._U, self._V, out=self._tmp); np.multiply(self._tmp, self._V, out=self._uvv)
        np.multiply(lU, self._du, out=lU); self._U += lU; self._U -= self._uvv
        np.multiply(self._U, -self._feed, out=self._tmp); self._tmp += self._feed; self._U += self._tmp
        np.clip(self._U, 0, 1, out=self._U); np.multiply(lV, self._dv, out=lV)
        self._V += lV; self._V += self._uvv; np.multiply(self._V, -(self._feed+self._kill), out=self._tmp)
        self._V += self._tmp; np.clip(self._V, 0, 1, out=self._V)
    def render_frame(self, dt, amp, bands):
        self._time += dt; self._sa = 0.7*self._sa+0.3*amp; amp = self._sa
        if len(bands) >= 8: self._sb = 0.8*self._sb+0.2*bands[:8]
        bass = self._sb[0] if len(self._sb)>0 else 0; mid = self._sb[3] if len(self._sb)>3 else 0; treb = self._sb[6] if len(self._sb)>6 else 0
        self._feed = 0.9*self._feed+0.1*(0.035+bass*0.035+amp*0.015); self._kill = 0.9*self._kill+0.1*(0.060+treb*0.030+amp*0.012)
        if amp > 0.04:
            cx = max(2, min(self._sw-2, int(self._sw*(0.25+mid*0.5)))); cy = max(2, min(self._sh-2, int(self._sh*(0.25+treb*0.5))))
            self._inj(cx, cy, max(2, int(amp*10)), amp*0.7)
        if amp > 0.12:
            for _ in range(min(4, int(amp*8))): self._inj(np.random.randint(2,self._sw-2), np.random.randint(2,self._sh-2), max(2,int(amp*6)), amp*0.4)
        self._pt += dt
        if self._pt > 8.0+amp*4.0:
            self._pt = 0.0; ns = list(self.PRESETS.keys()); self._pick(ns[int(bass*len(ns))%len(ns)])
        for _ in range(self._steps): self._sim()
        pat = self._V; bg = 1.0 - self._U
        hue = (self._theme["hue_base"] + self._spatial + pat*self._theme["hue_range"]*0.5 + self._time*20) % 360
        rgb = hsv_to_rgb_array(hue.ravel(), self._sat(), self._val()).reshape(self._sh, self._sw, 3).astype(np.float32)
        rgb *= (pat*1.4 + bg*0.08).reshape(self._sh, self._sw, 1)
        sy, sx = max(1, self._h//self._sh), max(1, self._w//self._sw)
        up = np.repeat(np.repeat(rgb, sy, axis=0), sx, axis=1)
        if up.shape[0] != self._h or up.shape[1] != self._w:
            r = np.zeros((self._h, self._w, 3), dtype=np.float32)
            r[:min(up.shape[0], self._h), :min(up.shape[1], self._w)] = up[:min(up.shape[0], self._h), :min(up.shape[1], self._w)]; up = r
        self._backbuf = up; self._backbuf_flat = self._backbuf.reshape(-1)
        np.clip(self._backbuf, 0, 255, out=self._backbuf); self._outbuf[:] = self._backbuf; return self._qimg

class RainParticlesRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Ocean", motion_speed=0.3, softness=0.6):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Ocean"])
        self._motion_speed = motion_speed; self._softness = softness; self._n = 600
        self._px = np.random.rand(self._n).astype(np.float32)*self._w; self._py = np.random.rand(self._n).astype(np.float32)*self._h
        self._sp = (np.random.rand(self._n).astype(np.float32)*2+1.5); self._st = (np.random.rand(self._n).astype(np.float32)*8+5)
        self._hue = np.random.rand(self._n).astype(np.float32)*25+185; self._time = 0.0; self._sa = 0.0; self._sb = np.zeros(8, dtype=np.float32)
        self._splashes = deque(maxlen=30)
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h
        self._sa = 0.9*self._sa+0.1*amp
        if len(bands) >= 8: self._sb = 0.9*self._sb+0.1*bands[:8]
        amp = self._sa; bass = self._sb[0] if len(self._sb)>0 else 0; mid = self._sb[3] if len(self._sb)>3 else 0
        self._py += self._sp*(1.0+bass*1.5)*(dt*60)
        wind = math.sin(self._time*0.4)*(0.5+mid*2.5); self._px += wind*(dt*60)
        fallen = self._py > h+self._st; nf = int(np.sum(fallen))
        if nf > 0:
            fi = np.where(fallen)[0]
            for i in range(min(nf, 5)): self._splashes.append((float(self._px[fi[i]]), self._time))
            self._px[fallen] = np.random.rand(nf)*w; self._py[fallen] = -np.random.rand(nf)*30
            self._sp[fallen] = np.random.rand(nf)*2+1.5; self._st[fallen] = np.random.rand(nf)*8+5
        np.mod(self._px, w, out=self._px); self._fade(0.80)
        ch = (self._hue+self._time*1.5)%360; pp = 4
        ax = np.repeat(self._px, pp); ay = np.repeat(self._py, pp) - np.tile(np.arange(pp)*2.0, self._n).astype(np.float32)
        ah = np.repeat(ch, pp); ab = np.repeat(self._sp/3.5, pp) * np.tile(np.array([1.0,0.7,0.4,0.15], dtype=np.float32), self._n) * (0.3+amp*2.0)
        rgb = hsv_to_rgb_array(ah, self._sat(), self._val()).astype(np.float32) * ab[:, np.newaxis]
        self._scatter(ax, ay, rgb, 0.12)
        for sx, st in list(self._splashes):
            age = self._time - st
            if age > 0.8: continue
            rad = age*40; al = (1.0-age/0.8)*0.25; nr = max(12, int(min(60, rad/3)))
            an = np.linspace(0, 2*math.pi, nr, endpoint=False)
            rx = np.full(nr, sx)+rad*np.cos(an); ry = np.full(nr, h-2)+rad*0.2*np.sin(an)
            rr = np.tile(np.array([100,150,200], dtype=np.float32), (nr, 1))*al
            self._scatter(rx, ry, rr, 0.1)
        return self._finalize()

class ZenMandalaRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic", motion_speed=0.15, softness=0.5):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._motion_speed = motion_speed; self._softness = softness
        self._rot = 0.0; self._pulse = 0.0; self._time = 0.0; self._sa = 0.0
        self._sp = self._gen()
    def _gen(self):
        pts = []
        for ri, (r, cnt, st, sz) in enumerate([(0.10,8,"lotus",0.10),(0.20,12,"diamond",0.08),(0.32,16,"circle",0.05),(0.42,24,"dot",0.0),(0.48,32,"dot",0.0)]):
            for i in range(cnt):
                ba = i*2*math.pi/cnt
                if st == "lotus":
                    for j in range(8):
                        a = ba+(j/7-0.5)*0.5; pr = r+sz*math.sin(j/7*math.pi); pts.append((a, pr, ri))
                elif st == "diamond":
                    for j in range(5): a = ba+(j-2)*0.15; pr = r+abs(j-2)*sz*0.5; pts.append((a, pr, ri))
                elif st == "circle":
                    for j in range(6): a = ba+j*math.pi/3; pr = r+sz*math.cos(j*math.pi/3); pts.append((a, pr, ri))
                else: pts.append((ba, r, ri))
        return np.array(pts, dtype=np.float32)
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; self._rot += dt*0.12; self._pulse = 0.95*self._pulse+0.05*amp; amp = self._pulse
        cx, cy = self._w/2, self._h/2; md = min(self._w, self._h); sc = md*0.5*(1.0+amp*0.08)
        ri = self._sp[:, 2].astype(np.int32); rs = np.where(ri%2==0, 1.0, -1.0)
        ang = self._sp[:, 0] + self._rot*rs; rad = self._sp[:, 1]*sc*(1.0+amp*0.04)
        xs = cx+rad*np.cos(ang); ys = cy+rad*np.sin(ang)
        hue = (self._theme["hue_base"] + ri*25 + self._time*2.5) % 360
        rgb = hsv_to_rgb_array(hue, self._sat(), self._val()).astype(np.float32) * (0.4+amp*2.5)
        self._fade(0.89); self._scatter(xs, ys, rgb, 0.35)
        cr = md*0.04*(1.0+amp*0.6); nc = 24; ca = np.linspace(0, 2*math.pi, nc, endpoint=False)
        cxs = cx+cr*np.cos(ca); cys = cy+cr*np.sin(ca)
        ch = np.full(nc, (self._theme["hue_base"]+self._time*5)%360, dtype=np.float32)
        crgb = hsv_to_rgb_array(ch, self._sat(), self._val()).astype(np.float32)*3.0
        self._scatter(cxs, cys, crgb, 0.5); return self._finalize()

class SoftOrbsRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Cosmic", motion_speed=0.1, softness=0.8):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Cosmic"])
        self._motion_speed = motion_speed; self._softness = softness; self._n = 20
        self._ox = np.random.rand(self._n).astype(np.float32)*self._w; self._oy = np.random.rand(self._n).astype(np.float32)*self._h
        self._ovx = (np.random.rand(self._n)-0.5).astype(np.float32)*0.5; self._ovy = (np.random.rand(self._n)-0.5).astype(np.float32)*0.5
        self._or = np.random.rand(self._n).astype(np.float32)*20+30; self._ohue = np.random.rand(self._n).astype(np.float32)*360
        self._time = 0.0; self._sa = 0.0; self._pp = 60
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h
        self._sa = 0.92*self._sa+0.08*amp; amp = self._sa
        self._ox += self._ovx*(dt*60); self._oy += self._ovy*(dt*60)
        bx = (self._ox<0)|(self._ox>w); by = (self._oy<0)|(self._oy>h)
        self._ovx[bx] *= -1; self._ovy[by] *= -1; np.clip(self._ox, 0, w, out=self._ox); np.clip(self._oy, 0, h, out=self._oy)
        self._ohue = (self._ohue + dt*2.0) % 360; self._fade(0.90)
        n = self._n*self._pp; ax = np.empty(n, dtype=np.float32); ay = np.empty(n, dtype=np.float32); ar = np.empty((n, 3), dtype=np.float32)
        for i in range(self._n):
            b = i*self._pp; an = np.random.rand(self._pp)*2*math.pi; ra = np.sqrt(np.random.rand(self._pp))*self._or[i]*(1.0+amp*0.3)
            ax[b:b+self._pp] = self._ox[i]+ra*np.cos(an); ay[b:b+self._pp] = self._oy[i]+ra*np.sin(an)
            fo = np.exp(-(ra/max(self._or[i], 1))**2 * 2.5)
            rgb = hsv_to_rgb_array(np.full(self._pp, self._ohue[i], dtype=np.float32), self._sat(), self._val()).astype(np.float32)
            ar[b:b+self._pp] = rgb * fo[:, np.newaxis] * (0.5+amp*2.0)
        self._scatter(ax, ay, ar, 0.4); return self._finalize()

class BreathingWaveRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Ocean", motion_speed=0.2, softness=0.7):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Ocean"])
        self._motion_speed = motion_speed; self._softness = softness; self._time = 0.0; self._sa = 0.0; self._nwp = 300
        self._layers = [{"f":1.0,"s":0.15,"a":1.0,"o":0.0,"h":0}, {"f":1.7,"s":0.10,"a":0.5,"o":0.3,"h":30}, {"f":0.6,"s":0.20,"a":0.7,"o":-0.2,"h":-20}]
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h
        self._sa = 0.93*self._sa+0.07*amp; amp = self._sa; self._fade(0.88)
        xp = np.linspace(0, w, self._nwp, dtype=np.float32); my = h/2.0; ax, ay, ah, ab = [], [], [], []
        for L in self._layers:
            wa = h*0.12*L["a"]*(0.3+amp*2.5); ph = self._time*L["s"]+L["o"]*2*math.pi
            wy = my + wa*np.sin(xp/w*2*math.pi*L["f"]+ph)
            for j in range(3):
                jit = (j-1)*1.5; ax.append(xp); ay.append(wy+jit)
                ah.append(np.full(self._nwp, (self._theme["hue_base"]+L["h"]+self._time*3)%360, dtype=np.float32))
                ab.append(np.full(self._nwp, (1.0-abs(j-1)*0.3)*(0.4+amp*2.0), dtype=np.float32))
        ax, ay, ah, ab = np.concatenate(ax), np.concatenate(ay), np.concatenate(ah), np.concatenate(ab)
        rgb = hsv_to_rgb_array(ah, self._sat(), self._val()).astype(np.float32) * ab[:, np.newaxis]
        self._scatter(ax, ay, rgb, 0.35); return self._finalize()

class FireplaceRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Fire", motion_speed=0.5, softness=0.4):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Fire"])
        self._motion_speed = motion_speed; self._softness = softness; self._n = 500
        self._px = np.random.rand(self._n).astype(np.float32)*self._w; self._py = np.random.rand(self._n).astype(np.float32)*self._h
        self._pvx = np.zeros(self._n, dtype=np.float32); self._pvy = np.zeros(self._n, dtype=np.float32)
        self._plife = np.random.rand(self._n).astype(np.float32)*100+50; self._pmax = self._plife.copy()
        self._time = 0.0; self._sa = 0.0
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h
        self._sa = 0.88*self._sa+0.12*amp; amp = self._sa
        self._pvy -= (1.5+amp*6.0)*(dt*60)
        tb = (np.sin(self._py*0.02+self._time*0.5)+np.cos(self._px*0.01+self._time*0.3))*0.5
        self._pvx += tb.astype(np.float32)*(dt*60); self._pvx *= 0.95; self._pvy *= 0.97
        self._px += self._pvx*(dt*60); self._py += self._pvy*(dt*60); self._plife -= dt*30
        dead = (self._plife<=0)|(self._py<0)|(self._py>h)|(self._px<-20)|(self._px>w+20)
        nd = int(np.sum(dead))
        if nd > 0:
            cx = w/2; sp = w*0.15
            self._px[dead] = cx+np.random.randn(nd).astype(np.float32)*sp; self._py[dead] = h-np.random.rand(nd)*10
            self._pvx[dead] = (np.random.rand(nd)-0.5); self._pvy[dead] = -(np.random.rand(nd)*1.5+0.5)
            self._plife[dead] = np.random.rand(nd)*80+40; self._pmax[dead] = self._plife[dead]
        self._fade(0.84); hr = np.clip(1.0-self._py/h, 0, 1)
        hue = (self._theme["hue_base"]+hr*50+self._time*1.5)%360
        lr = np.clip(self._plife/self._pmax, 0, 1)
        rgb = hsv_to_rgb_array(hue, self._sat(), self._val()).astype(np.float32)
        rgb *= lr*(0.3+amp*3.0)*(0.5+hr*0.5)[:, np.newaxis]
        self._scatter(self._px, self._py, rgb, 0.3)
        ne = 80; ex = np.random.rand(ne).astype(np.float32)*w; ey = h-np.random.rand(ne)*5
        eh = np.random.rand(ne).astype(np.float32)*30; er = hsv_to_rgb_array(eh, self._sat(), self._val()).astype(np.float32)*(0.5+amp*2.0)
        self._scatter(ex, ey, er, 0.4); return self._finalize()

class WaterRipplesRenderer(RendererBase):
    def __init__(self, w=800, h=600, theme="Ocean", motion_speed=0.25, softness=0.6):
        self._init_buffers(max(w,2), max(h,2)); self._theme = VIDEO_COLOR_THEMES.get(theme, VIDEO_COLOR_THEMES["Ocean"])
        self._motion_speed = motion_speed; self._softness = softness; self._time = 0.0; self._sa = 0.0; self._sb = np.zeros(8, dtype=np.float32)
        self._ripples = []; self._lrt = 0.0
        self._ap = np.random.rand(200).astype(np.float32)*self._w; self._ay = np.random.rand(200).astype(np.float32)*self._h
    def resize(self, w, h): self._resize_buffers(w, h)
    def set_theme(self, t): self._theme = VIDEO_COLOR_THEMES.get(t, self._theme)
    def render_frame(self, dt, amp, bands):
        dt *= self._motion_speed; self._time += dt; w, h = self._w, self._h
        self._sa = 0.92*self._sa+0.08*amp
        if len(bands) >= 8: self._sb = 0.9*self._sb+0.1*bands[:8]
        amp = self._sa; mid = self._sb[3] if len(self._sb)>3 else 0; treb = self._sb[6] if len(self._sb)>6 else 0
        if amp > 0.08 and self._time-self._lrt > 0.3:
            self._lrt = self._time; cx, cy = w/2, h/2
            self._ripples.append((cx+(mid-0.5)*w*0.3, cy+(treb-0.5)*h*0.3, self._time, amp))
        if int(self._time*2) > int((self._time-dt)*2): self._ripples.append((w/2, h/2, self._time, 0.3))
        self._ripples = [(rx, ry, st, s) for rx, ry, st, s in self._ripples if self._time-st < 3.0]
        self._fade(0.85); ax, ay, ah, ab = [], [], [], []
        for rx, ry, st, s in self._ripples:
            age = self._time-st; rad = age*80; al = (1.0-age/3.0)*(0.2+s*0.5)
            if al <= 0 or rad < 2: continue
            nr = max(12, int(min(60, rad/3))); an = np.linspace(0, 2*math.pi, nr, endpoint=False)
            rx_arr = np.full(nr, rx)+rad*np.cos(an); ry_arr = np.full(nr, ry)+rad*np.sin(an)
            v = (rx_arr >= 0) & (rx_arr < w) & (ry_arr >= 0) & (ry_arr < h)
            if not np.any(v): continue
            hue = (self._theme["hue_base"]+rad*0.5+self._time*3)%360
            ax.append(rx_arr[v]); ay.append(ry_arr[v]); ah.append(np.full(int(np.sum(v)), hue, dtype=np.float32)); ab.append(np.full(int(np.sum(v)), al, dtype=np.float32))
        self._ay = np.mod(self._ay + 0.2*(dt*60), h)
        ax.append(self._ap); ay.append(self._ay); ah.append(np.full(200, self._theme["hue_base"]+10, dtype=np.float32)); ab.append(np.full(200, 0.05+amp*0.15, dtype=np.float32))
        if ax:
            ax, ay, ah, ab = np.concatenate(ax), np.concatenate(ay), np.concatenate(ah), np.concatenate(ab)
            rgb = hsv_to_rgb_array(ah, self._sat(), self._val()).astype(np.float32) * ab[:, np.newaxis]
            self._scatter(ax, ay, rgb, 0.3)
        if amp > 0.02:
            cx, cy = w/2, h/2; nc = 12; cr = 3+amp*15; an = np.linspace(0, 2*math.pi, nc, endpoint=False)
            ch = np.full(nc, self._theme["hue_base"], dtype=np.float32)
            crgb = hsv_to_rgb_array(ch, self._sat(), self._val()).astype(np.float32)*(0.3+amp*2.0)
            self._scatter(cx+cr*np.cos(an), cy+cr*np.sin(an), crgb, 0.5)
        return self._finalize()

# ═══════════════════════════════════════════════════════════════════════
# Spectrogram Widget
# ═══════════════════════════════════════════════════════════════════════
class SpectrogramWidget(QWidget):
    seekRequested = Signal(float)
    def __init__(self, parent=None, audio_cache=None):
        super().__init__(parent); self._ac = audio_cache; self._raw = None; self._sr = 16000
        self._qpix = None; self._pp = 0.0; self._dur = 0.0; self._minH = 150
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.setMouseTracking(True)
        self._pending = None; self._rt = QTimer(self); self._rt.setSingleShot(True); self._rt.setInterval(60); self._rt.timeout.connect(self._flush)
    def set_audio(self, path): self._pending = path; self._rt.start()
    def _flush(self):
        path = self._pending
        if not path: return
        self._pending = None; c = self._ac.get(path) if self._ac else None
        if c: sr, d = c
        else:
            l = AudioDataCache.load_from_disk(path)
            if not l: self._raw=None; self._qpix=None; self.update(); return
            sr, d = l; self._ac.put(path, sr, d) if self._ac else None
        self._sr = sr; self._raw = d; self._dur = len(d)/sr if sr > 0 else 0; self._compute(); self.update()
    def _compute(self):
        if self._raw is None or self._raw.size == 0: self._qpix = None; return
        ns = 2048 if self._sr >= 44100 else 1024; ns = min(ns, len(self._raw))
        _, _, Sxx = scipy.signal.spectrogram(self._raw, fs=self._sr, nperseg=ns, noverlap=ns//2, window='hann')
        Sl = 10*np.log10(Sxx+1e-10); vm, vx = float(np.percentile(Sl, 10)), float(np.percentile(Sl, 99))
        if vx <= vm: vx = vm+1; Sn = np.clip((Sl-vm)/(vx-vm), 0, 1)
        del Sl, Sxx; xp = [0.0, 0.25, 0.5, 0.75, 1.0]
        r = np.interp(Sn, xp, [68,59,33,53,253]); g = np.interp(Sn, xp, [1,82,145,148,231]); b = np.interp(Sn, xp, [84,139,140,27,36])
        del Sn; rgb = np.ascontiguousarray(np.stack((r,g,b), axis=-1).astype(np.uint8)[::-1]); del r, g, b
        h, w, _ = rgb.shape; self._qpix = QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888).copy()); del rgb; self._raw = None
    def set_playback_position(self, p): self._pp = max(0, min(1, p)); self.update()
    def clear(self): self._raw=None; self._qpix=None; self._dur=0; self.update()
    def mousePressEvent(self, e):
        if self._qpix and e.button()==Qt.LeftButton: p = e.position().x()/max(self.width(),1); self.seekRequested.emit(p); self.set_playback_position(p)
    def mouseMoveEvent(self, e):
        if self._qpix and self._dur > 0 and self._sr > 0:
            x, y = e.position().x(), e.position().y(); ts = (x/max(self.width(),1))*self._dur; fq = (self._sr/2)*(1-y/max(self.height(),1))
            QToolTip.showText(e.globalPosition().toPoint(), f"Time: {ts:.2f}s | Freq: {fq/1000:.2f}kHz" if fq>=1000 else f"Time: {ts:.2f}s | Freq: {fq:.0f}Hz", self)
            if e.buttons() & Qt.LeftButton: p = x/max(self.width(),1); self.seekRequested.emit(p); self.set_playback_position(p)
    def leaveEvent(self, e): QToolTip.hideText()
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(10,10,12))
        if not self._qpix: p.setPen(QColor(100,100,120)); p.drawText(self.rect(), Qt.AlignCenter, "No spectrogram"); return
        p.drawPixmap(self.rect(), self._qpix); p.setPen(QPen(QColor(255,255,255,100), 1, Qt.DashLine)); p.setFont(QFont("Segoe UI", 8))
        for i in range(1, 4):
            y = int(i/4*self.height()); p.drawLine(0, y, self.width(), y); fq = int((self._sr/2)*(1-i/4))
            p.drawText(5, y-2, f"{fq/1000:.1f}kHz" if fq>=1000 else f"{fq}Hz")
        if self._dur > 0:
            ss = 15.0 if self._dur>60 else (5.0 if self._dur>30 else (2.0 if self._dur>10 else 1.0)); t = ss
            while t < self._dur: x = int(t/self._dur*self.width()); p.drawLine(x, 0, x, self.height()); p.drawText(x+3, self.height()-5, format_time(t)); t += ss
        if self._pp >= 0 and self._qpix: px = int(self._pp*(self.width()-1)); p.setPen(QPen(QColor(255,255,255,220), 2)); p.drawLine(px, 0, px, self.height())

# ═══════════════════════════════════════════════════════════════════════
# Waveform Widget
# ═══════════════════════════════════════════════════════════════════════
class WaveformWidget(QWidget):
    seekRequested = Signal(float)
    def __init__(self, parent=None, audio_cache=None):
        super().__init__(parent); self._ac = audio_cache; self._raw = None; self._peaks = None; self._rms = None; self._pw = 0
        self._ppk = None; self._prm = None; self._sr = 16000; self._dur = 0.0; self._pp = 0.0; self._hp = -1.0
        self.setMinimumHeight(120); self.setMaximumHeight(180); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
    def set_audio(self, path):
        try:
            with open(path, 'rb') as f: f.read(4)
        except: self._raw=None; self._peaks=None; self._ppk=None; self._prm=None; self.update(); return
        c = self._ac.get(path) if self._ac else None
        if c: sr, d = c
        else:
            try: sr, raw = wavfile.read(path)
            except: self._raw=None; self._peaks=None; self._ppk=None; self._prm=None; self.update(); return
            d = normalize_audio_data(raw)
            if d.size == 0: self._raw=None; self._peaks=None; self._ppk=None; self._prm=None; self.update(); return
            d = np.ascontiguousarray(d, dtype=np.float32); self._ac.put(path, sr, d) if self._ac else None
        self._sr = sr; self._raw = d; self._dur = len(d)/sr if sr > 0 else 0
        self._peaks=None; self._ppk=None; self._prm=None; self._ensure(self.width()); self.update()
    def resizeEvent(self, e): super().resizeEvent(e); self._peaks=None; self._ppk=None; self._prm=None; self._ensure(self.width()); self.update()
    def _ensure(self, w):
        if self._peaks is not None and self._pw == w: return
        if self._raw is None or w <= 0: self._peaks=None; self._ppk=None; self._prm=None; return
        n = len(self._raw)
        if n == 0: self._peaks = np.zeros(w, dtype=np.float32); self._rms = np.zeros(w, dtype=np.float32); self._pw = w
        else:
            aa = np.abs(self._raw)
            if n <= w: self._peaks = np.pad(aa, (0, w-n), 'constant').astype(np.float32); self._rms = self._peaks*0.7
            else:
                cs = max(1, n//w); nf = (n//cs)*cs; t = aa[:nf].reshape(-1, cs)
                self._peaks = t.max(axis=1).astype(np.float32); self._rms = np.sqrt(np.mean(t**2, axis=1)).astype(np.float32)
                if len(self._peaks) < w: self._peaks=np.pad(self._peaks, (0, w-len(self._peaks)), 'constant'); self._rms=np.pad(self._rms, (0, w-len(self._rms)), 'constant')
                elif len(self._peaks) > w: self._peaks=self._peaks[:w]; self._rms=self._rms[:w]
            del aa; self._pw = w
        self._ppk = self._bp(self._peaks, w); self._prm = self._bp(self._rms, w); self._raw = None; self.update()
    def _bp(self, h, w):
        mid = self.height()/2; mh = self.height()*0.42; p = QPainterPath()
        if w == 0: return p; p.moveTo(0, mid)
        for x, hh in enumerate(h): p.lineTo(x, mid - hh*mh)
        p.lineTo(w, mid)
        for x in range(w-1, -1, -1): p.lineTo(x, mid + h[x]*mh)
        p.closeSubpath(); return p
    def set_playback_position(self, p): self._pp = max(0, min(1, p)); self.update()
    def clear(self): self._raw=None; self._peaks=None; self._ppk=None; self._prm=None; self._dur=0; self.update()
    def rms_at(self, p):
        if self._rms is None or len(self._rms) == 0: return 0.0
        return float(self._rms[int(max(0, min(1, p))*(len(self._rms)-1))])
    def mousePressEvent(self, e):
        if self._ppk and e.button()==Qt.LeftButton: p=e.position().x()/max(self.width(),1); self.seekRequested.emit(p); self.set_playback_position(p)
    def mouseMoveEvent(self, e):
        if self._ppk: self._hp=e.position().x()/max(self.width(),1); (self.seekRequested.emit(self._hp), self.set_playback_position(self._hp)) if e.buttons()&Qt.LeftButton else None; self.update()
    def leaveEvent(self, e): self._hp=-1; self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(13, 27, 42))
        if not self._ppk: p.setPen(QColor(100,100,120)); p.drawText(self.rect(), Qt.AlignCenter, "No audio"); return
        mid = h/2; px = int(self._pp*(w-1))
        if self._dur > 0:
            ss = 15.0 if self._dur>60 else (5.0 if self._dur>30 else (2.0 if self._dur>10 else 1.0))
            p.setPen(QPen(QColor(30,40,55), 1, Qt.DashLine)); p.setFont(QFont("Segoe UI", 8)); t = 0.0
            while t < self._dur: gx = int(t/self._dur*w); p.drawLine(gx, 10, gx, h-10); p.drawText(gx+3, 12, 40, 15, Qt.AlignLeft, format_time(t)); t += ss
        p.setClipRect(px, 0, w-px, h); p.setBrush(QColor(60,65,75)); p.setPen(Qt.NoPen); p.drawPath(self._ppk)
        p.setBrush(QColor(80,90,100)); p.drawPath(self._prm)
        p.setClipRect(0, 0, px+1, h); g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0, QColor(233,69,96)); g.setColorAt(0.5, QColor(255,100,130)); g.setColorAt(1, QColor(233,69,96))
        p.setBrush(g); p.drawPath(self._ppk); p.setBrush(QColor(255,150,180,220)); p.drawPath(self._prm); p.setClipping(False)
        p.setPen(QPen(QColor(35,35,40), 1)); p.drawLine(0, mid, w, mid)
        if 0 <= self._hp <= 1: hx = int(self._hp*(w-1)); p.setPen(QPen(QColor(255,255,255,80), 1, Qt.DashLine)); p.drawLine(hx, 0, hx, h)
        if self._pp >= 0 and self._ppk: p.setPen(QPen(QColor(255,255,255,220), 2)); p.drawLine(px, 0, px, h); p.setBrush(QColor(255,255,255,220)); p.drawPolygon(QPolygon([QPoint(px-5,0), QPoint(px+5,0), QPoint(px,6)]))

# ═══════════════════════════════════════════════════════════════════════
# Visualizer Widget
# ═══════════════════════════════════════════════════════════════════════
class VisualizerWidget(QWidget):
    seekRequested = Signal(float)
    def __init__(self, parent=None, audio_cache=None, style="Electric Sheep", theme="Cosmic"):
        super().__init__(parent); self._ac = audio_cache; self._raw = None; self._sr = 44100; self._path = None
        self._pp = 0.0; self._dur = 0.0; self._style = style; self._theme = theme
        self._motion_speed = 1.0; self._softness = 0.0; self._renderer = None; self._lt = 0.0
        self.setMinimumSize(400, 300); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.setMouseTracking(True)
        self._init_renderer(); self._timer = QTimer(self); self._timer.setInterval(33); self._timer.timeout.connect(self._tick); self._timer.start()
    def _init_renderer(self):
        w, h = max(self.width(), 400), max(self.height(), 300)
        R = {"Electric Sheep": lambda: ElectricSheepRenderer(w,h,self._theme),
             "Frequency Bars": lambda: FrequencyBarsRenderer(w,h,self._theme),
             "Radial Spectrum": lambda: RadialSpectrumRenderer(w,h,self._theme),
             "Spectrogram Waterfall": lambda: SpectrogramWaterfallRenderer(w,h,self._theme),
             "Waveform Landscape": lambda: WaveformLandscapeRenderer(w,h,self._theme),
             "Chaos Game": lambda: ChaosGameRenderer(w,h,self._theme),
             "Reactive Diffusion": lambda: ReactiveDiffusionRenderer(w,h,self._theme),
             "Rain Particles": lambda: RainParticlesRenderer(w,h,self._theme,self._motion_speed,self._softness),
             "Zen Mandala": lambda: ZenMandalaRenderer(w,h,self._theme,self._motion_speed,self._softness),
             "Soft Orbs": lambda: SoftOrbsRenderer(w,h,self._theme,self._motion_speed,self._softness),
             "Breathing Wave": lambda: BreathingWaveRenderer(w,h,self._theme,self._motion_speed,self._softness),
             "Fireplace": lambda: FireplaceRenderer(w,h,self._theme,self._motion_speed,self._softness),
             "Water Ripples": lambda: WaterRipplesRenderer(w,h,self._theme,self._motion_speed,self._softness)}
        self._renderer = R.get(self._style, R["Electric Sheep"])()
        if hasattr(self._renderer, 'set_motion_speed'): self._renderer.set_motion_speed(self._motion_speed)
        if hasattr(self._renderer, 'set_softness'): self._renderer.set_softness(self._softness)
    def set_style(self, s):
        if s == self._style: return
        self._style = s; self._init_renderer(); self.update()
    def set_theme(self, t):
        self._theme = t
        if self._renderer: self._renderer.set_theme(t)
        self.update()
    def set_motion_speed(self, s):
        self._motion_speed = s
        if self._renderer and hasattr(self._renderer, 'set_motion_speed'): self._renderer.set_motion_speed(s)
    def set_softness(self, s):
        self._softness = s
        if self._renderer and hasattr(self._renderer, 'set_softness'): self._renderer.set_softness(s)
    def set_audio(self, path):
        c = self._ac.get(path) if self._ac else None
        if c: self._sr, self._raw = c
        else:
            l = AudioDataCache.load_from_disk(path)
            if not l: return
            self._sr, self._raw = l; self._ac.put(path, self._sr, self._raw) if self._ac else None
        self._path = path; self._dur = len(self._raw)/self._sr if self._raw is not None else 0
    def set_playback_position(self, p): self._pp = max(0, min(1, p))
    def clear(self): self._raw=None; self._path=None; self._dur=0
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._renderer: self._renderer.resize(max(self.width(), 2), max(self.height(), 2))
    def _tick(self):
        now = time.time(); dt = min(now-self._lt, 0.1) if self._lt > 0 else 0.033; self._lt = now
        if self._renderer is None: return
        dt_adj = dt * self._motion_speed
        if self._raw is not None and self._raw.size > 0:
            t = self._pp * self._dur; amp, bands, wf = get_audio_features(self._raw, self._sr, t)
        else:
            amp = 0.1 + 0.05 * math.sin(now * 0.7)
            bands = np.array([0.3 + 0.2 * math.sin(now * 0.5 + i) for i in range(8)], dtype=np.float32); wf = None
        if isinstance(self._renderer, WaveformLandscapeRenderer): self._renderer.render_frame(dt_adj, amp, bands, wf)
        else: self._renderer.render_frame(dt_adj, amp, bands)
        self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor(5, 5, 15))
        if self._renderer: p.drawImage(self.rect(), self._renderer._qimg)
        else: p.setPen(QColor(100,100,120)); p.drawText(self.rect(), Qt.AlignCenter, "No visualizer")
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: p = e.position().x()/max(self.width(),1); self.seekRequested.emit(p); self.set_playback_position(p)
    def mouseMoveEvent(self, e):
        if self._raw is not None and self._dur > 0 and e.buttons() & Qt.LeftButton:
            p = e.position().x()/max(self.width(),1); self.seekRequested.emit(p); self.set_playback_position(p)

# ═══════════════════════════════════════════════════════════════════════
# Mini Waveform + Thumbnail Worker
# ═══════════════════════════════════════════════════════════════════════
class MiniWaveformItem(QTableWidgetItem):
    @staticmethod
    def generate_pixmap(path):
        if not path or not os.path.exists(path): return None
        p = None
        try:
            _, d = wavfile.read(path); d = normalize_audio_data(d)
            if d.size == 0: return None
            c = max(1, len(d)//THUMB_BARS); nf = (len(d)//c)*c
            pk = np.abs(d[:nf]).reshape(-1, c).max(axis=1)
            if len(pk) < THUMB_BARS: pk = np.pad(pk, (0, THUMB_BARS-len(pk)), 'constant')
            pk = pk[:THUMB_BARS].astype(np.float32); pix = QPixmap(THUMB_WIDTH, THUMB_HEIGHT); pix.fill(Qt.transparent)
            p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(233, 69, 96), 2); pen.setCapStyle(Qt.RoundCap); p.setPen(pen); mid = THUMB_HEIGHT/2
            for i, v in enumerate(pk): h = min(v*12, 12); x = i*2; p.drawLine(x, mid-h, x, mid+h)
            return pix
        except: return None
        finally:
            if p: p.end()
    def __init__(self, path, duration):
        super().__init__(); self.setText(f"{duration:.1f}s"); self.setTextAlignment(Qt.AlignCenter)
        self.setToolTip(path); self._path = path
    def set_pixmap(self, pix): self.setIcon(QIcon(pix)); self.setSizeHint(QSize(THUMB_WIDTH+10, THUMB_HEIGHT+4))

class ThumbnailWorker(QThread):
    thumbReady = Signal(int, QPixmap)
    def __init__(self, table, parent=None):
        super().__init__(parent); self._table = table; self._q = deque(maxlen=THUMB_QUEUE_MAXLEN)
        self._lock = threading.Lock(); self._running = True
    def enqueue(self, row, path):
        with self._lock:
            for r, _ in self._q:
                if r == row: return
            self._q.append((row, path))
    def stop(self): self._running = False; self.wait(2000)
    def run(self):
        while self._running:
            item = None
            with self._lock:
                if self._q: item = self._q.popleft()
            if item is None: self.msleep(40); continue
            row, path = item; pix = MiniWaveformItem.generate_pixmap(path)
            if pix and self._running: self.thumbReady.emit(row, pix)

# ═══════════════════════════════════════════════════════════════════════
# Video Export Worker (segment-based, with resume, thumbnails, YouTube encoding)
# ═══════════════════════════════════════════════════════════════════════
class VideoExportWorker(QThread):
    progress_updated = Signal(int); status_updated = Signal(str); eta_updated = Signal(str)
    export_completed = Signal(str); error_occurred = Signal(str); thumbnail_saved = Signal(str)
    SEGMENT_FRAMES = 5000
    def __init__(self, audio_path, output_path, width, height, fps, style, theme,
                 ffmpeg_preset="slow", crf=18, audio_cache=None, parent=None,
                 motion_speed=1.0, softness=0.0, audio_bitrate="320k", extra_args=None,
                 watermark=False, watermark_pos="Bottom Right"):
        super().__init__(parent)
        self._ap = audio_path; self._op = output_path; self._w = width; self._h = height; self._fps = fps
        self._style = style; self._theme = theme; self._preset = ffmpeg_preset; self._crf = crf
        self._ac = audio_cache; self._ms = motion_speed; self._soft = softness
        self._ab = audio_bitrate; self._ea = extra_args or []; self._wm = watermark; self._wp = watermark_pos
        self._cancel = False; self._proc = None; self._best_score = 0; self._best_frame = None
    def cancel(self):
        self._cancel = True
        if self._proc:
            try: self._proc.stdin.close(); self._proc.kill()
            except: pass
    def _eval_thumb(self, raw, fi, tf):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self._h, self._w, 3)
        cy, cx = self._h//4, self._w//4; ch, cw = self._h//2, self._w//2
        c = arr[cy:cy+ch, cx:cx+cw]; sc = float(np.mean(c))*0.3 + float(np.std(c.astype(np.float32)))*0.7
        if sc > self._best_score: self._best_score = sc; self._best_frame = arr.copy()
    def _save_thumbs(self, op):
        if self._best_frame is not None:
            tp = op.replace('.mp4', '_thumbnail.png')
            QImage(self._best_frame.data, self._w, self._h, self._w*3, QImage.Format_RGB888).save(tp, "PNG", 95)
            return tp
        return None
    def _create_renderer(self):
        w, h = self._w, self._h; t = self._theme; ms = self._ms; so = self._soft
        R = {"Electric Sheep": lambda: ElectricSheepRenderer(w,h,t),
             "Frequency Bars": lambda: FrequencyBarsRenderer(w,h,t),
             "Radial Spectrum": lambda: RadialSpectrumRenderer(w,h,t),
             "Spectrogram Waterfall": lambda: SpectrogramWaterfallRenderer(w,h,t),
             "Waveform Landscape": lambda: WaveformLandscapeRenderer(w,h,t),
             "Chaos Game": lambda: ChaosGameRenderer(w,h,t),
             "Reactive Diffusion": lambda: ReactiveDiffusionRenderer(w,h,t),
             "Rain Particles": lambda: RainParticlesRenderer(w,h,t,ms,so),
             "Zen Mandala": lambda: ZenMandalaRenderer(w,h,t,ms,so),
             "Soft Orbs": lambda: SoftOrbsRenderer(w,h,t,ms,so),
             "Breathing Wave": lambda: BreathingWaveRenderer(w,h,t,ms,so),
             "Fireplace": lambda: FireplaceRenderer(w,h,t,ms,so),
             "Water Ripples": lambda: WaterRipplesRenderer(w,h,t,ms,so)}
        return R.get(self._style, lambda: ElectricSheepRenderer(w,h,t))()
    def run(self):
        try:
            if not shutil.which("ffmpeg") and not os.path.exists("ffmpeg.exe"):
                self.error_occurred.emit("FFmpeg not found."); return
            c = self._ac.get(self._ap) if self._ac else None
            if c: sr, audio = c
            else:
                l = AudioDataCache.load_from_disk(self._ap)
                if not l: self.error_occurred.emit("Failed to load audio."); return
                sr, audio = l; self._ac.put(self._ap, sr, audio) if self._ac else None
            dur = len(audio)/sr; tf = int(dur * self._fps)
            if tf <= 0: self.error_occurred.emit("Audio too short."); return
            sf = self.SEGMENT_FRAMES; ts = math.ceil(tf / sf)
            pf = self._op + '.progress.json'; done = []
            if os.path.exists(pf):
                try:
                    with open(pf) as f: done = json.load(f).get('completed', [])
                except: done = []
            if not done:
                self.status_updated.emit("Pre-computing audio features..."); self.progress_updated.emit(0)
                nb = 8; ws = min(int(sr*0.05), len(audio)); hn = np.hanning(ws).astype(np.float32)
                from scipy.fft import rfft; amps = np.zeros(tf, dtype=np.float32); bands_all = np.zeros((tf, nb), dtype=np.float32)
                fdt = 1.0/self._fps
                for i in range(tf):
                    t = i*fdt; si = int(t*sr)
                    if si+ws > len(audio):
                        win = audio[si:] if si < len(audio) else np.zeros(ws, dtype=np.float32)
                        if len(win) < ws: win = np.pad(win, (0, ws-len(win)))
                    else: win = audio[si:si+ws]
                    amps[i] = float(np.sqrt(np.mean(win**2)))
                    if ws >= 64:
                        fft = rfft(win * hn[:len(win)]); mag = np.abs(fft[:len(fft)//2])
                        bs = max(1, len(mag)//nb)
                        for j in range(nb): bands_all[i, j] = np.mean(mag[j*bs:(j+1)*bs])
                        mx = bands_all[i].max()
                        if mx > 0: bands_all[i] /= mx
                    if i % 1000 == 0: self.progress_updated.emit(int(i/tf*5)); self.status_updated.emit(f"Pre-computing: {i}/{tf}")
                np.savez_compressed(pf+'.features', amp=amps, bands=bands_all)
            else:
                d = np.load(pf+'.features.npz'); amps = d['amp']; bands_all = d['bands']
            renderer = self._create_renderer()
            if not renderer: self.error_occurred.emit(f"Unknown style: {self._style}"); return
            if hasattr(renderer, 'set_motion_speed'): renderer.set_motion_speed(self._ms)
            if hasattr(renderer, 'set_softness'): renderer.set_softness(self._soft)
            sd = self._op + '.segments'; os.makedirs(sd, exist_ok=True)
            fdt = 1.0/self._fps; st = time.time()
            for si in range(ts):
                if self._cancel: self.error_occurred.emit("Cancelled."); return
                if si in done: continue
                ss, se = si*sf, min(si*sf+sf, tf)
                sp = os.path.join(sd, f'seg_{si:04d}.mp4')
                self.status_updated.emit(f"Segment {si+1}/{ts} (frames {ss:,}-{se:,})")
                ea = list(self._ea)
                if self._wm:
                    pos = {"Bottom Right": f"w-tw-20:h-th-20", "Bottom Left": "20:h-th-20",
                           "Top Right": f"w-tw-20:20", "Top Left": "20:20"}.get(self._wp, "w-tw-20:h-th-20")
                    ea.extend(["-vf", f"drawtext=text='AI Generated':fontsize={max(12,self._h//60)}:fontcolor=white@0.5:x={pos}:"])
                cmd = ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{self._w}x{self._h}','-r',str(self._fps),'-i','-',
                       '-ss',str(ss/self._fps),'-i',self._ap,'-t',str((se-ss)/self._fps),
                       '-c:v','libx264','-preset',self._preset,'-crf',str(self._crf),'-pix_fmt','yuv420p',
                       '-c:a','aac','-b:a',self._ab] + ea + [sp]
                cf = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, creationflags=cf)
                for i in range(ss, se):
                    if self._cancel:
                        try: self._proc.kill()
                        except: pass
                        self.error_occurred.emit("Cancelled."); return
                    amp = amps[i]; bands = bands_all[i]; dt_a = fdt * self._ms
                    if isinstance(renderer, WaveformLandscapeRenderer):
                        si2 = int(i*fdt*sr); ws2 = min(int(sr*0.05), len(audio)-si2)
                        win = audio[si2:si2+ws2] if ws2 > 0 else np.zeros(1024, dtype=np.float32)
                        if len(win) < 1024: win = np.pad(win, (0, 1024-len(win)))
                        renderer.render_frame(dt_a, amp, bands, win)
                    else: renderer.render_frame(dt_a, amp, bands)
                    try: self._proc.stdin.write(renderer.get_raw_rgb())
                    except BrokenPipeError:
                        err = self._proc.stderr.read().decode('utf-8', errors='replace')
                        self.error_occurred.emit(f"Pipe broke:\n{err[-500:]}"); return
                    if i % 100 == 0: self._eval_thumb(renderer.get_raw_rgb(), i, tf)
                    if i % 30 == 0 or i == se-1:
                        op = 5 + int((i+1)/tf*90); self.progress_updated.emit(op)
                        el = time.time()-st; fd = i-ss+1+sum(se-ss for s in done)
                        if el > 0 and fd > 0:
                            rf = fd/el; rem = (tf-i-1)/max(rf, 0.1)
                            self.eta_updated.emit(f"{int(rem//3600)}h {int((rem%3600)//60)}m {int(rem%60)}s")
                        self.status_updated.emit(f"Frame {i+1:,}/{tf:,} ({op}%) | Render: {rf:.1f} fps")
                    if i % 1000 == 0 and i > 0: gc.collect()
                try: self._proc.stdin.close()
                except: pass
                self._proc.wait(timeout=300)
                if self._proc.returncode != 0:
                    err = self._proc.stderr.read().decode('utf-8', errors='replace') if self._proc.stderr else ""
                    self.error_occurred.emit(f"Segment {si} failed:\n{err[-500:]}"); return
                done.append(si)
                try:
                    with open(pf, 'w') as f: json.dump({'completed': done, 'total_segments': ts, 'total_frames': tf}, f)
                except: pass
                gc.collect()
            self._proc = None; self.status_updated.emit("Concatenating..."); self.progress_updated.emit(96)
            cf_file = os.path.join(sd, 'concat.txt')
            with open(cf_file, 'w') as f:
                for i in range(ts): f.write(f"file '{os.path.join(sd, f'seg_{i:04d}.mp4')}'\n")
            cc = ['ffmpeg','-y','-f','concat','-safe','0','-i',cf_file,'-c','copy','-movflags','+faststart',self._op]
            cf = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            r = subprocess.run(cc, capture_output=True, creationflags=cf)
            if r.returncode != 0:
                cc = ['ffmpeg','-y','-f','concat','-safe','0','-i',cf_file,'-c:v','libx264','-crf',str(self._crf),'-preset','fast','-c:a','aac','-b:a',self._ab,'-movflags','+faststart',self._op]
                r = subprocess.run(cc, capture_output=True, creationflags=cf)
                if r.returncode != 0: self.error_occurred.emit(f"Concat failed:\n{r.stderr.decode('utf-8',errors='replace')[-500:]}"); return
            self.progress_updated.emit(98); self.status_updated.emit("Cleaning up...")
            try: shutil.rmtree(sd, ignore_errors=True); os.remove(pf) if os.path.exists(pf) else None; os.remove(pf+'.features.npz') if os.path.exists(pf+'.features.npz') else None
            except: pass
            tp = self._save_thumbs(self._op)
            if tp: self.thumbnail_saved.emit(tp)
            self.progress_updated.emit(100); self.eta_updated.emit(""); self.export_completed.emit(self._op)
        except Exception as e:
            logger.error(f"Video export failed: {e}", exc_info=True); self.error_occurred.emit(str(e))
        finally: self._proc = None; gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# Sound Layering Dialog
# ═══════════════════════════════════════════════════════════════════════
class SoundLayeringDialog(QDialog):
    mixed_audio_ready = Signal(str)
    def __init__(self, parent=None, audio_cache=None):
        super().__init__(parent); self.setWindowTitle("ASMR Sound Mixer"); self.setMinimumWidth(650)
        self._ac = audio_cache; self._layers = []
        layout = QVBoxLayout(self)
        h = QLabel("Layer multiple ASMR sounds to create rich soundscapes."); h.setWordWrap(True)
        h.setStyleSheet("color: #8888bb; font-size: 11px;"); layout.addWidget(h)
        self.lt = QTableWidget(); self.lt.setColumnCount(5)
        self.lt.setHorizontalHeaderLabels(["Layer","Source","Volume","Mute",""])
        self.lt.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.lt)
        br = QHBoxLayout()
        ab = QPushButton("+ Add Audio Layer"); ab.clicked.connect(self._add); br.addWidget(ab)
        mb = QPushButton("Mix & Save"); mb.setStyleSheet("QPushButton{background-color:#e94560;color:white;font-weight:bold;}")
        mb.clicked.connect(self._mix); br.addWidget(mb); layout.addLayout(br)
        mg = QGroupBox("Master"); ml = QHBoxLayout(mg)
        ml.addWidget(QLabel("Master:")); self.ms = QSlider(Qt.Horizontal); self.ms.setRange(0, 200); self.ms.setValue(100)
        ml.addWidget(self.ms); self.ml = QLabel("100%"); self.ml.setFixedWidth(40); ml.addWidget(self.ml)
        self.ms.valueChanged.connect(lambda v: self.ml.setText(f"{v}%"))
        ml.addWidget(QLabel("Loop to:")); self.ld = QDoubleSpinBox(); self.ld.setRange(0, 36000); self.ld.setValue(3600); self.ld.setSuffix(" s"); self.ld.setSingleStep(300); ml.addWidget(self.ld)
        self.lc = QCheckBox("Loop"); ml.addWidget(self.lc); layout.addWidget(mg)
        pg = QGroupBox("ASMR Processing"); pl = QVBoxLayout(pg)
        self.pc = QCheckBox("Apply ASMR processing"); self.pc.setChecked(True); pl.addWidget(self.pc)
        er = QHBoxLayout(); er.addWidget(QLabel("Low-pass:")); self.lp = QDoubleSpinBox(); self.lp.setRange(1000, 20000); self.lp.setValue(8000); self.lp.setSuffix(" Hz"); er.addWidget(self.lp)
        er.addWidget(QLabel("Noise floor:")); self.nf = QDoubleSpinBox(); self.nf.setRange(-80, -20); self.nf.setValue(-50); self.nf.setSuffix(" dB"); er.addWidget(self.nf); pl.addLayout(er); layout.addWidget(pg)
        self.sl = QLabel("Add layers to create a mixed ASMR soundscape."); self.sl.setStyleSheet("color:#8888bb;font-size:11px;"); layout.addWidget(self.sl)
    def _add(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if not path: return
        c = self._ac.get(path) if self._ac else None
        if c: sr, audio = c
        else:
            l = AudioDataCache.load_from_disk(path)
            if not l: QMessageBox.warning(self, "Error", "Failed to load."); return
            sr, audio = l; self._ac.put(path, sr, audio) if self._ac else None
        self._layers.append({'name': os.path.splitext(os.path.basename(path))[0], 'path': path, 'sr': sr, 'audio': audio, 'gain': 1.0, 'muted': False})
        self._refresh(); self.sl.setText(f"Layer added ({len(audio)/sr:.1f}s). {len(self._layers)} layer(s).")
    def _refresh(self):
        self.lt.setRowCount(len(self._layers))
        for i, l in enumerate(self._layers):
            self.lt.setItem(i, 0, QTableWidgetItem(l['name'])); self.lt.setItem(i, 1, QTableWidgetItem(f"{len(l['audio'])/l['sr']:.1f}s"))
            v = QSlider(Qt.Horizontal); v.setRange(0, 200); v.setValue(int(l['gain']*100)); v.valueChanged.connect(lambda val, idx=i: self._set_g(idx, val/100)); self.lt.setCellWidget(i, 2, v)
            m = QCheckBox(); m.setChecked(l['muted']); m.toggled.connect(lambda c, idx=i: self._set_m(idx, c)); self.lt.setCellWidget(i, 3, m)
            r = QPushButton("✕"); r.setStyleSheet("color:#e94560;"); r.clicked.connect(lambda _, idx=i: self._rm(idx)); self.lt.setCellWidget(i, 4, r)
    def _set_g(self, i, g):
        if 0 <= i < len(self._layers): self._layers[i]['gain'] = g
    def _set_m(self, i, m):
        if 0 <= i < len(self._layers): self._layers[i]['muted'] = m
    def _rm(self, i):
        if 0 <= i < len(self._layers): self._layers.pop(i); self._refresh()
    def _mix(self):
        if not self._layers: QMessageBox.warning(self, "No Layers", "Add at least one."); return
        tsr = max(l['sr'] for l in self._layers); ms = max(int(len(l['audio'])*tsr/l['sr']) for l in self._layers)
        mixed = np.zeros(ms, dtype=np.float32); mg = self.ms.value()/100
        for l in self._layers:
            if l['muted']: continue
            a = l['audio']
            if l['sr'] != tsr: a = np.interp(np.linspace(0, len(a)-1, int(len(a)*tsr/l['sr'])), np.arange(len(a)), a).astype(np.float32)
            if len(a) < ms: a = np.pad(a, (0, ms-len(a)))
            elif len(a) > ms: a = a[:ms]
            mixed += a * l['gain']
        mixed *= mg; pk = float(np.max(np.abs(mixed)))
        if pk > 0.98: mixed *= 0.98/pk
        if self.pc.isChecked():
            self.sl.setText("Processing..."); QApplication.processEvents()
            mixed = process_asmr_audio(mixed, tsr, self.lp.value(), noise_floor_db=self.nf.value())
        if self.lc.isChecked():
            self.sl.setText(f"Looping to {format_time(self.ld.value())}..."); QApplication.processEvents()
            mixed = create_seamless_loop(mixed, tsr, self.ld.value())
        path, _ = QFileDialog.getSaveFileName(self, "Save Mixed", "mixed_asmr.wav", "WAV (*.wav)")
        if path:
            wavfile.write(path, tsr, (np.clip(mixed, -1, 1)*32767).astype(np.int16))
            self.sl.setText(f"✅ Saved: {path}"); self.mixed_audio_ready.emit(path)
            QMessageBox.information(self, "Mixed", f"Mixed {len(self._layers)} layers\nDuration: {format_time(len(mixed)/tsr)}")

# ═══════════════════════════════════════════════════════════════════════
# ASMR Batch Queue + System Tray + Misc Widgets
# ═══════════════════════════════════════════════════════════════════════
class ASMRBatchQueue:
    FILE = os.path.join(CWD, "asmr_queue.json")
    def __init__(self): self.items = []; self.idx = 0; self.running = False
    def add(self, name, overrides=None):
        if name in ASMR_PRESETS:
            item = ASMR_PRESETS[name].copy(); item['name'] = name
            if overrides: item.update(overrides)
            self.items.append(item); self.save()
    def remove(self, i):
        if 0 <= i < len(self.items): self.items.pop(i); self.idx = min(self.idx, len(self.items)-1) if self.items else 0; self.save()
    def clear(self): self.items = []; self.idx = 0; self.save()
    def get_current(self): return self.items[self.idx] if self.idx < len(self.items) else None
    def advance(self): self.idx += 1; self.save()
    def is_done(self): return self.idx >= len(self.items)
    def save(self):
        try:
            with open(self.FILE, 'w') as f: json.dump({'items': self.items, 'idx': self.idx}, f, indent=2)
        except: pass
    def load(self):
        try:
            if os.path.exists(self.FILE):
                with open(self.FILE) as f: d = json.load(f); self.items = d.get('items', []); self.idx = d.get('idx', 0)
        except: pass

class ASMRSystemTray(QSystemTrayIcon):
    def __init__(self, parent=None):
        super().__init__(parent); self._p = parent
        pix = QPixmap(32, 32); pix.fill(QColor(233, 69, 96))
        p = QPainter(pix); p.setPen(QPen(QColor(255, 255, 255), 2)); p.setFont(QFont("Arial", 14, QFont.Bold))
        p.drawText(pix.rect(), Qt.AlignCenter, "A"); p.end()
        self.setIcon(QIcon(pix)); self.setToolTip(APP_NAME)
        m = QMenu(); m.addAction("Show", self._show); m.addSeparator()
        self.sa = m.addAction("Status: Ready"); self.sa.setEnabled(False); m.addSeparator()
        m.addAction("Quit", self._quit); self.setContextMenu(m)
        self.activated.connect(self._on)
    def _show(self):
        w = self._p; w.show(); w.raise_(); w.activateWindow()
        w.setWindowState(w.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    def _quit(self): self._p.close()
    def _on(self, r):
        if r == QSystemTrayIcon.DoubleClick: self._show()
    def notify(self, title, msg):
        if self.supportsMessages(): self.showMessage(title, msg, QSystemTrayIcon.Information, 5000)

class ClickableLabel(QLabel):
    clicked = Signal()
    def mousePressEvent(self, e): super().mousePressEvent(e); self.clicked.emit()

class PromptHistoryLineEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent); self.history = []; self.hi = -1; self._st = ""
        self.setPlaceholderText("Describe the audio... (Ctrl+Enter to generate, Ctrl+Up/Down for history)")
        self.setMaximumHeight(70); self.setTabChangesFocus(True)
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up and (e.modifiers() & Qt.ControlModifier): self._nav(-1)
        elif e.key() == Qt.Key_Down and (e.modifiers() & Qt.ControlModifier): self._nav(1)
        elif e.key() == Qt.Key_Return and (e.modifiers() & Qt.ControlModifier): self.window().start_generation()
        else: super().keyPressEvent(e)
    def _nav(self, d):
        if not self.history: return
        if self.hi == -1: self._st = self.toPlainText()
        ni = max(-1, min(len(self.history)-1, self.hi+d))
        if ni == self.hi: return
        self.hi = ni; self.setPlainText(self._st if self.hi == -1 else self.history[self.hi])
    def add_to_history(self, t):
        if t and (not self.history or self.history[-1] != t):
            self.history.append(t)
            if len(self.history) > MAX_PROMPT_HISTORY: self.history.pop(0)
            self.hi = -1

# ═══════════════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════════════
class AudioLDM2Studio(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1400, 900); self.setMinimumSize(1100, 700); log_system_info()
        if not ML_AVAILABLE:
            QMessageBox.critical(self, "Missing Dependencies",
                "ML libraries not installed.\npip install torch diffusers transformers")
            sys.exit(1)
        self.settings = JsonSettings(os.path.join(CWD, "config.json"))
        self._audio_cache = AudioDataCache(AUDIO_CACHE_MAX_ENTRIES)
        self.worker_thread = None; self.worker = None; self.video_worker = None
        self.current_audio_path = None; self.current_audio_paths = []
        self.batch_state = BatchState.IDLE; self.batch_prompts = []; self.batch_index = 0
        self.setAcceptDrops(True)
        self.check_library_versions(); self.setup_ui(); self.setup_audio()
        self.setup_statusbar(); self.load_settings(); self.setup_shortcuts()
        self.check_interrupted_batch()
        self._thumb_worker = ThumbnailWorker(self.playlist_table, self)
        self._thumb_worker.thumbReady.connect(self._on_thumb_ready); self._thumb_worker.start()
        self.autosave_timer = QTimer(self); self.autosave_timer.timeout.connect(self.auto_save_state)
        self.autosave_timer.start(AUTO_SAVE_INTERVAL_MS)
        self._tray = ASMRSystemTray(self); self._tray.show()
        self._asmr_queue = ASMRBatchQueue(); self._asmr_queue.load()

    def _on_thumb_ready(self, row, pix):
        if 0 <= row < self.playlist_table.rowCount():
            item = self.playlist_table.item(row, 1)
            if isinstance(item, MiniWaveformItem): item.set_pixmap(pix)

    def check_library_versions(self):
        try:
            if int(transformers.__version__.split('.')[0]) >= 4 and int(transformers.__version__.split('.')[1]) >= 40:
                if int(diffusers.__version__.split('.')[0]) == 0 and int(diffusers.__version__.split('.')[1]) < 29:
                    QMessageBox.warning(self, "Version Conflict", "pip install --upgrade diffusers transformers")
        except: pass

    def setup_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QHBoxLayout(cw); ml.setContentsMargins(2,2,2,2); ml.setSpacing(2)
        sp = QSplitter(Qt.Horizontal); ml.addWidget(sp)
        self.create_left_panel(sp); self.create_right_panel(sp)
        sp.setSizes([440, 960]); sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        self.create_menus()
        try:
            g = self.settings.value("geometry")
            if g: self.restoreGeometry(g)
            s = self.settings.value("windowState")
            if s: self.restoreState(s)
        except: pass

    def create_left_panel(self, parent):
        lp = QWidget(); lp.setMinimumWidth(400)
        ll = QVBoxLayout(lp); ll.setContentsMargins(0,0,0,0); ll.setSpacing(4)
        self.mode_tabs = QTabWidget(); ll.addWidget(self.mode_tabs)
        self._create_generate_tab(); self._create_batch_tab()
        self._create_video_studio_tab(); self._create_asmr_studio_tab()
        pg = QGroupBox("Status"); pl = QVBoxLayout(pg); pl.setSpacing(4); pl.setContentsMargins(8,16,8,8)
        pr = QHBoxLayout(); self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False); pr.addWidget(self.progress_bar, 1)
        self.eta_label = QLabel(""); self.eta_label.setMinimumWidth(70); self.eta_label.setAlignment(Qt.AlignRight|Qt.AlignVCenter); pr.addWidget(self.eta_label); pl.addLayout(pr)
        self.status_label = QLabel("Ready"); self.status_label.setStyleSheet("font-weight:bold;"); pl.addWidget(self.status_label)
        ll.addWidget(pg); parent.addWidget(lp)

    def _create_generate_tab(self):
        w = QWidget(); ol = QVBoxLayout(w); ol.setContentsMargins(0,0,0,0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        c = QWidget(); gl = QVBoxLayout(c); gl.setContentsMargins(8,8,8,8); gl.setSpacing(8)
        pg = QGroupBox("Prompt"); pl = QVBoxLayout(pg); pl.setSpacing(6)
        cr = QHBoxLayout(); cr.setSpacing(2); cr.addWidget(QLabel("Quick:"))
        for tag in ["Rain","Impact","Ocean","Music","Bass","Engine"]:
            b = QPushButton(tag); b.setStyleSheet("padding:2px 4px;font-size:10px;min-height:18px;")
            b.clicked.connect(lambda _, t=tag: self.prompt_input.setPlainText(self.prompt_input.toPlainText()+" "+t)); cr.addWidget(b)
        cr.addStretch(); pl.addLayout(cr)
        pr = QHBoxLayout(); pr.addWidget(QLabel("Presets:")); self.preset_combo = QComboBox()
        self.preset_combo.addItems([""]+DEFAULT_PROMPT_PRESETS); self.preset_combo.currentTextChanged.connect(self.on_preset_selected); pr.addWidget(self.preset_combo, 1); pl.addLayout(pr)
        self.prompt_input = PromptHistoryLineEdit(); pl.addWidget(self.prompt_input)
        nr = QHBoxLayout(); nr.addWidget(QLabel("Negative:")); self.neg_preset_combo = QComboBox()
        self.neg_preset_combo.addItems([""]+NEGATIVE_PROMPT_PRESETS); self.neg_preset_combo.currentTextChanged.connect(self.on_neg_preset_selected); nr.addWidget(self.neg_preset_combo, 1); pl.addLayout(nr)
        self.neg_prompt_input = QLineEdit(); self.neg_prompt_input.setPlaceholderText("What to avoid..."); pl.addWidget(self.neg_prompt_input)
        gl.addWidget(pg)
        pm = QGroupBox("Parameters"); fl = QFormLayout(pm); fl.setSpacing(6); fl.setLabelAlignment(Qt.AlignRight)
        mr = QHBoxLayout(); self.model_combo = QComboBox(); self.model_combo.setEditable(True); self.model_combo.addItems(DEFAULT_MODELS); mr.addWidget(self.model_combo, 1)
        sm = QToolButton(); sm.setText("+"); sm.setToolTip("Save model"); sm.clicked.connect(self.save_current_model); mr.addWidget(sm); fl.addRow("Model:", mr)
        self.duration_spin = QDoubleSpinBox(); self.duration_spin.setRange(1,600); self.duration_spin.setValue(10); self.duration_spin.setSuffix(" s"); fl.addRow("Duration:", self.duration_spin)
        self.chunk_size_spin = QDoubleSpinBox(); self.chunk_size_spin.setRange(MIN_CHUNK_S, 60); self.chunk_size_spin.setValue(10); self.chunk_size_spin.setSingleStep(0.5); self.chunk_size_spin.setSuffix(" s"); fl.addRow("Chunk Size:", self.chunk_size_spin)
        self.chunk_preview_label = QLabel(""); self.chunk_preview_label.setStyleSheet("color:#888;font-style:italic;font-size:10px;"); fl.addRow("", self.chunk_preview_label)
        self.duration_spin.valueChanged.connect(self.update_chunk_preview); self.chunk_size_spin.valueChanged.connect(self.update_chunk_preview)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(10, 500); self.steps_spin.setValue(200); fl.addRow("Steps:", self.steps_spin)
        self.guidance_spin = QDoubleSpinBox(); self.guidance_spin.setRange(1, 20); self.guidance_spin.setValue(3.5); self.guidance_spin.setSingleStep(0.5); fl.addRow("Guidance:", self.guidance_spin)
        sr = QHBoxLayout(); self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 999999999); self.seed_spin.setValue(-1); sr.addWidget(self.seed_spin, 1)
        rs = QToolButton(); rs.setText("D"); rs.clicked.connect(lambda: self.seed_spin.setValue(-1)); sr.addWidget(rs); fl.addRow("Seed:", sr)
        self.variations_spin = QSpinBox(); self.variations_spin.setRange(1, 10); self.variations_spin.setValue(1); fl.addRow("Variations:", self.variations_spin)
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto","cpu","cuda","mps"])
        if torch.cuda.is_available(): self.device_combo.setCurrentText("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device_combo.setCurrentText("mps")
        self.device_combo.currentTextChanged.connect(self.on_device_changed); fl.addRow("Device:", self.device_combo)
        cr2 = QHBoxLayout(); self.cache_dir_edit = QLineEdit(); self.cache_dir_edit.setPlaceholderText(DEFAULT_CACHE_DIR); cb = QPushButton("..."); cb.setFixedWidth(30); cb.clicked.connect(self.browse_cache_dir); cr2.addWidget(self.cache_dir_edit); cr2.addWidget(cb); fl.addRow("Cache:", cr2)
        or2 = QHBoxLayout(); self.output_dir_edit = QLineEdit(); self.output_dir_edit.setPlaceholderText("./generated_audio"); ob = QPushButton("..."); ob.setFixedWidth(30); ob.clicked.connect(self.browse_output_dir); or2.addWidget(self.output_dir_edit); or2.addWidget(ob)
        oo = QPushButton("F"); oo.setFixedWidth(30); oo.clicked.connect(self.open_output_folder); or2.addWidget(oo); fl.addRow("Output:", or2)
        self.section_edit = QLineEdit(); self.section_edit.setPlaceholderText("e.g., SFX, Music"); fl.addRow("Section:", self.section_edit)
        self.cpu_offload_check = QCheckBox("CPU Offload (Low VRAM)"); fl.addRow("", self.cpu_offload_check)
        gl.addWidget(pm)
        pp = QGroupBox("Post-Processing"); ppl = QVBoxLayout(pp); ppl.setSpacing(4)
        self.normalize_check = QCheckBox("Normalize"); self.normalize_check.setChecked(True); ppl.addWidget(self.normalize_check)
        self.trim_check = QCheckBox("Trim silence"); self.trim_check.setChecked(True); ppl.addWidget(self.trim_check)
        fr = QHBoxLayout(); self.fade_in_check = QCheckBox("Fade in"); self.fade_in_check.setChecked(True); fr.addWidget(self.fade_in_check)
        self.fade_out_check = QCheckBox("Fade out"); self.fade_out_check.setChecked(True); fr.addWidget(self.fade_out_check); ppl.addLayout(fr)
        er = QHBoxLayout(); er.addWidget(QLabel("HP:")); self.hp_spin = QDoubleSpinBox(); self.hp_spin.setRange(0,20000); self.hp_spin.setSuffix(" Hz"); self.hp_spin.setSpecialValueText("Off"); er.addWidget(self.hp_spin)
        er.addWidget(QLabel("LP:")); self.lp_spin = QDoubleSpinBox(); self.lp_spin.setRange(0,20000); self.lp_spin.setSuffix(" Hz"); self.lp_spin.setSpecialValueText("Off"); er.addWidget(self.lp_spin); ppl.addLayout(er)
        gl.addWidget(pp)
        gr = QHBoxLayout(); self.generate_btn = QPushButton("Generate Audio"); self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet("QPushButton{background-color:#e94560;color:white;font-size:14px;font-weight:bold;border:none;border-radius:6px;}QPushButton:hover{background-color:#ff5577;}QPushButton:disabled{background-color:#333;color:#666;}")
        self.generate_btn.clicked.connect(self.start_generation); gr.addWidget(self.generate_btn)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setMinimumHeight(40); self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton{background-color:#c0392b;color:white;font-weight:bold;border:none;border-radius:6px;}QPushButton:disabled{background-color:#333;color:#666;}")
        self.cancel_btn.clicked.connect(self.cancel_generation); gr.addWidget(self.cancel_btn)
        gl.addLayout(gr); gl.addStretch(); sc.setWidget(c); ol.addWidget(sc); self.mode_tabs.addTab(w, "Generate")

    def _create_batch_tab(self):
        w = QWidget(); ol = QVBoxLayout(w); ol.setContentsMargins(0,0,0,0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        c = QWidget(); bl = QVBoxLayout(c); bl.setContentsMargins(8,8,8,8); bl.setSpacing(8)
        bl.addWidget(QLabel("Select a .txt file with one prompt per line:"))
        self.batch_file_btn = QPushButton("Select Text File"); self.batch_file_btn.clicked.connect(self.select_batch_file); bl.addWidget(self.batch_file_btn)
        self.batch_file_label = QLabel("No file selected"); self.batch_file_label.setWordWrap(True); self.batch_file_label.setStyleSheet("color:#888;font-style:italic;"); bl.addWidget(self.batch_file_label)
        bsg = QGroupBox("Batch Settings"); bsl = QFormLayout(bsg); bsl.setSpacing(6); bsl.setLabelAlignment(Qt.AlignRight)
        self.batch_variations_spin = QSpinBox(); self.batch_variations_spin.setRange(1,10); self.batch_variations_spin.setValue(1); bsl.addRow("Variations:", self.batch_variations_spin)
        self.batch_delay_spin = QDoubleSpinBox(); self.batch_delay_spin.setRange(0,30); self.batch_delay_spin.setValue(1.0); self.batch_delay_spin.setSuffix(" s"); bsl.addRow("Delay:", self.batch_delay_spin)
        self.batch_continue_on_error_check = QCheckBox("Continue on Error"); self.batch_continue_on_error_check.setChecked(True); bsl.addRow("", self.batch_continue_on_error_check)
        bl.addWidget(bsg)
        self.batch_start_btn = QPushButton("Start Batch"); self.batch_start_btn.setMinimumHeight(40)
        self.batch_start_btn.setStyleSheet("QPushButton{background-color:#27ae60;color:white;font-weight:bold;border:none;border-radius:6px;}QPushButton:disabled{background-color:#333;color:#666;}")
        self.batch_start_btn.clicked.connect(self.start_batch); bl.addWidget(self.batch_start_btn)
        self.batch_progress_label = QLabel(""); self.batch_progress_label.setWordWrap(True); bl.addWidget(self.batch_progress_label)
        bl.addWidget(QLabel("Batch Log:")); self.batch_log = QPlainTextEdit(); self.batch_log.setReadOnly(True); self.batch_log.setMaximumBlockCount(1000); bl.addWidget(self.batch_log); bl.addStretch()
        sc.setWidget(c); ol.addWidget(sc); self.mode_tabs.addTab(w, "Batch")

    def _create_video_studio_tab(self):
        w = QWidget(); ol = QVBoxLayout(w); ol.setContentsMargins(0,0,0,0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        c = QWidget(); vl = QVBoxLayout(c); vl.setContentsMargins(8,8,8,8); vl.setSpacing(8)
        vl.addWidget(QLabel("Create MP4 videos with audio-reactive visuals."))
        sg = QGroupBox("Audio Source"); sl = QFormLayout(sg); sl.setLabelAlignment(Qt.AlignRight)
        self.video_audio_path_edit = QLineEdit(); self.video_audio_path_edit.setReadOnly(True); sl.addRow("Audio:", self.video_audio_path_edit)
        ba = QPushButton("Browse..."); ba.clicked.connect(self._video_browse_audio); sl.addRow("", ba)
        self.video_use_current_btn = QPushButton("Use Current Audio"); self.video_use_current_btn.clicked.connect(self._video_use_current_audio); sl.addRow("", self.video_use_current_btn)
        vl.addWidget(sg)
        vg = QGroupBox("Visual Style"); vsl = QFormLayout(vg); vsl.setLabelAlignment(Qt.AlignRight)
        self.video_style_combo = QComboBox(); self.video_style_combo.addItems(VIDEO_STYLES); self.video_style_combo.currentTextChanged.connect(lambda s: self.viz_style_combo.setCurrentText(s)); vsl.addRow("Style:", self.video_style_combo)
        self.video_theme_combo = QComboBox(); self.video_theme_combo.addItems(list(VIDEO_COLOR_THEMES.keys())); vsl.addRow("Theme:", self.video_theme_combo)
        vl.addWidget(vg)
        rg = QGroupBox("Video Settings"); rl = QFormLayout(rg); rl.setLabelAlignment(Qt.AlignRight)
        self.video_res_combo = QComboBox(); self.video_res_combo.addItems(list(VIDEO_RESOLUTIONS.keys())); self.video_res_combo.setCurrentText("1080p (1920×1080)"); rl.addRow("Resolution:", self.video_res_combo)
        self.video_fps_combo = QComboBox()
        for f in VIDEO_FPS_OPTIONS: self.video_fps_combo.addItem(f"{f} fps", f)
        self.video_fps_combo.setCurrentIndex(1); rl.addRow("FPS:", self.video_fps_combo)
        self.video_preset_combo = QComboBox(); self.video_preset_combo.addItems(["ultrafast","fast","medium","slow"]); self.video_preset_combo.setCurrentText("slow"); rl.addRow("Encoding:", self.video_preset_combo)
        self.video_crf_spin = QSpinBox(); self.video_crf_spin.setRange(0,51); self.video_crf_spin.setValue(18); rl.addRow("CRF:", self.video_crf_spin)
        vl.addWidget(rg)
        og = QGroupBox("Output"); ol2 = QFormLayout(og); ol2.setLabelAlignment(Qt.AlignRight)
        self.video_output_edit = QLineEdit(); self.video_output_edit.setPlaceholderText("video.mp4"); ol2.addRow("File:", self.video_output_edit)
        bo = QPushButton("Browse..."); bo.clicked.connect(self._video_browse_output); ol2.addRow("", bo); vl.addWidget(og)
        self.video_export_btn = QPushButton("Export MP4"); self.video_export_btn.setMinimumHeight(44)
        self.video_export_btn.setStyleSheet("QPushButton{background-color:#e94560;color:white;font-weight:bold;border:none;border-radius:6px;}QPushButton:disabled{background-color:#333;color:#666;}")
        self.video_export_btn.clicked.connect(self._video_start_export); vl.addWidget(self.video_export_btn)
        self.video_cancel_btn = QPushButton("Cancel"); self.video_cancel_btn.setEnabled(False); self.video_cancel_btn.clicked.connect(self._video_cancel_export); vl.addWidget(self.video_cancel_btn)
        self.video_progress_bar = QProgressBar(); self.video_progress_bar.setVisible(False); vl.addWidget(self.video_progress_bar)
        self.video_status_label = QLabel(""); self.video_status_label.setWordWrap(True); vl.addWidget(self.video_status_label)
        vl.addStretch(); sc.setWidget(c); ol.addWidget(sc); self.mode_tabs.addTab(w, "Video Studio")

    def _create_asmr_studio_tab(self):
        w = QWidget(); ol = QVBoxLayout(w); ol.setContentsMargins(0,0,0,0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        c = QWidget(); al = QVBoxLayout(c); al.setContentsMargins(8,8,8,8); al.setSpacing(8)
        h = QLabel("ASMR Studio"); h.setStyleSheet("font-size:16px;font-weight:bold;color:#e94560;padding:4px;"); al.addWidget(h)
        info = QLabel("Generate ASMR audio + create YouTube-ready video in one click."); info.setWordWrap(True); info.setStyleSheet("color:#8888bb;font-size:11px;"); al.addWidget(info)
        pg = QGroupBox("ASMR Preset"); pl = QVBoxLayout(pg)
        self.asmr_preset_combo = QComboBox(); self.asmr_preset_combo.addItems(list(ASMR_PRESETS.keys()))
        self.asmr_preset_combo.currentTextChanged.connect(self._asmr_preset_selected); pl.addWidget(self.asmr_preset_combo); al.addWidget(pg)
        ppg = QGroupBox("Prompt"); ppl = QVBoxLayout(ppg)
        self.asmr_prompt_edit = QPlainTextEdit(); self.asmr_prompt_edit.setMaximumHeight(80); ppl.addWidget(self.asmr_prompt_edit)
        self.asmr_neg_edit = QLineEdit(); self.asmr_neg_edit.setPlaceholderText("Negative prompt..."); ppl.addWidget(self.asmr_neg_edit); al.addWidget(ppg)
        gg = QGroupBox("Audio Generation"); gl = QFormLayout(gg); gl.setLabelAlignment(Qt.AlignRight)
        self.asmr_model_combo = QComboBox(); self.asmr_model_combo.setEditable(True); self.asmr_model_combo.addItems(DEFAULT_MODELS); gl.addRow("Model:", self.asmr_model_combo)
        self.asmr_duration_spin = QDoubleSpinBox(); self.asmr_duration_spin.setRange(1, 36000); self.asmr_duration_spin.setSuffix(" s"); self.asmr_duration_spin.setSingleStep(60); gl.addRow("Duration:", self.asmr_duration_spin)
        self.asmr_steps_spin = QSpinBox(); self.asmr_steps_spin.setRange(10, 500); self.asmr_steps_spin.setValue(200); gl.addRow("Steps:", self.asmr_steps_spin)
        self.asmr_guidance_spin = QDoubleSpinBox(); self.asmr_guidance_spin.setRange(1, 20); self.asmr_guidance_spin.setValue(3.5); self.asmr_guidance_spin.setSingleStep(0.5); gl.addRow("Guidance:", self.asmr_guidance_spin)
        self.asmr_chunk_spin = QDoubleSpinBox(); self.asmr_chunk_spin.setRange(2, 60); self.asmr_chunk_spin.setValue(10); self.asmr_chunk_spin.setSuffix(" s"); gl.addRow("Chunk:", self.asmr_chunk_spin)
        self.asmr_seed_spin = QSpinBox(); self.asmr_seed_spin.setRange(-1, 999999999); self.asmr_seed_spin.setValue(-1); gl.addRow("Seed:", self.asmr_seed_spin); al.addWidget(gg)
        lg = QGroupBox("Audio Looping"); ll = QVBoxLayout(lg)
        li = QLabel("Generate short audio, loop to target duration (saves hours of compute)"); li.setWordWrap(True); li.setStyleSheet("color:#8888bb;font-size:10px;"); ll.addWidget(li)
        lr = QHBoxLayout(); self.asmr_loop_check = QCheckBox("Loop to target"); self.asmr_loop_check.setChecked(True); lr.addWidget(self.asmr_loop_check); ll.addLayout(lr)
        tr = QHBoxLayout(); tr.addWidget(QLabel("Target:")); self.asmr_target_duration_spin = QDoubleSpinBox(); self.asmr_target_duration_spin.setRange(1, 36000); self.asmr_target_duration_spin.setValue(3600); self.asmr_target_duration_spin.setSuffix(" s"); self.asmr_target_duration_spin.setSingleStep(600); tr.addWidget(self.asmr_target_duration_spin); ll.addLayout(tr)
        cr = QHBoxLayout(); cr.addWidget(QLabel("Crossfade:")); self.asmr_crossfade_spin = QDoubleSpinBox(); self.asmr_crossfade_spin.setRange(0.5, 10); self.asmr_crossfade_spin.setValue(3); self.asmr_crossfade_spin.setSuffix(" s"); cr.addWidget(self.asmr_crossfade_spin); ll.addLayout(cr); al.addWidget(lg)
        prg = QGroupBox("ASMR Processing"); prl = QVBoxLayout(prg)
        self.asmr_process_check = QCheckBox("Apply ASMR processing (EQ+compression+noise)"); self.asmr_process_check.setChecked(True); prl.addWidget(self.asmr_process_check)
        er = QHBoxLayout(); er.addWidget(QLabel("Low-pass:")); self.asmr_lp_spin = QDoubleSpinBox(); self.asmr_lp_spin.setRange(1000, 20000); self.asmr_lp_spin.setValue(8000); self.asmr_lp_spin.setSuffix(" Hz"); er.addWidget(self.asmr_lp_spin)
        er.addWidget(QLabel("Noise:")); self.asmr_nf_spin = QDoubleSpinBox(); self.asmr_nf_spin.setRange(-80, -20); self.asmr_nf_spin.setValue(-50); self.asmr_nf_spin.setSuffix(" dB"); er.addWidget(self.asmr_nf_spin); prl.addLayout(er); al.addWidget(prg)
        vg = QGroupBox("Visual Settings"); vl2 = QFormLayout(vg); vl2.setLabelAlignment(Qt.AlignRight)
        self.asmr_visual_combo = QComboBox(); self.asmr_visual_combo.addItems(ASMR_VISUAL_STYLES); vl2.addRow("Style:", self.asmr_visual_combo)
        self.asmr_theme_combo = QComboBox(); self.asmr_theme_combo.addItems(list(VIDEO_COLOR_THEMES.keys())); vl2.addRow("Theme:", self.asmr_theme_combo)
        mr = QHBoxLayout(); mr.addWidget(QLabel("Motion:")); self.asmr_motion_slider = QSlider(Qt.Horizontal); self.asmr_motion_slider.setRange(1, 200); self.asmr_motion_slider.setValue(30); mr.addWidget(self.asmr_motion_slider)
        self.asmr_motion_label = QLabel("0.30x"); self.asmr_motion_label.setFixedWidth(40); mr.addWidget(self.asmr_motion_label)
        self.asmr_motion_slider.valueChanged.connect(lambda v: self.asmr_motion_label.setText(f"{v/100:.2f}x")); vl2.addRow(mr)
        sr = QHBoxLayout(); sr.addWidget(QLabel("Softness:")); self.asmr_soft_slider = QSlider(Qt.Horizontal); self.asmr_soft_slider.setRange(0, 100); self.asmr_soft_slider.setValue(60); sr.addWidget(self.asmr_soft_slider)
        self.asmr_soft_label = QLabel("0.60"); self.asmr_soft_label.setFixedWidth(40); sr.addWidget(self.asmr_soft_label)
        self.asmr_soft_slider.valueChanged.connect(lambda v: self.asmr_soft_label.setText(f"{v/100:.2f}")); vl2.addRow(sr); al.addWidget(vg)
        eg = QGroupBox("Video Export"); el = QFormLayout(eg); el.setLabelAlignment(Qt.AlignRight)
        self.asmr_encoding_combo = QComboBox(); self.asmr_encoding_combo.addItems(list(YOUTUBE_ENCODING_PRESETS.keys())); el.addRow("Encoding:", self.asmr_encoding_combo)
        self.asmr_output_edit = QLineEdit(); self.asmr_output_edit.setPlaceholderText("asmr_video.mp4"); el.addRow("Output:", self.asmr_output_edit)
        eb = QPushButton("Browse..."); eb.clicked.connect(self._asmr_browse_output); el.addRow("", eb); al.addWidget(eg)
        self.asmr_estimate_label = QLabel(""); self.asmr_estimate_label.setWordWrap(True); self.asmr_estimate_label.setStyleSheet("color:#8888bb;font-size:11px;"); al.addWidget(self.asmr_estimate_label)
        self.asmr_duration_spin.valueChanged.connect(self._asmr_update_estimate); self.asmr_encoding_combo.currentTextChanged.connect(self._asmr_update_estimate); self.asmr_target_duration_spin.valueChanged.connect(self._asmr_update_estimate); self.asmr_loop_check.toggled.connect(self._asmr_update_estimate)
        sb = QPushButton("Open Sound Mixer"); sb.clicked.connect(self._open_sound_mixer); al.addWidget(sb)
        bqg = QGroupBox("Batch Queue"); bql = QVBoxLayout(bqg)
        self.asmr_batch_list = QListWidget(); bql.addWidget(self.asmr_batch_list)
        bqr = QHBoxLayout()
        ba = QPushButton("+ Add to Queue"); ba.clicked.connect(self._asmr_add_to_queue); bqr.addWidget(ba)
        br = QPushButton("Remove"); br.clicked.connect(self._asmr_remove_from_queue); bqr.addWidget(br)
        bc = QPushButton("Clear"); bc.clicked.connect(self._asmr_clear_queue); bqr.addWidget(bc); bql.addLayout(bqr)
        self.asmr_batch_start_btn = QPushButton("Start Batch"); self.asmr_batch_start_btn.setStyleSheet("QPushButton{background-color:#27ae60;color:white;font-weight:bold;border:none;border-radius:6px;}QPushButton:disabled{background-color:#333;color:#666;}")
        self.asmr_batch_start_btn.clicked.connect(self._asmr_start_batch); bql.addWidget(self.asmr_batch_start_btn); al.addWidget(bqg)
        pv = QPushButton("Preview 10s (480p)"); pv.clicked.connect(self._asmr_preview_render); al.addWidget(pv)
        self.asmr_generate_btn = QPushButton("Generate ASMR Video"); self.asmr_generate_btn.setMinimumHeight(50)
        self.asmr_generate_btn.setStyleSheet("QPushButton{background-color:#e94560;color:white;font-size:15px;font-weight:bold;border:none;border-radius:8px;}QPushButton:hover{background-color:#ff5577;}QPushButton:disabled{background-color:#333;color:#666;}")
        self.asmr_generate_btn.clicked.connect(self._asmr_one_click_generate); al.addWidget(self.asmr_generate_btn)
        self.asmr_progress_bar = QProgressBar(); self.asmr_progress_bar.setVisible(False); al.addWidget(self.asmr_progress_bar)
        self.asmr_status_label = QLabel(""); self.asmr_status_label.setWordWrap(True); al.addWidget(self.asmr_status_label)
        al.addStretch(); sc.setWidget(c); ol.addWidget(sc); self.mode_tabs.addTab(w, "ASMR Studio")
        self._asmr_preset_selected(self.asmr_preset_combo.currentText())

    def create_right_panel(self, parent):
        rp = QWidget(); rl = QVBoxLayout(rp); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        self.right_tabs = QTabWidget(); rl.addWidget(self.right_tabs)
        vw = QWidget(); vl = QVBoxLayout(vw); vl.setContentsMargins(8,8,8,8); vl.setSpacing(8)
        vcr = QHBoxLayout(); vcr.addWidget(QLabel("Style:")); self.viz_style_combo = QComboBox()
        self.viz_style_combo.addItems(VIDEO_STYLES); self.viz_style_combo.currentTextChanged.connect(lambda s: self.visualizer_widget.set_style(s)); vcr.addWidget(self.viz_style_combo)
        vcr.addWidget(QLabel("Theme:")); self.viz_theme_combo = QComboBox(); self.viz_theme_combo.addItems(list(VIDEO_COLOR_THEMES.keys()))
        self.viz_theme_combo.currentTextChanged.connect(lambda t: self.visualizer_widget.set_theme(t)); vcr.addWidget(self.viz_theme_combo); vcr.addStretch(); vl.addLayout(vcr)
        self.visualizer_widget = VisualizerWidget(audio_cache=self._audio_cache, style="Electric Sheep", theme="Cosmic")
        self.visualizer_widget.seekRequested.connect(self._seek_to); vl.addWidget(self.visualizer_widget, 3)
        self.viz_secondary_tabs = QTabWidget(); self.viz_secondary_tabs.setMaximumHeight(220)
        self.waveform_widget = WaveformWidget(audio_cache=self._audio_cache); self.waveform_widget.seekRequested.connect(self._seek_to)
        self.viz_secondary_tabs.addTab(self.waveform_widget, "Waveform")
        self.spectrogram_widget = SpectrogramWidget(audio_cache=self._audio_cache); self.spectrogram_widget.seekRequested.connect(self._seek_to)
        self.viz_secondary_tabs.addTab(self.spectrogram_widget, "Spectrogram"); vl.addWidget(self.viz_secondary_tabs, 1)
        self.audio_stats_label = QLabel("Peak: 0.0 dB | RMS: 0.0 dB | Headroom: 0.0 dB"); self.audio_stats_label.setAlignment(Qt.AlignCenter)
        self.audio_stats_label.setStyleSheet("color:#888;font-size:11px;font-family:monospace;padding:2px;"); vl.addWidget(self.audio_stats_label)
        self.right_tabs.addTab(vw, "Visualizer")
        pw = QWidget(); pl = QVBoxLayout(pw); pl.setContentsMargins(8,8,8,8); pl.setSpacing(8)
        pg = QGroupBox("Playback"); ppl = QVBoxLayout(pg); ppl.setSpacing(6)
        pmr = QHBoxLayout(); pbr = QHBoxLayout(); pbr.setSpacing(6)
        self.load_btn = QPushButton("Load"); self.load_btn.setFixedHeight(36); self.load_btn.clicked.connect(self.open_audio); pbr.addWidget(self.load_btn)
        self.play_btn = QPushButton("Play"); self.play_btn.setFixedHeight(36); self.play_btn.setMinimumWidth(80); self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.toggle_playback); pbr.addWidget(self.play_btn)
        self.save_btn = QPushButton("Save"); self.save_btn.setFixedHeight(36); self.save_btn.setEnabled(False); self.save_btn.clicked.connect(self.save_audio); pbr.addWidget(self.save_btn)
        pmr.addLayout(pbr); vr = QHBoxLayout(); vr.addWidget(QLabel("Variation:")); self.variation_combo = QComboBox(); self.variation_combo.setEnabled(False); self.variation_combo.currentIndexChanged.connect(self.on_variation_changed); vr.addWidget(self.variation_combo, 1); pmr.addLayout(vr)
        self.vu_meter = QProgressBar(); self.vu_meter.setOrientation(Qt.Vertical); self.vu_meter.setRange(0, 100); self.vu_meter.setFixedWidth(20); self.vu_meter.setFixedHeight(70); self.vu_meter.setTextVisible(False)
        self.vu_meter.setStyleSheet("QProgressBar{background-color:#0d1b2a;border:1px solid #2a2a4a;border-radius:2px;}QProgressBar::chunk{background-color:#2ecc71;border-radius:1px;}")
        pmr.addWidget(self.vu_meter); ppl.addLayout(pmr)
        vol = QHBoxLayout(); vol.addWidget(QLabel("Volume:")); self.vol_slider = QSlider(Qt.Horizontal); self.vol_slider.setRange(0, 100); self.vol_slider.setValue(50); self.vol_slider.valueChanged.connect(self.set_volume); vol.addWidget(self.vol_slider)
        self.vol_label = QLabel("50%"); self.vol_label.setFixedWidth(35); vol.addWidget(self.vol_label); ppl.addLayout(vol)
        self.time_label = QLabel("00:00 / 00:00"); self.time_label.setAlignment(Qt.AlignCenter); self.time_label.setStyleSheet("font-size:14px;font-weight:bold;color:#e94560;"); ppl.addWidget(self.time_label)
        pl.addWidget(pg); pl.addStretch(); self.right_tabs.addTab(pw, "Player")
        hw = QWidget(); hl = QVBoxLayout(hw); hl.setContentsMargins(8,8,8,8); hl.setSpacing(6)
        hr = QHBoxLayout(); hlbl = QLabel("History"); hlbl.setStyleSheet("font-size:13px;font-weight:bold;"); hr.addWidget(hlbl); hr.addStretch()
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("Search..."); self.search_edit.setFixedWidth(180); hr.addWidget(self.search_edit)
        eh = QPushButton("Export"); eh.setFixedHeight(28); eh.clicked.connect(self.export_history); hr.addWidget(eh)
        ch = QPushButton("Clear"); ch.setFixedHeight(28); ch.clicked.connect(self.clear_history); hr.addWidget(ch); hl.addLayout(hr)
        self._search_timer = QTimer(self); self._search_timer.setSingleShot(True); self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(lambda: self._do_filter(self.search_edit.text())); self.search_edit.textChanged.connect(self._search_timer.start)
        self.playlist_table = QTableWidget(); self.playlist_table.setColumnCount(5)
        self.playlist_table.setHorizontalHeaderLabels(["Prompt","Waveform","Date","Seed","Model"])
        self.playlist_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.playlist_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for i in range(2, 5): self.playlist_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.playlist_table.setSelectionBehavior(QAbstractItemView.SelectRows); self.playlist_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.playlist_table.setAlternatingRowColors(True)
        self.playlist_table.itemDoubleClicked.connect(self.play_from_playlist); self.playlist_table.setContextMenuPolicy(Qt.CustomContextMenu); self.playlist_table.customContextMenuRequested.connect(self.show_history_context_menu)
        hl.addWidget(self.playlist_table); self.right_tabs.addTab(hw, "History")
        parent.addWidget(rp)

    def setup_audio(self):
        self.audio_output = QAudioOutput(self); self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.mediaStatusChanged.connect(self.on_media_status); self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.positionChanged.connect(self.on_position_changed); self.media_player.durationChanged.connect(self.on_duration_changed); self.set_volume(50)

    def setup_statusbar(self):
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_led = QLabel(); self.status_led.setFixedSize(12, 12); self.status_led.setStyleSheet("background-color:#555;border-radius:6px;margin:2px;"); self.status_bar.addPermanentWidget(self.status_led)
        di = "CPU"
        if torch.cuda.is_available(): di = f"GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): di = "MPS"
        self.device_status_label = QLabel(f"Device: {di}"); self.status_bar.addPermanentWidget(self.device_status_label)
        self.gpu_mem_label = ClickableLabel(); self.gpu_mem_label.clicked.connect(self.show_gpu_memory)
        if torch.cuda.is_available():
            self.update_gpu_memory_label(); self.gpu_mem_timer = QTimer(self); self.gpu_mem_timer.timeout.connect(self.update_gpu_memory_label); self.gpu_mem_timer.start(GPU_MEM_POLL_MS)
        else: self.gpu_mem_label.setText("VRAM: N/A")
        self.status_bar.addPermanentWidget(self.gpu_mem_label); self.status_bar.showMessage("Ready")

    def update_gpu_memory_label(self):
        if not torch.cuda.is_available(): return
        a = torch.cuda.memory_allocated()/(1024**3); t = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        self.gpu_mem_label.setText(f"VRAM: {a:.1f}/{t:.1f} GB")

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.start_generation)
        sc = QShortcut(QKeySequence("Space"), self); sc.setContext(Qt.WidgetWithChildrenShortcut); sc.activated.connect(self.toggle_playback)
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_audio)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_audio)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_generation)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(lambda: self.prompt_input.setFocus())
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.regenerate_new_seed)

    def create_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File"); fm.addAction("Export Session", self.export_session); fm.addAction("Import Session", self.import_session); fm.addSeparator(); fm.addAction("Exit", self.close).setShortcut("Ctrl+Q")
        tm = mb.addMenu("&Tools"); tm.addAction("GPU Memory", self.show_gpu_memory); tm.addAction("Clear GPU Cache", self.clear_gpu_cache)
        em = mb.addMenu("&Edit"); em.addAction("Clear Prompt", lambda: self.prompt_input.clear()).setShortcut("Ctrl+Shift+X")
        em.addAction("Paste", self.paste_from_clipboard).setShortcut("Ctrl+V"); em.addAction("Copy Settings", self.copy_settings_to_clipboard)
        em.addSeparator(); em.addAction("Save Preset", self.save_preset); em.addAction("Load Preset", self.load_preset); em.addSeparator(); em.addAction("Reset Settings", self.reset_settings)
        hm = mb.addMenu("&Help"); hm.addAction("Documentation", self.open_docs); hm.addAction("Tips", self.show_prompt_tips); hm.addSeparator(); hm.addAction("About", self.show_about)

    def show_gpu_memory(self):
        if not ML_AVAILABLE or not torch.cuda.is_available(): QMessageBox.information(self, "GPU", "CUDA not available."); return
        a = torch.cuda.memory_allocated()/(1024**3); r = torch.cuda.memory_reserved()/(1024**3); t = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        QMessageBox.information(self, "GPU", f"GPU: {torch.cuda.get_device_name(0)}\nTotal: {t:.2f} GB\nAllocated: {a:.2f} GB\nReserved: {r:.2f} GB\nFree: {t-r:.2f} GB")

    def clear_gpu_cache(self):
        if not ML_AVAILABLE or not torch.cuda.is_available(): return
        gc.collect(); torch.cuda.empty_cache(); self.update_gpu_memory_label(); self.status_bar.showMessage("GPU cache cleared", 3000)

    def reset_settings(self):
        if QMessageBox.question(self, "Reset", "Reset all settings?") == QMessageBox.Yes: self.settings.clear(); self.load_settings()

    def on_preset_selected(self, t):
        if t: self.prompt_input.setPlainText(t)
    def on_neg_preset_selected(self, t):
        if t: self.neg_prompt_input.setText(t)
    def save_current_model(self):
        m = self.model_combo.currentText().strip()
        if m and self.model_combo.findText(m) == -1:
            self.model_combo.addItem(m); self.model_combo.setCurrentText(m)
            cm = self.settings.value("custom_models", [])
            if m not in cm: cm.append(m); self.settings.setValue("custom_models", cm); self.settings.sync()
    def browse_cache_dir(self):
        cur = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        p = QFileDialog.getExistingDirectory(self, "Cache Folder", cur if os.path.isdir(cur) else "")
        if p: self.cache_dir_edit.setText(p)
    def browse_output_dir(self):
        cur = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        p = QFileDialog.getExistingDirectory(self, "Output Folder", cur if os.path.isdir(cur) else "")
        if p: self.output_dir_edit.setText(p)
    def open_output_folder(self):
        out = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        if self.section_edit.text().strip(): out = os.path.join(out, sanitize_filename(self.section_edit.text().strip()))
        os.makedirs(out, exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(out))
    def update_chunk_preview(self):
        d, c = self.duration_spin.value(), self.chunk_size_spin.value()
        if c <= OVERLAP_S: self.chunk_preview_label.setText("Chunk must be > 1.0s"); return
        if d > c:
            n = math.ceil((d - OVERLAP_S) / (c - OVERLAP_S))
            self.chunk_preview_label.setText(f"{n} chunks of ~{c:.1f}s" + (" (many!)" if n > 50 else ""))
        else: self.chunk_preview_label.setText("Single chunk")
    def on_device_changed(self, t):
        self.cpu_offload_check.setEnabled(t != "cpu")
        if t == "cpu": self.cpu_offload_check.setChecked(False)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction(); self.setStyleSheet(STYLESHEET + "QMainWindow{border:3px dashed #e94560;}")
    def dragLeaveEvent(self, e): self.setStyleSheet(STYLESHEET)
    def dropEvent(self, e):
        self.setStyleSheet(STYLESHEET)
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(('.wav','.mp3','.flac')): self._load_audio_file(p); break
            elif p.lower().endswith('.txt'): self._load_batch_file(p); break
    def update_audio_stats(self, path):
        try:
            c = self._audio_cache.get(path)
            if c: sr, d = c
            else:
                l = AudioDataCache.load_from_disk(path)
                if not l: self._set_na(); return
                sr, d = l; self._audio_cache.put(path, sr, d)
            if d.size == 0: self._set_na(); return
            pk = float(np.max(np.abs(d))); pdb = 20*math.log10(pk) if pk > 0 else -120
            rms = float(np.sqrt(np.mean(d**2))); rdb = 20*math.log10(rms) if rms > 0 else -120
            self.audio_stats_label.setText(f"Peak: {pdb:.1f} dB | RMS: {rdb:.1f} dB | Headroom: {-pdb:.1f} dB")
            self.audio_stats_label.setStyleSheet("color:#e94560;font-size:11px;font-family:monospace;padding:2px;")
        except: self._set_na()
    def _set_na(self):
        self.audio_stats_label.setText("Peak: N/A | RMS: N/A | Headroom: N/A"); self.audio_stats_label.setStyleSheet("color:#888;font-size:11px;font-family:monospace;padding:2px;")
    def _load_audio_file(self, path):
        try:
            with open(path, 'rb') as f: f.read(4)
        except OSError as e: QMessageBox.warning(self, "Error", f"Cannot read: {e}"); return
        self._audio_cache.invalidate(path); self.current_audio_path = path; self.current_audio_paths = [path]
        self.variation_combo.blockSignals(True); self.variation_combo.clear(); self.variation_combo.addItem("Default"); self.variation_combo.blockSignals(False)
        self.play_btn.setEnabled(True); self.save_btn.setEnabled(True)
        self.waveform_widget.set_audio(path); self.spectrogram_widget.set_audio(path); self.visualizer_widget.set_audio(path); self.update_audio_stats(path)
        self.right_tabs.setCurrentIndex(0); self.settings.setValue("last_audio_path", path); self.settings.sync()
    def expand_prompt_matrix(self, text):
        m = list(re.finditer(r'\[([^]]+)\]', text))
        if not m: return [text]
        opts = [[o.strip() for o in mm.group(1).split(',')] for mm in m]
        total = 1
        for g in opts:
            total *= len(g)
            if total > PROMPT_MATRIX_MAX: raise ValueError(f"Too many combinations ({total}>{PROMPT_MATRIX_MAX})")
        return [text[:]+"".join(c[i] for i, c in enumerate(list(itertools.product(*opts))[j])) if False else (lambda t: [t:=t.replace(m[i].group(0), c[i], 1) for i in range(len(m))][-1] if len(m)>0 else text)(text) for j, c in enumerate(itertools.product(*opts))]
    def _load_batch_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f: lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            exp = []
            for l in lines: exp.extend(self.expand_prompt_matrix(l))
            self.batch_prompts = exp; self.batch_file_label.setText(path)
            self.batch_log.appendPlainText(f"Loaded {len(lines)} prompts. Expanded to {len(exp)}.")
            self.mode_tabs.setCurrentIndex(1)
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def start_generation(self):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "Busy", "Engine still cleaning up. Wait a moment."); return False
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt: QMessageBox.warning(self, "Error", "Enter a prompt."); return False
        model = self.model_combo.currentText().strip()
        if '/' not in model: QMessageBox.warning(self, "Invalid", "Model must be 'org/model'."); return False
        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        try: os.makedirs(out_dir, exist_ok=True)
        except OSError as e: QMessageBox.critical(self, "Error", f"Cannot create output:\n{e}"); return False
        cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        try: os.makedirs(cache_dir, exist_ok=True)
        except OSError as e: QMessageBox.critical(self, "Error", f"Cannot create cache:\n{e}"); return False
        self.prompt_input.add_to_history(prompt)
        self.generate_btn.setEnabled(False); self.cancel_btn.setEnabled(True); self.play_btn.setEnabled(False); self.save_btn.setEnabled(False); self.variation_combo.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0); self.eta_label.setText(""); self.status_label.setText("Starting...")
        self.status_bar.showMessage("Generating..."); self.status_led.setStyleSheet("background-color:#f1c40f;border-radius:6px;margin:2px;")
        params = GenerationParams(prompt=prompt, negative_prompt=self.neg_prompt_input.text().strip(), model_name=model,
            duration=self.duration_spin.value(), steps=self.steps_spin.value(), guidance=self.guidance_spin.value(),
            seed=self.seed_spin.value(), device=self.device_combo.currentText(), cache_dir=cache_dir, output_dir=out_dir,
            num_variations=self.variations_spin.value(), normalize=self.normalize_check.isChecked(), fade_in=self.fade_in_check.isChecked(),
            fade_out=self.fade_out_check.isChecked(), trim=self.trim_check.isChecked(), chunk_size=self.chunk_size_spin.value(),
            use_cpu_offload=self.cpu_offload_check.isChecked(), section=self.section_edit.text().strip(),
            high_pass_freq=self.hp_spin.value(), low_pass_freq=self.lp_spin.value())
        self.worker_thread = QThread(); self.worker = AudioGenerationWorker(params); self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.worker.status_updated.connect(self.status_label.setText, Qt.QueuedConnection)
        self.worker.eta_updated.connect(self.eta_label.setText, Qt.QueuedConnection)
        self.worker.generation_completed.connect(self.generation_finished, Qt.QueuedConnection)
        self.worker.error_occurred.connect(self.generation_error, Qt.QueuedConnection)
        self.worker.cancelled.connect(self.generation_cancelled, Qt.QueuedConnection)
        self.worker_thread.finished.connect(self.on_worker_thread_finished)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start(); return True
    def regenerate_new_seed(self): self.seed_spin.setValue(-1); self.start_generation()
    def cancel_generation(self):
        if self.batch_state == BatchState.RUNNING:
            if QMessageBox.question(self, "Cancel Batch", "Stop entire batch?") != QMessageBox.Yes: return
            self.batch_state = BatchState.CANCELLED
        if self.worker: self.worker.is_cancelled = True; self.status_label.setText("Cancelling..."); self.cancel_btn.setEnabled(False)
    def generation_cancelled(self):
        self.progress_bar.setVisible(False); self.eta_label.setText(""); self.status_label.setText("Ready")
        self.status_led.setStyleSheet("background-color:#555;border-radius:6px;margin:2px;")
        if self.batch_state == BatchState.RUNNING: self.batch_state = BatchState.CANCELLED
    def generation_finished(self, path, metadata):
        self.current_audio_path = path; self.current_audio_paths = metadata.get('paths', [path])
        self.progress_bar.setVisible(False); self.eta_label.setText(""); self.status_label.setText("Ready")
        self.status_led.setStyleSheet("background-color:#555;border-radius:6px;margin:2px;")
        self.play_btn.setEnabled(True); self.save_btn.setEnabled(True)
        self.variation_combo.blockSignals(True); self.variation_combo.clear()
        if len(self.current_audio_paths) > 1:
            for i in range(len(self.current_audio_paths)): self.variation_combo.addItem(f"Variation {i+1}")
            self.variation_combo.setEnabled(True)
        else: self.variation_combo.addItem("Default")
        self.variation_combo.blockSignals(False)
        self.waveform_widget.set_audio(path); self.spectrogram_widget.set_audio(path); self.visualizer_widget.set_audio(path); self.update_audio_stats(path)
        ml = metadata['metadata']; hd = self.settings.value("history", [])
        row = self.playlist_table.rowCount()
        for i, m in enumerate(ml):
            hd.append(m); self.playlist_table.insertRow(row+i)
            pi = QTableWidgetItem(m['prompt']); pi.setToolTip(m['path'])
            if m['path'] and not os.path.exists(m['path']): pi.setForeground(QColor('red'))
            self.playlist_table.setItem(row+i, 0, pi)
            mi = MiniWaveformItem(m['path'], m['duration']); self.playlist_table.setItem(row+i, 1, mi)
            if self._thumb_worker.isRunning(): self._thumb_worker.enqueue(row+i, m['path'])
            self.playlist_table.setItem(row+i, 2, QTableWidgetItem(m['timestamp'][:10]))
            self.playlist_table.setItem(row+i, 3, QTableWidgetItem(str(m['seed'])))
            self.playlist_table.setItem(row+i, 4, QTableWidgetItem(m['model'].split('/')[-1]))
        if len(hd) > MAX_HISTORY:
            hd = hd[-MAX_HISTORY:]
            while self.playlist_table.rowCount() > MAX_HISTORY: self.playlist_table.removeRow(0)
        self.settings.setValue("history", hd); self.settings.sync(); self.settings.setValue("last_audio_path", path); self.settings.sync()
        self.right_tabs.setCurrentIndex(0); self.status_bar.showMessage(f"Generated {len(self.current_audio_paths)} file(s) in {metadata.get('gen_time',0):.2f}s", 5000)
        if getattr(self, '_asmr_auto_export', False): self._asmr_check_auto_export(path, metadata)
    def generation_error(self, msg):
        self.progress_bar.setVisible(False); self.eta_label.setText(""); self.status_label.setText("Error")
        self.status_led.setStyleSheet("background-color:#c0392b;border-radius:6px;margin:2px;")
        if self.batch_state == BatchState.RUNNING:
            self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {msg[:100]}")
            if not self.batch_continue_on_error_check.isChecked(): self.batch_state = BatchState.CANCELLED
        else: QMessageBox.critical(self, "Error", msg)
    def _trigger_next_batch_item(self):
        if self.batch_state == BatchState.CANCELLED: return
        if not self.start_generation(): self.on_worker_thread_finished()
    def on_worker_thread_finished(self):
        self.generate_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        if self.batch_state == BatchState.RUNNING:
            self.batch_index += 1; self.save_batch_state()
            if self.batch_index < len(self.batch_prompts):
                np_ = self.batch_prompts[self.batch_index]; self.prompt_input.setPlainText(np_)
                self.batch_progress_label.setText(f"Batch: {self.batch_index+1} / {len(self.batch_prompts)}")
                self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Generated: {np_[:50]}...")
                QTimer.singleShot(int(self.batch_delay_spin.value()*1000), self._trigger_next_batch_item)
            else:
                self.finalize_batch_state(); self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Batch complete!")
                if os.path.exists(BATCH_STATE_FILE):
                    try: os.remove(BATCH_STATE_FILE)
                    except: pass
                QMessageBox.information(self, "Complete", f"All {len(self.batch_prompts)} prompts processed!")
        else: self.finalize_batch_state(); self.status_label.setText("Ready")
    def on_variation_changed(self, idx):
        if 0 <= idx < len(self.current_audio_paths):
            self.current_audio_path = self.current_audio_paths[idx]
            self.waveform_widget.set_audio(self.current_audio_path); self.spectrogram_widget.set_audio(self.current_audio_path)
            self.visualizer_widget.set_audio(self.current_audio_path); self.update_audio_stats(self.current_audio_path)
            if self.media_player.playbackState() == QMediaPlayer.PlayingState: self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))
    def toggle_playback(self):
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox)): return
        if not self.current_audio_path or not os.path.exists(self.current_audio_path): return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.stop(); self.status_led.setStyleSheet("background-color:#555;border-radius:6px;margin:2px;")
        else:
            self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path)); self.media_player.play()
            self.status_led.setStyleSheet("background-color:#2ecc71;border-radius:6px;margin:2px;")
    def on_playback_state_changed(self, s):
        self.play_btn.setText("Stop" if s == QMediaPlayer.PlayingState else "Play")
        if s != QMediaPlayer.PlayingState: self.status_led.setStyleSheet("background-color:#555;border-radius:6px;margin:2px;"); self.vu_meter.setValue(0)
    def play_from_playlist(self, item):
        path = self.playlist_table.item(item.row(), 0).toolTip() if isinstance(item, QTableWidgetItem) else item; self._play_path(path)
    def _play_path(self, path):
        if path and os.path.exists(path): self._load_audio_file(path); self.toggle_playback()
        else: QMessageBox.warning(self, "Missing", "File moved or deleted.")
    def show_history_context_menu(self, pos):
        item = self.playlist_table.itemAt(pos)
        if not item: return
        path = self.playlist_table.item(item.row(), 0).toolTip()
        m = QMenu(self); m.addAction("Play", lambda: self._play_path(path)); m.addAction("Open Folder", lambda: self.open_file_in_folder(path))
        m.addAction("Copy Path", lambda: QApplication.clipboard().setText(path)); m.addAction("Reuse Settings", lambda: self.reuse_settings(item.row()))
        m.addSeparator(); m.addAction("Delete", lambda: self.delete_history_row(item.row())); m.exec(self.playlist_table.viewport().mapToGlobal(pos))
    def open_file_in_folder(self, path):
        if not path or not os.path.exists(path): return
        if sys.platform == 'win32': subprocess.Popen(['explorer', '/select,', path])
        elif sys.platform == 'darwin': subprocess.Popen(['open', '-R', path])
        else: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
    def reuse_settings(self, row):
        hd = self.settings.value("history", [])
        if 0 <= row < len(hd):
            e = hd[row]; self.prompt_input.setPlainText(e.get("prompt","")); self.neg_prompt_input.setText(e.get("negative_prompt",""))
            self.duration_spin.setValue(e.get("duration", 10)); self.steps_spin.setValue(e.get("steps", 200))
            self.guidance_spin.setValue(e.get("guidance", 3.5)); self.seed_spin.setValue(e.get("seed", -1))
            self.model_combo.setCurrentText(e.get("model", "cvssp/audioldm2"))
        self.mode_tabs.setCurrentIndex(0)
    def delete_history_row(self, row):
        self.playlist_table.removeRow(row); hd = self.settings.value("history", [])
        if 0 <= row < len(hd): hd.pop(row); self.settings.setValue("history", hd); self.settings.sync()
    def on_media_status(self, s):
        if s == QMediaPlayer.EndOfMedia: self.play_btn.setText("Play"); self.waveform_widget.set_playback_position(0); self.spectrogram_widget.set_playback_position(0); self.visualizer_widget.set_playback_position(0); self.vu_meter.setValue(0)
    def on_position_changed(self, pos):
        dur = self.media_player.duration()
        if dur > 0:
            self.time_label.setText(f"{format_time(pos/1000)} / {format_time(dur/1000)}")
            self.waveform_widget.set_playback_position(pos/dur); self.spectrogram_widget.set_playback_position(pos/dur); self.visualizer_widget.set_playback_position(pos/dur)
            self.vu_meter.setValue(int(min(100, self.waveform_widget.rms_at(pos/dur)*150)))
    def on_duration_changed(self, dur):
        if dur > 0: self.time_label.setText(f"00:00 / {format_time(dur/1000)}")
    def _seek_to(self, pos):
        if self.media_player.duration() > 0: self.media_player.setPosition(int(pos*self.media_player.duration()))
    def set_volume(self, v): self.audio_output.setVolume(v/100); self.vol_label.setText(f"{v}%")
    def save_audio(self):
        if not self.current_audio_path: return
        dn = os.path.basename(self.current_audio_path).replace('.wav', '')
        path, _ = QFileDialog.getSaveFileName(self, "Save", dn, "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac);;MP4 (*.mp4)")
        if path:
            if path.endswith('.mp4'): self.export_to_media(self.current_audio_path, path, 'mp4')
            elif path.endswith('.mp3'): self.export_to_media(self.current_audio_path, path, 'mp3')
            elif path.endswith('.flac'): self.export_to_media(self.current_audio_path, path, 'flac')
            else:
                try: shutil.copy2(self.current_audio_path, path)
                except Exception as e: QMessageBox.critical(self, "Error", f"Failed: {e}")
    def export_to_media(self, wav_path, out_path, fmt):
        if not shutil.which("ffmpeg") and not os.path.exists("ffmpeg.exe"): QMessageBox.critical(self, "FFmpeg", "Install FFmpeg."); return
        cmd = ["ffmpeg", "-y", "-i", wav_path]
        if fmt == 'mp4': cmd += ["-f","lavfi","-i","color=c=black:s=1280x720:r=30","-c:v","libx264","-c:a","aac","-shortest","-pix_fmt","yuv420p"]
        elif fmt == 'mp3': cmd += ["-c:a","libmp3lame","-b:a","320k"]
        elif fmt == 'flac': cmd += ["-c:a","flac"]
        cmd.append(out_path)
        try:
            cf = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            r = subprocess.run(cmd, capture_output=True, text=True, creationflags=cf)
            if r.returncode == 0: QMessageBox.information(self, "Done", f"Exported to:\n{out_path}")
            else: QMessageBox.critical(self, "Error", r.stderr[-500:])
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def open_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if path: self._load_audio_file(path)
    def paste_from_clipboard(self):
        t = QApplication.clipboard().text()
        if t: self.prompt_input.setPlainText(t)
    def select_batch_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Text", "", "Text (*.txt)")
        if path: self._load_batch_file(path)
    def start_batch(self):
        if not self.batch_prompts: QMessageBox.warning(self, "Error", "Select a text file first."); return
        if QMessageBox.question(self, "Batch", f"Generate {len(self.batch_prompts)} prompts?") != QMessageBox.Yes: return
        self.batch_state = BatchState.RUNNING; self.batch_index = 0; self.batch_start_btn.setEnabled(False); self.batch_file_btn.setEnabled(False)
        self.variations_spin.setValue(self.batch_variations_spin.value()); self.prompt_input.setPlainText(self.batch_prompts[0])
        self.batch_progress_label.setText(f"Batch: 1 / {len(self.batch_prompts)}"); self.batch_log.clear()
        self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Starting batch..."); self.save_batch_state(); self.mode_tabs.setCurrentIndex(0)
        if not self.start_generation(): self.finalize_batch_state()
    def save_batch_state(self):
        if self.batch_state == BatchState.RUNNING and self.batch_prompts:
            try:
                with open(BATCH_STATE_FILE, "w") as f: json.dump({"prompts": self.batch_prompts, "index": self.batch_index, "file": self.batch_file_label.text()}, f)
            except: pass
    def check_interrupted_batch(self):
        if os.path.exists(BATCH_STATE_FILE):
            if QMessageBox.question(self, "Resume", "Resume interrupted batch?") == QMessageBox.Yes:
                try:
                    with open(BATCH_STATE_FILE) as f: s = json.load(f)
                    self.batch_prompts = s["prompts"]; self.batch_index = s["index"]; self.batch_file_label.setText(s["file"])
                    self.batch_state = BatchState.RUNNING; self.batch_start_btn.setEnabled(False); self.batch_file_btn.setEnabled(False)
                    self.prompt_input.setPlainText(self.batch_prompts[self.batch_index])
                    self.batch_progress_label.setText(f"Batch: {self.batch_index+1} / {len(self.batch_prompts)}")
                    self.batch_log.clear(); self.batch_log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming..."); self.mode_tabs.setCurrentIndex(0)
                    if not self.start_generation(): self.finalize_batch_state()
                except: pass
            else:
                try: os.remove(BATCH_STATE_FILE)
                except: pass
    def finalize_batch_state(self):
        self.batch_start_btn.setEnabled(True); self.batch_file_btn.setEnabled(True)
        if self.batch_state == BatchState.RUNNING: self.batch_state = BatchState.IDLE; self.batch_progress_label.setText("Batch finished.")
        elif self.batch_state == BatchState.CANCELLED: self.batch_state = BatchState.IDLE
    def copy_settings_to_clipboard(self):
        d = {"prompt": self.prompt_input.toPlainText(), "model": self.model_combo.currentText(), "duration": self.duration_spin.value(), "steps": self.steps_spin.value(), "guidance": self.guidance_spin.value(), "seed": self.seed_spin.value()}
        QApplication.clipboard().setText(json.dumps(d, indent=2))
    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Name:")
        if not ok or not name.strip(): return
        p = {"prompt": self.prompt_input.toPlainText(), "model": self.model_combo.currentText(), "duration": self.duration_spin.value(), "steps": self.steps_spin.value(), "guidance": self.guidance_spin.value()}
        ps = self.settings.value("presets", {}); ps[name] = p; self.settings.setValue("presets", ps); self.settings.sync()
    def load_preset(self):
        ps = self.settings.value("presets", {})
        if not ps: return
        name, ok = QInputDialog.getItem(self, "Load Preset", "Select:", list(ps.keys()), 0, False)
        if not ok or not name: return
        p = ps[name]; self.prompt_input.setPlainText(p.get("prompt","")); self.model_combo.setCurrentText(p.get("model","cvssp/audioldm2"))
        self.duration_spin.setValue(p.get("duration", 10)); self.steps_spin.setValue(p.get("steps", 200)); self.guidance_spin.setValue(p.get("guidance", 3.5))
    def export_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Session", "session.json", "JSON (*.json)")
        if not path: return
        s = {"prompt": self.prompt_input.toPlainText(), "settings": {"model": self.model_combo.currentText(), "duration": self.duration_spin.value(), "steps": self.steps_spin.value()}, "current_audio": self.current_audio_path}
        try:
            with open(path, "w") as f: json.dump(s, f, indent=2)
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def import_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Session", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path) as f: s = json.load(f)
            self.prompt_input.setPlainText(s.get("prompt","")); st = s.get("settings",{})
            self.model_combo.setCurrentText(st.get("model","cvssp/audioldm2")); self.duration_spin.setValue(st.get("duration",10)); self.steps_spin.setValue(st.get("steps",200))
            if s.get("current_audio") and os.path.exists(s["current_audio"]): self._load_audio_file(s["current_audio"])
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def open_docs(self):
        QMessageBox.information(self, "Docs", f"{APP_NAME} v{APP_VERSION}\n\nASMR STUDIO:\nSelect preset → click Generate ASMR Video.\nAudio is generated, looped, processed, then rendered to MP4.\n\nVISUALIZERS:\n13 styles including Electric Sheep, Chaos Game, Rain, Fire, etc.\n\nLOOPING:\nGenerate 10min → loop to 8hrs.\n\nSHORTCUTS:\nCtrl+Enter: Generate | Space: Play/Stop | Esc: Cancel")
    def show_prompt_tips(self):
        QMessageBox.information(self, "Tips", "1. BE SPECIFIC\n2. DESCRIBE THE SCENE\n3. MENTION QUALITY\n4. USE NEGATIVE PROMPTS\n5. EXPERIMENT WITH GUIDANCE")
    def show_about(self):
        QMessageBox.about(self, "About", f"<h2>{APP_NAME}</h2><p>Version: {APP_VERSION}</p><p>ASMR video creation suite with 13 visualizers.</p>")
    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export History", "history.csv", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(['Prompt','Duration','Date','Seed','Model','Path'])
                for r in range(self.playlist_table.rowCount()):
                    w.writerow([self.playlist_table.item(r,0).text() if self.playlist_table.item(r,0) else "",
                                self.playlist_table.item(r,1).text() if self.playlist_table.item(r,1) else "",
                                self.playlist_table.item(r,2).text() if self.playlist_table.item(r,2) else "",
                                self.playlist_table.item(r,3).text() if self.playlist_table.item(r,3) else "",
                                self.playlist_table.item(r,4).text() if self.playlist_table.item(r,4) else "",
                                self.playlist_table.item(r,0).toolTip() if self.playlist_table.item(r,0) else ""])
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
    def _do_filter(self, text):
        tl = text.lower()
        for r in range(self.playlist_table.rowCount()):
            it = self.playlist_table.item(r, 0)
            if it: self.playlist_table.setRowHidden(r, tl not in it.text().lower())
    def load_history(self):
        hd = self.settings.value("history", [])
        if not hd: return
        if len(hd) > MAX_HISTORY: hd = hd[-MAX_HISTORY:]; self.settings.setValue("history", hd); self.settings.sync()
        self.playlist_table.setUpdatesEnabled(False); self.playlist_table.setRowCount(0); self.playlist_table.setRowCount(len(hd))
        try:
            for r, e in enumerate(hd):
                p = e.get("path",""); pi = QTableWidgetItem(e.get("prompt","")); pi.setToolTip(p)
                if p and not os.path.exists(p): pi.setForeground(QColor('red'))
                self.playlist_table.setItem(r, 0, pi)
                mi = MiniWaveformItem(p, e.get('duration', 0)); self.playlist_table.setItem(r, 1, mi)
                if self._thumb_worker.isRunning(): self._thumb_worker.enqueue(r, p)
                self.playlist_table.setItem(r, 2, QTableWidgetItem(e.get("timestamp","")[:10]))
                self.playlist_table.setItem(r, 3, QTableWidgetItem(str(e.get("seed",""))))
                self.playlist_table.setItem(r, 4, QTableWidgetItem(e.get("model","").split('/')[-1]))
        finally: self.playlist_table.setUpdatesEnabled(True)
    def clear_history(self):
        if QMessageBox.question(self, "Clear", "Remove all history?") == QMessageBox.Yes:
            self.playlist_table.setRowCount(0); self.settings.remove("history"); self.settings.sync()
    def load_settings(self):
        self.prompt_input.setPlainText(self.settings.value("prompt", "Musical constellations twinkling in the night sky", type=str))
        self.neg_prompt_input.setText(self.settings.value("negative_prompt", "", type=str))
        self.model_combo.setCurrentText(self.settings.value("model", "cvssp/audioldm2", type=str))
        self.duration_spin.setValue(float(self.settings.value("duration", 10.0)))
        self.chunk_size_spin.setValue(float(self.settings.value("chunk_size", 10.0)))
        self.steps_spin.setValue(int(self.settings.value("steps", 200)))
        self.guidance_spin.setValue(float(self.settings.value("guidance", 3.5)))
        self.seed_spin.setValue(int(self.settings.value("seed", -1)))
        self.variations_spin.setValue(int(self.settings.value("variations", 1)))
        dd = "cpu"
        if torch.cuda.is_available(): dd = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): dd = "mps"
        self.device_combo.setCurrentText(self.settings.value("device", dd, type=str))
        self.cache_dir_edit.setText(self.settings.value("cache_dir", DEFAULT_CACHE_DIR, type=str))
        self.output_dir_edit.setText(self.settings.value("output_dir", os.path.join(os.getcwd(), "generated_audio"), type=str))
        self.section_edit.setText(self.settings.value("section", "", type=str))
        for m in self.settings.value("custom_models", []):
            if m not in DEFAULT_MODELS and self.model_combo.findText(m) == -1: self.model_combo.addItem(m)
        self.normalize_check.setChecked(self.settings.value("normalize", True, type=bool))
        self.trim_check.setChecked(self.settings.value("trim", True, type=bool))
        self.fade_in_check.setChecked(self.settings.value("fade_in", True, type=bool))
        self.fade_out_check.setChecked(self.settings.value("fade_out", True, type=bool))
        self.cpu_offload_check.setChecked(self.settings.value("cpu_offload", False, type=bool))
        self.hp_spin.setValue(float(self.settings.value("high_pass_freq", 0.0)))
        self.lp_spin.setValue(float(self.settings.value("low_pass_freq", 0.0)))
        self.batch_variations_spin.setValue(int(self.settings.value("batch_variations", 1)))
        self.batch_delay_spin.setValue(float(self.settings.value("batch_delay", 1.0)))
        self.batch_continue_on_error_check.setChecked(self.settings.value("batch_continue_on_error", True, type=bool))
        self.vol_slider.setValue(int(self.settings.value("volume", 50)))
        self.mode_tabs.setCurrentIndex(int(self.settings.value("left_tab", 0)))
        self.right_tabs.setCurrentIndex(int(self.settings.value("right_tab", 0)))
        self.prompt_input.history = self.settings.value("prompt_history", [])
        self.on_device_changed(self.device_combo.currentText()); self.update_chunk_preview(); self.load_history()
        la = self.settings.value("last_audio_path", "", type=str)
        if la and os.path.exists(la): self._load_audio_file(la)
    def auto_save_state(self):
        self.settings.setValue("prompt", self.prompt_input.toPlainText())
        self.settings.setValue("steps", self.steps_spin.value()); self.settings.setValue("seed", self.seed_spin.value())
        self.settings.setValue("duration", self.duration_spin.value()); self.settings.sync()
    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            if self.isMinimized() and hasattr(self, 'gpu_mem_timer'): self.gpu_mem_timer.stop()
            elif hasattr(self, 'gpu_mem_timer'): self.gpu_mem_timer.start(GPU_MEM_POLL_MS)
        super().changeEvent(e)
    def closeEvent(self, e):
        if (self.video_worker and self.video_worker.isRunning()) or (self.worker_thread and self.worker_thread.isRunning()):
            if QMessageBox.question(self, "Minimize?", "Still rendering. Minimize to tray?") == QMessageBox.Yes:
                self.hide(); self._tray.notify("Still Rendering", "Your video is being rendered."); e.ignore(); return
        self.settings.setValue("geometry", self.saveGeometry()); self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("left_tab", self.mode_tabs.currentIndex()); self.settings.setValue("right_tab", self.right_tabs.currentIndex())
        for k, w in [("prompt", self.prompt_input.toPlainText()), ("negative_prompt", self.neg_prompt_input.text()),
            ("model", self.model_combo.currentText()), ("duration", self.duration_spin.value()),
            ("chunk_size", self.chunk_size_spin.value()), ("steps", self.steps_spin.value()),
            ("guidance", self.guidance_spin.value()), ("seed", self.seed_spin.value()),
            ("variations", self.variations_spin.value()), ("device", self.device_combo.currentText()),
            ("cache_dir", self.cache_dir_edit.text().strip()), ("output_dir", self.output_dir_edit.text().strip()),
            ("section", self.section_edit.text().strip()), ("normalize", self.normalize_check.isChecked()),
            ("trim", self.trim_check.isChecked()), ("fade_in", self.fade_in_check.isChecked()),
            ("fade_out", self.fade_out_check.isChecked()), ("cpu_offload", self.cpu_offload_check.isChecked()),
            ("high_pass_freq", self.hp_spin.value()), ("low_pass_freq", self.lp_spin.value()),
            ("batch_variations", self.batch_variations_spin.value()), ("batch_delay", self.batch_delay_spin.value()),
            ("batch_continue_on_error", self.batch_continue_on_error_check.isChecked()),
            ("volume", self.vol_slider.value()), ("prompt_history", self.prompt_input.history[-MAX_PROMPT_HISTORY:])]:
            self.settings.setValue(k, w)
        if self.current_audio_path and os.path.exists(self.current_audio_path): self.settings.setValue("last_audio_path", self.current_audio_path)
        else: self.settings.remove("last_audio_path")
        try: self.media_player.stop()
        except: pass
        if hasattr(self, 'autosave_timer'): self.autosave_timer.stop()
        if hasattr(self, 'gpu_mem_timer'): self.gpu_mem_timer.stop()
        if hasattr(self, '_thumb_worker'):
            try: self._thumb_worker.thumbReady.disconnect()
            except: pass
            self._thumb_worker.stop()
        if self.worker: self.worker.is_cancelled = True
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            if not self.worker_thread.wait(5000): self.worker_thread.terminate(); self.worker_thread.wait(1000)
        AudioGenerationWorker.unload_pipeline()
        if hasattr(self, '_audio_cache'): self._audio_cache.clear()
        gc.collect()
        if ML_AVAILABLE and torch.cuda.is_available():
            try: torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            except: pass
        self.settings.sync(); e.accept()

    # ── Video Studio methods ──
    def _video_browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if path: self.video_audio_path_edit.setText(path)
    def _video_use_current_audio(self):
        if self.current_audio_path and os.path.exists(self.current_audio_path): self.video_audio_path_edit.setText(self.current_audio_path)
        else: QMessageBox.warning(self, "No Audio", "No audio loaded.")
    def _video_browse_output(self):
        dn = "video_export.mp4"
        if self.video_audio_path_edit.text(): dn = os.path.splitext(os.path.basename(self.video_audio_path_edit.text()))[0] + "_video.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", dn, "MP4 (*.mp4)")
        if path: self.video_output_edit.setText(path)
    def _video_start_export(self):
        ap = self.video_audio_path_edit.text().strip()
        if not ap or not os.path.exists(ap): QMessageBox.warning(self, "No Audio", "Select audio."); return
        op = self.video_output_edit.text().strip()
        if not op: QMessageBox.warning(self, "No Output", "Specify output."); return
        if not op.endswith('.mp4'): op += '.mp4'
        rk = self.video_res_combo.currentText(); w, h = VIDEO_RESOLUTIONS[rk]; fps = self.video_fps_combo.currentData()
        style = self.video_style_combo.currentText(); theme = self.video_theme_combo.currentText()
        preset = self.video_preset_combo.currentText(); crf = self.video_crf_spin.value()
        ok, free = check_disk_space(op, estimate_video_file_size(3600, w, h, fps, crf))
        if not ok: QMessageBox.warning(self, "Disk Space", f"Insufficient disk space ({free:.1f} GB free)."); return
        self.video_export_btn.setEnabled(False); self.video_cancel_btn.setEnabled(True); self.video_progress_bar.setVisible(True); self.video_progress_bar.setValue(0)
        self.video_worker = VideoExportWorker(ap, op, w, h, fps, style, theme, preset, crf, self._audio_cache)
        self.video_worker.progress_updated.connect(self.video_progress_bar.setValue)
        self.video_worker.status_updated.connect(self.video_status_label.setText)
        self.video_worker.export_completed.connect(self._video_export_done)
        self.video_worker.error_occurred.connect(self._video_export_error)
        self.video_worker.start()
    def _video_cancel_export(self):
        if self.video_worker: self.video_worker.cancel()
    def _video_export_done(self, path):
        self.video_export_btn.setEnabled(True); self.video_cancel_btn.setEnabled(False); self.video_progress_bar.setVisible(False)
        self.video_status_label.setText(f"Done: {path}")
        self._tray.notify("Video Complete", f"Saved: {os.path.basename(path)}")
        QMessageBox.information(self, "Complete", f"Video exported:\n{path}")
    def _video_export_error(self, msg):
        self.video_export_btn.setEnabled(True); self.video_cancel_btn.setEnabled(False); self.video_progress_bar.setVisible(False)
        self.video_status_label.setText(f"Error: {msg[:200]}"); QMessageBox.critical(self, "Error", msg)

    # ── ASMR Studio methods ──
    def _asmr_preset_selected(self, name):
        if name not in ASMR_PRESETS: return
        p = ASMR_PRESETS[name]
        self.asmr_prompt_edit.setPlainText(p["prompt"]); self.asmr_neg_edit.setText(p["negative_prompt"])
        self.asmr_duration_spin.setValue(p["duration"]); self.asmr_steps_spin.setValue(p["steps"])
        self.asmr_guidance_spin.setValue(p["guidance"]); self.asmr_chunk_spin.setValue(p["chunk_size"])
        self.asmr_visual_combo.setCurrentText(p["visual_style"]); self.asmr_theme_combo.setCurrentText(p["color_theme"])
        self.asmr_motion_slider.setValue(int(p["motion_speed"]*100)); self.asmr_soft_slider.setValue(int(p["softness"]*100))
        self.asmr_encoding_combo.setCurrentText(p["encoding"])
        self.asmr_loop_check.setChecked(p.get("loop", True)); self.asmr_target_duration_spin.setValue(p.get("target_duration", 3600))
        self.asmr_crossfade_spin.setValue(p.get("crossfade", 3.0)); self.asmr_process_check.setChecked(p.get("process_asmr", True))
        self.asmr_lp_spin.setValue(p.get("lowpass_hz", 8000)); self.asmr_nf_spin.setValue(p.get("noise_floor_db", -50))
        self._asmr_update_estimate()
    def _asmr_browse_output(self):
        dn = "asmr_video.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save ASMR Video", dn, "MP4 (*.mp4)")
        if path: self.asmr_output_edit.setText(path)
    def _asmr_update_estimate(self):
        dur = self.asmr_target_duration_spin.value() if self.asmr_loop_check.isChecked() else self.asmr_duration_spin.value()
        ek = self.asmr_encoding_combo.currentText()
        if ek not in YOUTUBE_ENCODING_PRESETS: return
        enc = YOUTUBE_ENCODING_PRESETS[ek]; w, h = VIDEO_RESOLUTIONS[enc["resolution"]]; fps = enc["fps"]
        tf = int(dur * fps); tpf = 0.015 if w <= 1920 else 0.050; erm = tf * tpf / 60
        sz = estimate_video_file_size(dur, w, h, fps, enc["crf"])
        h_, m_ = int(dur//3600), int((dur%3600)//60)
        self.asmr_estimate_label.setText(f"Video: {w}x{h} @ {fps}fps | Duration: {h_}h {m_}m | Frames: {tf:,} | Est. render: ~{erm:.0f} min | Est. size: ~{sz:.1f} GB")
    def _asmr_one_click_generate(self):
        prompt = self.asmr_prompt_edit.toPlainText().strip()
        if not prompt: QMessageBox.warning(self, "No Prompt", "Enter a prompt."); return
        op = self.asmr_output_edit.text().strip()
        if not op: QMessageBox.warning(self, "No Output", "Specify output."); return
        if not op.endswith('.mp4'): op += '.mp4'
        model = self.asmr_model_combo.currentText().strip()
        if '/' not in model: QMessageBox.warning(self, "Invalid", "Model must be 'org/model'."); return
        out_dir = self.output_dir_edit.text().strip() or os.path.join(os.getcwd(), "generated_audio")
        os.makedirs(out_dir, exist_ok=True); cache_dir = self.cache_dir_edit.text().strip() or DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        ek = self.asmr_encoding_combo.currentText(); enc = YOUTUBE_ENCODING_PRESETS[ek]; w, h = VIDEO_RESOLUTIONS[enc["resolution"]]
        self._asmr_pending = {'output_path': op, 'width': w, 'height': h, 'fps': enc["fps"],
            'style': self.asmr_visual_combo.currentText(), 'theme': self.asmr_theme_combo.currentText(),
            'motion_speed': self.asmr_motion_slider.value()/100, 'softness': self.asmr_soft_slider.value()/100,
            'preset': enc["preset"], 'crf': enc["crf"], 'audio_bitrate': enc["audio_bitrate"],
            'extra_args': enc.get("extra_args", []), 'prompt': prompt, 'duration': self.asmr_duration_spin.value(),
            'visual_style': self.asmr_visual_combo.currentText(), 'color_theme': self.asmr_theme_combo.currentText(),
            'encoding_preset': ek, 'loop': self.asmr_loop_check.isChecked(),
            'target_duration': self.asmr_target_duration_spin.value() if self.asmr_loop_check.isChecked() else self.asmr_duration_spin.value(),
            'crossfade': self.asmr_crossfade_spin.value(), 'process_asmr': self.asmr_process_check.isChecked(),
            'lowpass_hz': self.asmr_lp_spin.value(), 'noise_floor_db': self.asmr_nf_spin.value()}
        self.mode_tabs.setCurrentIndex(0); self.prompt_input.setPlainText(prompt)
        self.neg_prompt_input.setText(self.asmr_neg_edit.text()); self.model_combo.setCurrentText(model)
        self.duration_spin.setValue(self.asmr_duration_spin.value()); self.chunk_size_spin.setValue(self.asmr_chunk_spin.value())
        self.steps_spin.setValue(self.asmr_steps_spin.value()); self.guidance_spin.setValue(self.asmr_guidance_spin.value())
        self.seed_spin.setValue(self.asmr_seed_spin.value()); self.section_edit.setText("ASMR")
        self.viz_style_combo.setCurrentText(self.asmr_visual_combo.currentText()); self.viz_theme_combo.setCurrentText(self.asmr_theme_combo.currentText())
        self.visualizer_widget.set_motion_speed(self.asmr_motion_slider.value()/100)
        self.visualizer_widget.set_softness(self.asmr_soft_slider.value()/100)
        if self.start_generation():
            self._asmr_auto_export = True; self.asmr_progress_bar.setVisible(True); self.asmr_progress_bar.setValue(0)
            self.asmr_status_label.setText("Step 1/2: Generating audio...")
    def _asmr_check_auto_export(self, path, metadata):
        if not getattr(self, '_asmr_auto_export', False): return
        self._asmr_auto_export = False; s = self._asmr_pending
        # Loop audio if requested
        if s.get('loop', False) and s.get('target_duration', 0) > s.get('duration', 0):
            self.asmr_status_label.setText(f"Looping audio to {format_time(s['target_duration'])}...")
            QApplication.processEvents()
            cached = self._audio_cache.get(path)
            if cached:
                sr, audio = cached
            else:
                l = AudioDataCache.load_from_disk(path)
                if l: sr, audio = l
                else: self._asmr_export_error("Failed to load audio for looping."); return
            audio = create_seamless_loop(audio, sr, s['target_duration'], s.get('crossfade', 3.0))
            if s.get('process_asmr', False):
                self.asmr_status_label.setText("Applying ASMR processing...")
                QApplication.processEvents()
                audio = process_asmr_audio(audio, sr, s.get('lowpass_hz', 8000), noise_floor_db=s.get('noise_floor_db', -50))
            looped_path = path.replace('.wav', '_looped.wav')
            wavfile.write(looped_path, sr, (np.clip(audio, -1, 1)*32767).astype(np.int16))
            path = looped_path
            self._audio_cache.invalidate(path); self._audio_cache.put(path, sr, audio)
        self.asmr_status_label.setText("Step 2/2: Rendering video...")
        meta_path = self._generate_youtube_metadata(s['prompt'], s.get('target_duration', s['duration']), s['visual_style'], s['color_theme'], s['encoding_preset'], s['output_path'])
        self.video_worker = VideoExportWorker(path, s['output_path'], s['width'], s['height'], s['fps'], s['style'], s['theme'], s['preset'], s['crf'], self._audio_cache, None, s['motion_speed'], s['softness'], s['audio_bitrate'], s['extra_args'])
        self.video_worker.progress_updated.connect(self.asmr_progress_bar.setValue)
        self.video_worker.status_updated.connect(lambda m: self.asmr_status_label.setText(m))
        self.video_worker.thumbnail_saved.connect(lambda p: self.asmr_status_label.setText(self.asmr_status_label.text() + f"\nThumbnail: {p}"))
        self.video_worker.export_completed.connect(lambda p: self._asmr_export_complete(p, meta_path))
        self.video_worker.error_occurred.connect(self._asmr_export_error)
        self.video_worker.start()
    def _asmr_export_complete(self, video_path, meta_path):
        self.asmr_progress_bar.setValue(100)
        msg = f"Video: {video_path}"
        if meta_path: msg += f"\nMetadata: {meta_path}"
        self.asmr_status_label.setText(msg)
        self._tray.notify("ASMR Video Complete!", f"Saved: {os.path.basename(video_path)}")
        if hasattr(self, '_asmr_queue') and self._asmr_queue.running: self._asmr_batch_item_done()
        QMessageBox.information(self, "Complete", f"Video exported:\n{video_path}\n\nMetadata: {meta_path or 'N/A'}")
    def _asmr_export_error(self, msg):
        self.asmr_progress_bar.setVisible(False); self.asmr_status_label.setText(f"Error: {msg[:200]}")
        QMessageBox.critical(self, "Error", msg)
    def _generate_youtube_metadata(self, prompt, dur, style, theme, enc_key, out_path):
        h = int(dur//3600); m = int((dur%3600)//60)
        kw = [w.strip(",.?!") for w in prompt.split() if len(w) > 4][:6]
        title = " ".join(kw).title()
        if h > 0: title += f" | {h} Hour{'s' if h!=1 else ''} ASMR Sleep Sounds"
        elif m > 0: title += f" | {m} Min ASMR Sleep Sounds"
        enc = YOUTUBE_ENCODING_PRESETS.get(enc_key, {}); res = enc.get("resolution", "1080p"); fps = enc.get("fps", 30)
        chapters = generate_youtube_chapters(dur, 900 if dur > 3600 else 300)
        desc = f"{title}\n\nASMR Sleep Sounds for Relaxation & Sleep\n\nAudio prompt: \"{prompt}\"\nVisualizer: {style}\nDuration: {format_time(dur)}\nResolution: {res} @ {fps}fps\n\nBest experienced with headphones.\n\nTimestamps:\n" + "\n".join(chapters) + f"\n\n#ASMR #SleepSounds #Relaxation #{style.replace(' ','')}\n\nGenerated by {APP_NAME} v{APP_VERSION}"
        tags = ["ASMR","sleep","relaxation","sleep sounds","ASMR sleep","white noise","ambient","meditation","calm","soothing",style.lower().replace(" ",""),"AI ASMR"]
        mp = out_path.replace('.mp4', '_youtube_metadata.txt')
        try:
            with open(mp, 'w', encoding='utf-8') as f:
                f.write(f"TITLE:\n{title}\n\nDESCRIPTION:\n{desc}\n\nTAGS:\n{', '.join(tags)}\n\nCATEGORY: Music\nPRIVACY: Unlisted")
            return mp
        except: return None
    def _open_sound_mixer(self):
        dlg = SoundLayeringDialog(self, self._audio_cache); dlg.mixed_audio_ready.connect(self._load_audio_file); dlg.exec()
    def _asmr_add_to_queue(self):
        name = self.asmr_preset_combo.currentText()
        ov = {'prompt': self.asmr_prompt_edit.toPlainText().strip(), 'duration': self.asmr_duration_spin.value(),
              'visual_style': self.asmr_visual_combo.currentText(), 'color_theme': self.asmr_theme_combo.currentText(),
              'motion_speed': self.asmr_motion_slider.value()/100, 'softness': self.asmr_soft_slider.value()/100,
              'encoding': self.asmr_encoding_combo.currentText(), 'target_duration': self.asmr_target_duration_spin.value()}
        self._asmr_queue.add(name, ov); self._asmr_refresh_queue()
    def _asmr_remove_from_queue(self):
        r = self.asmr_batch_list.currentRow()
        if r >= 0: self._asmr_queue.remove(r); self._asmr_refresh_queue()
    def _asmr_clear_queue(self):
        self._asmr_queue.clear(); self._asmr_refresh_queue()
    def _asmr_refresh_queue(self):
        self.asmr_batch_list.clear()
        for i, item in enumerate(self._asmr_queue.items):
            st = ">" if i == self._asmr_queue.idx else "  "
            self.asmr_batch_list.addItem(f"{st} {i+1}. {item.get('name','')} ({format_time(item.get('target_duration', item.get('duration', 0)))})")
    def _asmr_start_batch(self):
        if self._asmr_queue.is_done(): QMessageBox.information(self, "Empty", "Add items first."); return
        item = self._asmr_queue.get_current()
        if not item: return
        self.asmr_preset_combo.setCurrentText(item.get('name', '')); self.asmr_prompt_edit.setPlainText(item.get('prompt', ''))
        self.asmr_neg_edit.setText(item.get('negative_prompt', '')); self.asmr_duration_spin.setValue(item.get('duration', 600))
        self.asmr_visual_combo.setCurrentText(item.get('visual_style', 'Rain Particles')); self.asmr_theme_combo.setCurrentText(item.get('color_theme', 'Ocean'))
        self.asmr_motion_slider.setValue(int(item.get('motion_speed', 0.3)*100)); self.asmr_soft_slider.setValue(int(item.get('softness', 0.6)*100))
        self.asmr_encoding_combo.setCurrentText(item.get('encoding', 'ASMR 1080p 24fps (Calm)'))
        self.asmr_loop_check.setChecked(item.get('loop', True)); self.asmr_target_duration_spin.setValue(item.get('target_duration', 3600))
        self._asmr_queue.running = True; self._asmr_one_click_generate()
    def _asmr_batch_item_done(self):
        if not hasattr(self, '_asmr_queue') or not self._asmr_queue.running: return
        self._asmr_queue.advance(); self._asmr_refresh_queue()
        if self._asmr_queue.is_done():
            self._asmr_queue.running = False; self._asmr_queue.idx = 0; self._asmr_queue.save()
            QMessageBox.information(self, "Batch Complete", f"All {len(self._asmr_queue.items)} ASMR videos processed!")
        else: QTimer.singleShot(2000, self._asmr_start_batch)
    def _asmr_preview_render(self):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "No Audio", "Generate or load audio first."); return
        pp = os.path.join(tempfile.gettempdir(), "asmr_preview.mp4")
        self.asmr_status_label.setText("Rendering 10s preview at 480p...")
        self._preview_worker = VideoExportWorker(self.current_audio_path, pp, 854, 480, 24, self.asmr_visual_combo.currentText(), self.asmr_theme_combo.currentText(), "ultrafast", 25, self._audio_cache, None, self.asmr_motion_slider.value()/100, self.asmr_soft_slider.value()/100)
        self._preview_worker.progress_updated.connect(self.asmr_progress_bar.setValue)
        self._preview_worker.status_updated.connect(self.asmr_status_label.setText)
        self._preview_worker.export_completed.connect(lambda p: self._load_audio_file(p) or self.toggle_playback() or QMessageBox.information(self, "Preview", f"Preview: {p}"))
        self._preview_worker.error_occurred.connect(self._asmr_export_error)
        self._preview_worker.start()

# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion"); app.setApplicationName(APP_NAME); app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(SETTINGS_ORG)
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(26, 26, 46)); pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor(13, 27, 42)); pal.setColor(QPalette.AlternateBase, QColor(22, 33, 62))
    pal.setColor(QPalette.ToolTipBase, QColor(22, 33, 62)); pal.setColor(QPalette.ToolTipText, Qt.white)
    pal.setColor(QPalette.Text, Qt.white); pal.setColor(QPalette.Button, QColor(15, 52, 96))
    pal.setColor(QPalette.ButtonText, Qt.white); pal.setColor(QPalette.BrightText, Qt.red)
    pal.setColor(QPalette.Link, QColor(233, 69, 96)); pal.setColor(QPalette.Highlight, QColor(233, 69, 96))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    app.setPalette(pal); app.setStyleSheet(STYLESHEET); app.setFont(QFont("Segoe UI", 9))
    window = AudioLDM2Studio(); window.show()
    sys.exit(app.exec())