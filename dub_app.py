#!/usr/bin/env python3
# dub_app.py - drag-and-drop GUI for the dubbing tool (wraps dub.py --jobs).
# Modern Sun Valley theme, queue with per-video cover boxes, persistence,
# folder drag, single-instance guard, tooltips, non-blocking preview.
import os
import re
import sys
import glob
import json
import time
import queue
import socket
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from self_learning_agent import optimize_dub_job_config

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = True
except Exception:
    _DND = False
try:
    import sv_ttk
    _THEMED = True
except Exception:
    _THEMED = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "dubbed")
WORK = os.path.join(BASE, "_audio_work")
SETTINGS = os.path.join(BASE, "_dubapp_settings.json")
JOBS = os.path.join(WORK, "_jobs.json")
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".ts")
PREVIEW_MAX = 320
_NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_ICON = {"pending": "•", "working": "▶", "done": "✓", "error": "✗"}
_LOCK_PORT = 47823

# Free Microsoft (edge-tts) neural voices offered in the picker, by language code.
# Pick any voice per language; the ▶ button previews it. Add more rows freely -
# any valid edge-tts voice id works (run `edge-tts --list-voices` to see them all).
VOICE_CHOICES = {
    "vi": [
        {"id": "vi-VN-HoaiMyNeural", "label": "HoaiMy — Female"},
        {"id": "vi-VN-NamMinhNeural", "label": "NamMinh — Male"},
    ],
    "id": [
        {"id": "id-ID-GadisNeural", "label": "Gadis — Female"},
        {"id": "id-ID-ArdiNeural", "label": "Ardi — Male"},
    ],
    "ms": [
        {"id": "ms-MY-YasminNeural", "label": "Yasmin — Female"},
        {"id": "ms-MY-OsmanNeural", "label": "Osman — Male"},
    ],
    "es": [
        {"id": "es-MX-DaliaNeural", "label": "Dalia — Female (Mexico)"},
        {"id": "es-MX-JorgeNeural", "label": "Jorge — Male (Mexico)"},
        {"id": "es-ES-ElviraNeural", "label": "Elvira — Female (Spain)"},
        {"id": "es-ES-AlvaroNeural", "label": "Alvaro — Male (Spain)"},
        {"id": "es-AR-ElenaNeural", "label": "Elena — Female (Argentina)"},
        {"id": "es-CO-SalomeNeural", "label": "Salome — Female (Colombia)"},
    ],
    "en": [
        {"id": "en-US-AriaNeural", "label": "Aria — Female (US)"},
        {"id": "en-US-JennyNeural", "label": "Jenny — Female (US)"},
        {"id": "en-US-GuyNeural", "label": "Guy — Male (US)"},
        {"id": "en-US-ChristopherNeural", "label": "Christopher — Male (US)"},
        {"id": "en-GB-SoniaNeural", "label": "Sonia — Female (UK)"},
        {"id": "en-GB-RyanNeural", "label": "Ryan — Male (UK)"},
        {"id": "en-AU-NatashaNeural", "label": "Natasha — Female (Australia)"},
    ],
    # --- the rest of the XTTS v2 languages (one female + one male each) ---
    "fr": [
        {"id": "fr-FR-DeniseNeural", "label": "Denise — Female"},
        {"id": "fr-FR-HenriNeural", "label": "Henri — Male"},
    ],
    "de": [
        {"id": "de-DE-KatjaNeural", "label": "Katja — Female"},
        {"id": "de-DE-ConradNeural", "label": "Conrad — Male"},
    ],
    "it": [
        {"id": "it-IT-ElsaNeural", "label": "Elsa — Female"},
        {"id": "it-IT-DiegoNeural", "label": "Diego — Male"},
    ],
    "pt": [
        {"id": "pt-BR-FranciscaNeural", "label": "Francisca — Female (Brazil)"},
        {"id": "pt-BR-AntonioNeural", "label": "Antonio — Male (Brazil)"},
    ],
    "pl": [
        {"id": "pl-PL-ZofiaNeural", "label": "Zofia — Female"},
        {"id": "pl-PL-MarekNeural", "label": "Marek — Male"},
    ],
    "tr": [
        {"id": "tr-TR-EmelNeural", "label": "Emel — Female"},
        {"id": "tr-TR-AhmetNeural", "label": "Ahmet — Male"},
    ],
    "ru": [
        {"id": "ru-RU-SvetlanaNeural", "label": "Svetlana — Female"},
        {"id": "ru-RU-DmitryNeural", "label": "Dmitry — Male"},
    ],
    "nl": [
        {"id": "nl-NL-ColetteNeural", "label": "Colette — Female"},
        {"id": "nl-NL-MaartenNeural", "label": "Maarten — Male"},
    ],
    "cs": [
        {"id": "cs-CZ-VlastaNeural", "label": "Vlasta — Female"},
        {"id": "cs-CZ-AntoninNeural", "label": "Antonin — Male"},
    ],
    "ar": [
        {"id": "ar-EG-SalmaNeural", "label": "Salma — Female (Egypt)"},
        {"id": "ar-EG-ShakirNeural", "label": "Shakir — Male (Egypt)"},
    ],
    "zh-CN": [
        {"id": "zh-CN-XiaoxiaoNeural", "label": "Xiaoxiao — Female"},
        {"id": "zh-CN-YunxiNeural", "label": "Yunxi — Male"},
    ],
    "hu": [
        {"id": "hu-HU-NoemiNeural", "label": "Noemi — Female"},
        {"id": "hu-HU-TamasNeural", "label": "Tamas — Male"},
    ],
    "ko": [
        {"id": "ko-KR-SunHiNeural", "label": "SunHi — Female"},
        {"id": "ko-KR-InJoonNeural", "label": "InJoon — Male"},
    ],
    "ja": [
        {"id": "ja-JP-NanamiNeural", "label": "Nanami — Female"},
        {"id": "ja-JP-KeitaNeural", "label": "Keita — Male"},
    ],
    "hi": [
        {"id": "hi-IN-SwaraNeural", "label": "Swara — Female"},
        {"id": "hi-IN-MadhurNeural", "label": "Madhur — Male"},
    ],
}

# The voice each language starts on (matches dub.py's built-in defaults).
DEFAULT_VOICE = {
    "vi": "vi-VN-HoaiMyNeural", "id": "id-ID-GadisNeural", "ms": "ms-MY-YasminNeural",
    "es": "es-MX-DaliaNeural", "en": "en-US-AriaNeural", "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural", "it": "it-IT-ElsaNeural", "pt": "pt-BR-FranciscaNeural",
    "pl": "pl-PL-ZofiaNeural", "tr": "tr-TR-EmelNeural", "ru": "ru-RU-SvetlanaNeural",
    "nl": "nl-NL-ColetteNeural", "cs": "cs-CZ-VlastaNeural", "ar": "ar-EG-SalmaNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural", "hu": "hu-HU-NoemiNeural", "ko": "ko-KR-SunHiNeural",
    "ja": "ja-JP-NanamiNeural", "hi": "hi-IN-SwaraNeural",
}

# Short line spoken when the user clicks ▶ to preview a voice.
SAMPLE_TEXT = {
    "vi": "Xin chào, đây là giọng đọc thử.",
    "id": "Halo, ini adalah contoh suara.",
    "ms": "Helo, ini ialah contoh suara.",
    "es": "Hola, esta es una muestra de voz.",
    "en": "Hello, this is a sample of this voice.",
    "fr": "Bonjour, ceci est un échantillon de voix.",
    "de": "Hallo, dies ist eine Sprachprobe.",
    "it": "Ciao, questo è un campione vocale.",
    "pt": "Olá, esta é uma amostra de voz.",
    "pl": "Cześć, to jest próbka głosu.",
    "tr": "Merhaba, bu bir ses örneğidir.",
    "ru": "Здравствуйте, это образец голоса.",
    "nl": "Hallo, dit is een spraakvoorbeeld.",
    "cs": "Ahoj, toto je ukázka hlasu.",
    "ar": "مرحبًا، هذه عينة صوتية.",
    "zh-CN": "你好，这是一段语音示例。",
    "hu": "Helló, ez egy hangminta.",
    "ko": "안녕하세요, 이것은 음성 샘플입니다.",
    "ja": "こんにちは、これは音声のサンプルです。",
    "hi": "नमस्ते, यह एक आवाज़ का नमूना है।",
}

# Order shown in the picker. The first three are edge-tts only; the rest are the
# XTTS v2 languages (natural local voice + cloning). (code, display name).
LANG_ORDER = [
    ("vi", "Vietnamese"), ("id", "Indonesian"), ("ms", "Malay"),
    ("en", "English"), ("es", "Spanish"), ("fr", "French"),
    ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"),
    ("pl", "Polish"), ("tr", "Turkish"), ("ru", "Russian"),
    ("nl", "Dutch"), ("cs", "Czech"), ("ar", "Arabic"),
    ("zh-CN", "Chinese"), ("hu", "Hungarian"), ("ko", "Korean"),
    ("ja", "Japanese"), ("hi", "Hindi"),
]
LANG_DEFAULT_ON = {"vi", "es"}   # ticked on first run / when no saved setting

# DeepSeek (same endpoint/model the dubber uses) for the in-app AI assistant.
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"
ASSISTANT_TASKS = ["Correct original script", "Improve translation",
                   "Back-translation check", "Custom instruction"]

# --- Apple-inspired palette (calm light-gray canvas, single blue accent) ---------
APPLE = {
    "bg": "#F5F5F7",          # window canvas (Apple light gray)
    "panel": "#FFFFFF",       # input surfaces (lists, log, fields)
    "text": "#1D1D1F",        # primary near-black
    "muted": "#86868B",       # secondary gray (captions, section headers)
    "border": "#D2D2D7",      # hairline
    "accent": "#0071E3",      # Apple blue (primary action)
    "accent_hover": "#0A84FF",
    "accent_press": "#0058B0",
    "ok": "#34C759",          # green
    "warn": "#FF9F0A",        # amber
    "err": "#FF3B30",         # red
    "drop": "#FBFBFD",        # hero drop-zone fill
}


def find_ffmpeg() -> str:
    hits = glob.glob(os.path.join(BASE, "ffmpeg*", "bin", "ffmpeg.exe"))
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join(BASE, "**", "ffmpeg.exe"), recursive=True)
    return hits[0] if hits else "ffmpeg"


def grab_single_instance_lock():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", _LOCK_PORT))
        s.listen(1)
        return s
    except OSError:
        return None


class _Tooltip:
    """Lightweight hover tooltip."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg="#ffffd9", fg="#222",
                 relief="solid", borderwidth=1, font=("Segoe UI", 8),
                 padx=6, pady=3, justify="left", wraplength=320).pack()

    def _hide(self, _e):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class DubApp:
    def __init__(self, root):
        self.root = root
        self.queue: list[dict] = []
        self.sel = None
        self.ffmpeg = find_ffmpeg()
        self.ffprobe = os.path.join(os.path.dirname(self.ffmpeg), "ffprobe.exe")
        self.q: queue.Queue = queue.Queue()
        self.running = False
        self.cancel = False
        self.proc = None
        self.dur = 0.0
        self.tkimg = None
        self.stage_start = 0.0
        self.error_lines: list[str] = []
        self._preview_token = 0
        # A/B preview (Original vs dubbed audio, played through bundled ffplay)
        self.ffplay = os.path.join(os.path.dirname(self.ffmpeg), "ffplay.exe")
        self._ab_map: dict = {}        # dropdown label -> (source_path, dub_path)
        self._ab_proc = None
        self._ab_side = None
        self._ab_t0 = 0.0
        self._ab_pos0 = 0.0
        self.s = self._load_settings()
        root.title("Douyin Dubbing")
        geom = self.s.get("geom")
        root.geometry(geom if geom else "780x900")
        root.minsize(720, 760)          # keep the Dub All button on-screen
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._restore_queue()
        self.root.after(120, self._pump)

    # ---------- settings ----------
    def _load_settings(self) -> dict:
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                data = json.load(f)
                data.pop("deepseek_key", None)
                return data
        except Exception:
            return {}

    def _save_settings(self):
        data = {"lang": self._cur_code(),
                "cover": self.cover.get(),
                "band": self.band.get(),
                "keepmusic": self.keepmusic.get(), "model": self.model.get(),
                "fb": self.fb.get(), "srt": self.srt.get(), "burn": self.burn.get(),
                "tone": self.tone.get(), "audioonly": self.audioonly.get(),
                "clone": self.clone.get(),
                "gender": self.gender.get(),
                "speakers": self.speakers.get(),
                "speaker_override": {
                    "enforce_single_speaker": self.single_speaker.get(),
                    "primary_speaker_id": "Speaker_1",
                    "diarization_sensitivity": 0.0 if self.single_speaker.get() else 0.5,
                },
                "scenefx": self.scenefx.get(),
                "reuse_analysis": self.reuse_analysis.get(),
                "length_fit": self.length_fit.get(),
                "multimodal": self.multimodal.get(),
                "multimodal_vision": self.multimodal_vision.get(),
                "multimodal_fps": 1.0,
                "preset": "facebook_reels" if self.reels_auto.get() else "",
                "rights_mode": "owned_or_licensed" if self.rights_owned.get() else "",
                "voices": {lg: self._voice_id(lg) for lg in VOICE_CHOICES},
                "geom": self.root.geometry(),
                "queue": [{"path": it["path"],
                           "region": list(it["region"]) if it["region"] else None}
                          for it in self.queue]}
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _on_close(self):
        if self.running and not messagebox.askyesno(
                "Quit?", "A batch is still running. Stop it and quit?"):
            return
        self._ab_stop(reset_status=False)
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        try:
            self._save_settings()
        finally:
            self.root.destroy()

    def _restore_queue(self):
        for it in self.s.get("queue", []):
            p = it.get("path")
            if p and os.path.exists(p):
                self.queue.append({
                    "path": p,
                    "region": tuple(it["region"]) if it.get("region") else None,
                    "box": None, "status": "pending"})
        self._refresh_queue()
        if self.queue:
            self.qlist.selection_set(0)
            self._on_select(None)

    # ---------- Apple-style theme ----------
    def _apply_apple_style(self):
        """Layer a calm, high-contrast Apple-like look over the base theme:
        SF-style typography, a unified light-gray canvas, muted section headers and
        a single blue accent. Falls back gracefully when a font isn't installed."""
        import tkinter.font as tkfont
        fams = set(tkfont.families())
        pick = lambda cands, d: next((c for c in cands if c in fams), d)
        self.ui = pick(["SF Pro Display", "SF Pro Text", "Helvetica Neue",
                        "Segoe UI Variable Text", "Segoe UI"], "Segoe UI")
        self.mono = pick(["SF Mono", "Cascadia Code", "Cascadia Mono",
                          "JetBrains Mono", "Consolas"], "Consolas")
        self.AP = APPLE
        A = APPLE
        self.root.configure(bg=A["bg"])
        st = ttk.Style()
        st.configure(".", background=A["bg"], foreground=A["text"], font=(self.ui, 10))
        st.configure("TFrame", background=A["bg"])
        st.configure("TLabel", background=A["bg"], foreground=A["text"], font=(self.ui, 10))
        st.configure("TCheckbutton", background=A["bg"], foreground=A["text"], font=(self.ui, 10))
        st.configure("TRadiobutton", background=A["bg"], foreground=A["text"], font=(self.ui, 10))
        st.configure("TLabelframe", background=A["bg"], borderwidth=0, relief="flat")
        st.configure("TLabelframe.Label", background=A["bg"], foreground=A["muted"],
                     font=(self.ui, 9, "bold"))
        st.configure("TButton", font=(self.ui, 10), padding=(12, 6))
        st.configure("TCombobox", font=(self.ui, 10))
        st.configure("TEntry", font=(self.ui, 10))
        st.configure("TSpinbox", font=(self.ui, 10))
        st.map("TCheckbutton", background=[("active", A["bg"])])
        st.map("TRadiobutton", background=[("active", A["bg"])])

    def _accent_button(self, parent, text, command, primary=True):
        """A flat, hand-coloured button (ttk themed buttons can't take a custom fill).
        primary=Apple-blue call-to-action; else a quiet neutral button."""
        A = self.AP
        fill = A["accent"] if primary else "#E8E8ED"
        fg = "#FFFFFF" if primary else A["text"]
        hover = A["accent_hover"] if primary else "#DEDEE3"
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                      bg=fill, fg=fg, activebackground=A["accent_press"] if primary else "#D4D4DA",
                      activeforeground=fg, font=(self.ui, 11, "bold" if primary else "normal"),
                      cursor="hand2", padx=18, pady=9, highlightthickness=0,
                      disabledforeground="#B0B0B5")
        b._fill, b._hover = fill, hover
        b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=b._hover))
        b.bind("<Leave>", lambda e: b.config(bg=b._fill))
        return b

    # ---------- UI ----------
    def _build(self):
        s = self.s
        self._apply_apple_style()
        pad = {"padx": 14, "pady": 5}

        self.drop = tk.Label(
            self.root, height=3, relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=self.AP["border"],
            highlightcolor=self.AP["border"],
            bg=self.AP["drop"], fg=self.AP["muted"], font=(self.ui, 12),
            text=("Drag videos or a folder here\nor click Browse" if _DND
                  else "Click Browse to add video(s)"))
        self.drop.pack(fill="x", padx=14, pady=(12, 6))
        if _DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)

        topbtn = ttk.Frame(self.root)
        topbtn.pack(fill="x", **pad)
        btn_browse = ttk.Button(topbtn, text="Browse...", command=self._browse)
        btn_browse.pack(side="left")
        _Tooltip(btn_browse, "Add video files (or a folder) to the queue.")
        btn_remove = ttk.Button(topbtn, text="Remove selected", command=self._remove_sel)
        btn_remove.pack(side="left", padx=6)
        _Tooltip(btn_remove, "Remove the highlighted video from the queue. (Delete key)")
        btn_clear = ttk.Button(topbtn, text="Clear queue", command=self._clear_queue)
        btn_clear.pack(side="left")
        _Tooltip(btn_clear, "Empty the entire queue.")
        ttk.Button(topbtn, text="↑", width=3, command=lambda: self._move(-1)).pack(side="left", padx=(12, 0))
        ttk.Button(topbtn, text="↓", width=3, command=lambda: self._move(1)).pack(side="left")

        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, **pad)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="y")
        qf = ttk.LabelFrame(left, text="Queue - click a video to set its box")
        qf.pack(fill="x")
        self.qlist = tk.Listbox(qf, height=6, width=42, font=(self.ui, 10),
                                activestyle="none", relief="flat", bd=0,
                                highlightthickness=1, highlightbackground=self.AP["border"],
                                bg=self.AP["panel"], fg=self.AP["text"],
                                selectbackground=self.AP["accent"], selectforeground="#FFFFFF")
        self.qlist.pack(side="left", fill="both", padx=(6, 0), pady=6)
        qsb = ttk.Scrollbar(qf, orient="vertical", command=self.qlist.yview)
        qsb.pack(side="left", fill="y", pady=6)
        self.qlist.config(yscrollcommand=qsb.set)
        self.qlist.bind("<<ListboxSelect>>", self._on_select)
        self.qlist.bind("<Delete>", lambda e: self._remove_sel())

        pf = ttk.LabelFrame(left, text="Preview - drag a box over THIS video's subtitle")
        pf.pack(fill="x", pady=(8, 0))
        self.canvas = tk.Canvas(pf, width=PREVIEW_MAX, height=180, bg="#1D1D1F",
                                highlightthickness=1, highlightbackground=self.AP["border"], cursor="cross")
        self.canvas.pack(padx=8, pady=6)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        sl = ttk.Frame(pf)
        sl.pack(fill="x", padx=8)
        ttk.Label(sl, text="Scrub:").pack(side="left")
        self.slider = ttk.Scale(sl, from_=0, to=1, orient="horizontal")
        self.slider.set(0.25)
        self.slider.pack(side="left", fill="x", expand=True, padx=4)
        self.slider.bind("<ButtonRelease-1>", lambda e: self._show_at_slider())
        self.region_lbl = tk.Label(pf, text="(no video selected)", fg=self.AP["muted"],
                                   bg=self.AP["bg"], font=(self.ui, 9))
        self.region_lbl.pack(fill="x", padx=8)
        bb = ttk.Frame(pf)
        bb.pack(pady=4)
        ttk.Button(bb, text="Clear box", command=self._clear_box).pack(side="left", padx=3)
        ttk.Button(bb, text="Preview cover", command=self._preview_cover).pack(side="left", padx=3)

        ab = ttk.LabelFrame(left, text="A/B preview  ·  Original ⇄ Translated")
        ab.pack(fill="x", pady=(8, 0))
        self.ab_choice = tk.StringVar()
        self.ab_cb = ttk.Combobox(ab, textvariable=self.ab_choice, state="readonly", width=36)
        self.ab_cb.pack(fill="x", padx=8, pady=(6, 4))
        abr = ttk.Frame(ab)
        abr.pack(padx=8, pady=(0, 4))
        self.ab_orig_btn = ttk.Button(abr, text="▶ Original", command=lambda: self._ab_play("orig"))
        self.ab_orig_btn.pack(side="left")
        self.ab_trans_btn = ttk.Button(abr, text="▶ Translated", command=lambda: self._ab_play("trans"))
        self.ab_trans_btn.pack(side="left", padx=6)
        self.ab_stop_btn = ttk.Button(abr, text="■ Stop", command=self._ab_stop)
        self.ab_stop_btn.pack(side="left")
        _Tooltip(self.ab_trans_btn, "Hear the dubbed audio. Switching sides keeps the same "
                                    "position, so you A/B the exact same moment.")
        self.ab_status = tk.Label(ab, text="dub a video, then compare here", bg=self.AP["bg"],
                                  fg=self.AP["muted"], font=(self.ui, 9))
        self.ab_status.pack(fill="x", padx=8, pady=(0, 6))

        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        lang = ttk.LabelFrame(right, text="Language & voice")
        lang.pack(fill="x")
        # One target language, picked from a dropdown (keeps all 20 languages without
        # a wall of checkboxes), and one voice dropdown for it. Back-compat: an old
        # multi-select setting picks its first enabled language as the new single one.
        self._name2code = {name: code for code, name in LANG_ORDER}
        self._code2name = {code: name for code, name in LANG_ORDER}
        saved_code = s.get("lang")
        if saved_code not in self._code2name:
            on = s.get("langs_on", {}) or {}
            saved_code = next((c for c, _ in LANG_ORDER if on.get(c)),
                              next(iter(LANG_DEFAULT_ON), LANG_ORDER[0][0]))
        ttk.Label(lang, text="Language").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.lang_sel = tk.StringVar(value=self._code2name.get(saved_code, LANG_ORDER[0][1]))
        self.lang_cb = ttk.Combobox(lang, textvariable=self.lang_sel, state="readonly",
                                    width=22, values=[n for _, n in LANG_ORDER])
        self.lang_cb.grid(row=0, column=1, sticky="w", padx=4, pady=8)
        self.lang_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_voices(self._cur_code()))
        ttk.Label(lang, text="Voice").grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))
        self.voice_sel = tk.StringVar()
        self.voice_cb = ttk.Combobox(lang, textvariable=self.voice_sel, state="readonly", width=28)
        self.voice_cb.grid(row=1, column=1, sticky="w", padx=4, pady=(0, 8))
        pv = ttk.Button(lang, text="▶", width=3,
                        command=lambda: self._preview_voice(self._cur_code()))
        pv.grid(row=1, column=2, padx=4, pady=(0, 8))
        _Tooltip(pv, "Hear a short sample of the selected voice.")
        self._refresh_voices(saved_code)

        opt = ttk.LabelFrame(right, text="Options")
        opt.pack(fill="x", pady=(8, 0))
        self.reels_auto = tk.BooleanVar(value=s.get("preset") == "facebook_reels")
        self.rights_owned = tk.BooleanVar(value=s.get("rights_mode") == "owned_or_licensed")
        preset_row = ttk.Frame(opt)
        preset_row.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Button(preset_row, text="Facebook Reels Auto",
                   command=self._apply_reels_preset).pack(side="left")
        ttk.Checkbutton(preset_row, text="Owned/licensed source",
                        variable=self.rights_owned).pack(side="left", padx=8)
        self.cover = tk.BooleanVar(value=s.get("cover", True))
        cb_cover = ttk.Checkbutton(opt, text="Cover original subtitles", variable=self.cover)
        cb_cover.pack(anchor="w", padx=8)
        _Tooltip(cb_cover, "Blur over the original burned-in (Chinese) subtitle area.\nUse the box on each video, or the bottom strip below if no box is drawn.")
        bf = ttk.Frame(opt)
        bf.pack(fill="x", padx=8)
        ttk.Label(bf, text="Bottom strip % (videos with no box):").pack(side="left")
        self.band = tk.IntVar(value=s.get("band", 18))
        sp_band = ttk.Spinbox(bf, from_=8, to=40, width=4, textvariable=self.band)
        sp_band.pack(side="left", padx=4)
        _Tooltip(sp_band, "Fallback: how much of the bottom to blur when you don't draw a box. Raise if subs still peek through.")
        self.fb = tk.BooleanVar(value=s.get("fb", False))
        cb_fb = ttk.Checkbutton(opt, text="Make 9:16 Facebook-ready (1080x1920)", variable=self.fb)
        cb_fb.pack(anchor="w", padx=8)
        _Tooltip(cb_fb, "Reframe the dubbed video to vertical 1080x1920 with blurred fill so it's post-ready for Facebook/TikTok.")
        self.keepmusic = tk.BooleanVar(value=s.get("keepmusic", False))
        cb_km = ttk.Checkbutton(opt, text="Keep original music (slower)", variable=self.keepmusic)
        cb_km.pack(anchor="w", padx=8)
        _Tooltip(cb_km, "Use Demucs to separate music from the original voice and keep the music bed under the new dub. Much slower.")
        self.srt = tk.BooleanVar(value=s.get("srt", False))
        cb_srt = ttk.Checkbutton(opt, text="Also save subtitle (.srt) file", variable=self.srt)
        cb_srt.pack(anchor="w", padx=8)
        _Tooltip(cb_srt, "Save a translated .srt next to each output (you can drag it into CapCut).")
        self.burn = tk.BooleanVar(value=s.get("burn", False))
        cb_burn = ttk.Checkbutton(opt, text="Burn subtitles onto the video", variable=self.burn)
        cb_burn.pack(anchor="w", padx=8)
        _Tooltip(cb_burn, "Hard-code the translated text onto the video (always visible).")
        self.audioonly = tk.BooleanVar(value=s.get("audioonly", False))
        cb_audio = ttk.Checkbutton(opt, text="Export dubbed audio only (no video)", variable=self.audioonly)
        cb_audio.pack(anchor="w", padx=8)
        _Tooltip(cb_audio, "Skip the video render and save just the dubbed audio (.wav) into the dubbed folder, "
                           "to drop into CapCut over your clip. Cover/burn/9:16 options are ignored in this mode.")
        self.clone = tk.BooleanVar(value=s.get("clone", False))
        cb_clone = ttk.Checkbutton(opt, text="Clone original speaker's voice (needs GPU + setup)",
                                   variable=self.clone)
        cb_clone.pack(anchor="w", padx=8)
        _Tooltip(cb_clone, "Make the dub SOUND LIKE the original speaker, using OpenVoice on your PC. "
                           "Needs an NVIDIA GPU and the one-time setup in SETUP_VOICECLONE.md. "
                           "If it isn't set up, this safely falls back to the normal voice. Slow on CPU.")
        self.gender = tk.BooleanVar(value=s.get("gender", False))
        cb_gender = ttk.Checkbutton(opt, text="Auto male/female voices (match each speaker's gender)",
                                    variable=self.gender)
        cb_gender.pack(anchor="w", padx=8)
        _Tooltip(cb_gender, "Give male lines a male voice and female lines a female voice, judged from "
                            "the original speaker's pitch. Overrides the per-language voice picker and "
                            "uses online (edge-tts) voices. Heuristic: deep women / high men / music-heavy "
                            "lines can misclassify, and two same-gender speakers share a voice. "
                            "Works best with 'Match original voice' on.")
        self.speakers = tk.BooleanVar(value=s.get("speakers", False))
        cb_speakers = ttk.Checkbutton(opt, text="Distinct voice per character (AI/script + audio)",
                                      variable=self.speakers)
        cb_speakers.pack(anchor="w", padx=8)
        _Tooltip(cb_speakers, "Use DeepSeek to read the script and assign stable voices to narrator "
                              "and recurring characters. Falls back to audio speaker detection when "
                              "AI casting is unavailable. With male/female mode, each character gets "
                              "a gender-matched voice, including separate voices for same-gender roles.")
        saved_override = s.get("speaker_override") if isinstance(s.get("speaker_override"), dict) else {}
        self.single_speaker = tk.BooleanVar(value=bool(saved_override.get("enforce_single_speaker")))
        cb_single = ttk.Checkbutton(opt, text="Force one speaker / one voice",
                                    variable=self.single_speaker)
        cb_single.pack(anchor="w", padx=8)
        _Tooltip(cb_single, "For narrator-only clips. Uses the self-learning memory rule to stop "
                            "false diarization splits from creating multiple AI voices.")
        self.scenefx = tk.BooleanVar(value=s.get("scenefx", False))
        cb_scenefx = ttk.Checkbutton(opt, text="Keep scene sounds (panting/shouts, not music)",
                                     variable=self.scenefx)
        cb_scenefx.pack(anchor="w", padx=8)
        _Tooltip(cb_scenefx, "Bring back the original action sounds (panting, grunts, fight shouts) in the "
                             "gaps between dialogue, while still dropping the music. Uses the separated "
                             "voice track, so it needs Demucs (first run downloads it). Note: pure "
                             "impact/hit sounds live in the music track and won't return, and faint "
                             "original speech can occasionally leak in the gaps.")
        self.reuse_analysis = tk.BooleanVar(value=s.get("reuse_analysis", True))
        cb_reuse = ttk.Checkbutton(opt, text="Fast re-dub: reuse transcription for other languages",
                                   variable=self.reuse_analysis)
        cb_reuse.pack(anchor="w", padx=8)
        _Tooltip(cb_reuse, "Cache the transcription, voice analysis and speaker split per video so dubbing "
                           "the SAME video into a DIFFERENT language skips the slow Whisper/diarization "
                           "steps (only translation + voicing run). The cache auto-refreshes if you edit "
                           "the video or change the Accuracy (Whisper model). Untick to force a fresh "
                           "transcription.")
        self.length_fit = tk.BooleanVar(value=s.get("length_fit", False))
        cb_lenfit = ttk.Checkbutton(opt, text="Fit translation length to timing (DeepSeek)",
                                    variable=self.length_fit)
        cb_lenfit.pack(anchor="w", padx=8)
        _Tooltip(cb_lenfit, "After translating, estimate each line's spoken length and ask DeepSeek to "
                            "trim or expand only the lines that would overrun (or fall short of) their "
                            "time slot. Keeps the dub in sync with less audio speed-up, so it sounds more "
                            "natural. Needs a DeepSeek API key; lines that already fit cost nothing.")
        self.multimodal = tk.BooleanVar(value=s.get("multimodal", False))
        cb_mm = ttk.Checkbutton(opt, text="Use video frames for context and sync anchors",
                                variable=self.multimodal)
        cb_mm.pack(anchor="w", padx=8)
        _Tooltip(cb_mm, "Sample video frames, attach visual context to each subtitle line, and use "
                        "lip/timing anchors to keep translated speech from overrunning into the next "
                        "shot. With a DeepSeek key it asks DeepSeek-VL for scene/emotion/ambiguity "
                        "context; without one it still applies local timing anchors.")
        self.multimodal_vision = tk.BooleanVar(value=s.get("multimodal_vision", True))
        mf = ttk.Frame(opt)
        mf.pack(fill="x", padx=8, pady=2)
        ttk.Label(mf, text="Accuracy:").pack(side="left")
        self.model = tk.StringVar(value=s.get("model", "medium"))
        cb_model = ttk.Combobox(mf, textvariable=self.model,
                                values=["tiny", "base", "small", "medium"],
                                width=8, state="readonly")
        cb_model.pack(side="left", padx=4)
        _Tooltip(cb_model, "Whisper transcription size. small is faster; medium is better at short dialogue under music/narration.")
        tf = ttk.Frame(opt)
        tf.pack(fill="x", padx=8, pady=2)
        lbl_tone = ttk.Label(tf, text="Tone:")
        lbl_tone.pack(side="left")
        self.tone = tk.StringVar(value=s.get("tone", "original"))
        ttk.Radiobutton(tf, text="Match original voice", value="original",
                        variable=self.tone).pack(side="left", padx=4)
        ttk.Radiobutton(tf, text="Natural (translation)", value="natural",
                        variable=self.tone).pack(side="left", padx=4)
        _Tooltip(lbl_tone,
                 "Match original voice: mimic the original speaker's pace, pitch and loudness "
                 "(still the AI voice).\nNatural (translation): even, calm pacing in the voice's default tone.")

        tr = ttk.LabelFrame(right, text="Translation (optional)")
        tr.pack(fill="x", pady=(8, 0))
        kr = ttk.Frame(tr)
        kr.pack(fill="x", padx=8, pady=5)
        ttk.Label(kr, text="DeepSeek API key:").pack(side="left")
        self.deepseek_key = tk.StringVar(value=s.get("deepseek_key", ""))
        ent_key = ttk.Entry(kr, textvariable=self.deepseek_key, show="•")
        ent_key.pack(side="left", fill="x", expand=True, padx=4)
        _Tooltip(ent_key,
                 "Optional. Paste a DeepSeek API key for higher-quality, timing-aware "
                 "translation with no Google throttling. The voice itself is unchanged "
                 "(DeepSeek has no text-to-speech). Saved locally only (gitignored). "
                 "Blank = free Google translator, or the DEEPSEEK_API_KEY env var.")
        ar = ttk.Frame(tr)
        ar.pack(fill="x", padx=8, pady=(0, 6))
        btn_asst = ttk.Button(ar, text="🤖  AI Assistant", command=self._open_assistant)
        btn_asst.pack(side="left")
        _Tooltip(btn_asst, "DeepSeek-powered helper: fix the original script, improve a "
                           "translation, or back-translate it to check the meaning. "
                           "Uses the DeepSeek API key above.")
        btn_editor = ttk.Button(ar, text="🖥  Open Editor", command=self._open_web_editor)
        btn_editor.pack(side="left", padx=(6, 0))
        _Tooltip(btn_editor, "Open the browser-based dubbing editor: side-by-side "
                             "source/translation script, Original/Translated preview, a "
                             "segment timeline, and the AI assistant. Opens in your browser.")

        rb = ttk.Frame(right)
        rb.pack(fill="x", pady=14)
        self.run_btn = self._accent_button(rb, "Dub All", self._start, primary=True)
        self.run_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = self._accent_button(rb, "Stop", self._stop, primary=False)
        self.stop_btn.config(state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        prog = ttk.Frame(self.root)
        prog.pack(fill="x", **pad)
        self.stage_lbl = tk.Label(prog, text="Idle", anchor="w", fg=self.AP["text"],
                                  bg=self.AP["bg"], font=(self.ui, 10, "bold"))
        self.stage_lbl.pack(fill="x")
        row = ttk.Frame(prog)
        row.pack(fill="x")
        self.bar = ttk.Progressbar(row, maximum=100, mode="determinate")
        self.bar.pack(side="left", fill="x", expand=True)
        self.pct_lbl = tk.Label(row, text="0%", width=5, bg=self.AP["bg"],
                                fg=self.AP["text"], font=(self.ui, 9))
        self.pct_lbl.pack(side="left")
        self.eta_lbl = tk.Label(row, text="", width=12, bg=self.AP["bg"],
                                fg=self.AP["muted"], font=(self.ui, 9))
        self.eta_lbl.pack(side="left")

        self.log = scrolledtext.ScrolledText(self.root, height=7, font=(self.mono, 9),
                                             relief="flat", bd=0, bg=self.AP["panel"],
                                             fg=self.AP["text"], highlightthickness=1,
                                             highlightbackground=self.AP["border"],
                                             insertbackground=self.AP["text"], padx=8, pady=6)
        self.log.pack(fill="both", expand=True, **pad)
        self.log.tag_configure("err", foreground=self.AP["err"])

        bot = ttk.Frame(self.root)
        bot.pack(fill="x", **pad)
        ttk.Button(bot, text="Open output folder", command=self._open_out).pack(side="left")
        self.status = tk.Label(bot, text="Ready", fg=self.AP["ok"], bg=self.AP["bg"],
                               font=(self.ui, 9))
        self.status.pack(side="right")

    # ---------- A/B preview ----------
    def _ab_add(self, src: str, dub: str):
        """Register a finished dub (and its source) and offer it in the A/B dropdown."""
        label = os.path.basename(dub)
        self._ab_map[label] = (src, dub)
        self.ab_cb["values"] = list(self._ab_map.keys())
        self.ab_choice.set(label)          # auto-select the newest output
        self.ab_status.config(text="ready — play Original or Translated", fg=self.AP["muted"])

    def _ab_pos(self) -> float:
        """Current playback position (s), so switching sides A/Bs the same moment."""
        if self._ab_proc and self._ab_proc.poll() is None:
            return max(0.0, self._ab_pos0 + (time.time() - self._ab_t0))
        return 0.0

    def _ab_play(self, side: str):
        pair = self._ab_map.get(self.ab_choice.get())
        if not pair:
            self.ab_status.config(text="nothing dubbed yet", fg=self.AP["warn"])
            return
        f = pair[0] if side == "orig" else pair[1]
        if not (f and os.path.exists(f)):
            self.ab_status.config(text="file not found", fg=self.AP["err"])
            return
        if not os.path.exists(self.ffplay):
            try:                            # no ffplay: just open in the default player
                os.startfile(f)             # type: ignore[attr-defined]
            except Exception:
                self.ab_status.config(text="ffplay not found", fg=self.AP["err"])
            return
        pos = self._ab_pos()                # keep position when switching sides
        self._ab_stop(reset_status=False)
        try:
            self._ab_proc = subprocess.Popen(
                [self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                 "-ss", f"{pos:.2f}", f], creationflags=_NOWIN)
        except Exception as e:
            self.ab_status.config(text=f"playback failed: {str(e)[:30]}", fg=self.AP["err"])
            return
        self._ab_side, self._ab_t0, self._ab_pos0 = side, time.time(), pos
        name = "Original" if side == "orig" else "Translated"
        self.ab_status.config(text=f"▶ {name}   (from {pos:.0f}s)", fg=self.AP["accent"])

    def _ab_stop(self, reset_status: bool = True):
        if self._ab_proc:
            try:
                self._ab_proc.terminate()
            except Exception:
                pass
        self._ab_proc = None
        self._ab_side = None
        if reset_status and getattr(self, "ab_status", None):
            self.ab_status.config(text="stopped", fg=self.AP["muted"])

    # ---------- AI assistant (DeepSeek) ----------
    def _deepseek(self, system: str, user: str, key: str):
        """One DeepSeek chat call. Returns (reply_text, None) or (None, error_msg).
        `key` is passed in (read on the main thread) - never touch tk vars off-thread."""
        import urllib.request
        body = json.dumps({"model": DEEPSEEK_MODEL, "temperature": 1.0, "stream": False,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode("utf-8")
        req = urllib.request.Request(
            DEEPSEEK_URL, data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"], None
        except Exception as e:
            return None, f"DeepSeek error: {str(e)[:90]}"

    @staticmethod
    def _srt_text(path: str) -> str:
        """Pull just the spoken lines out of a .srt (drop indices and timestamps)."""
        out = []
        try:
            for ln in open(path, encoding="utf-8"):
                ln = ln.strip()
                if ln and "-->" not in ln and not ln.isdigit():
                    out.append(ln)
        except Exception:
            return ""
        return "\n".join(out)

    def _asst_load_srt(self, quiet: bool = False):
        srts = glob.glob(os.path.join(OUT, "*.srt"))
        if not srts:
            if not quiet:
                self._asst_status.config(text="No .srt found — tick 'Also save subtitle' and dub first.",
                                         fg=self.AP["warn"])
            return
        newest = max(srts, key=os.path.getmtime)
        txt = self._srt_text(newest)
        self._asst_in.delete("1.0", "end")
        self._asst_in.insert("1.0", txt)
        self._asst_status.config(text=f"Loaded {os.path.basename(newest)}", fg=self.AP["muted"])

    def _open_web_editor(self):
        """Launch the standalone browser-based editor (web_editor.py) as a child
        process. It serves the dark side-by-side script UI and opens the browser."""
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "web_editor.py")
        if not os.path.exists(script):
            messagebox.showerror("Editor", "web_editor.py not found next to dub_app.py.")
            return
        proc = getattr(self, "_editor_proc", None)
        if proc is not None and proc.poll() is None:
            messagebox.showinfo("Editor", "The editor is already running.\n"
                                          "Check your browser (http://127.0.0.1:8765).")
            return
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._editor_proc = subprocess.Popen([sys.executable, script], cwd=here,
                                             creationflags=flags)

    def _open_assistant(self):
        if getattr(self, "_asst", None) is not None and self._asst.winfo_exists():
            self._asst.lift()
            return
        A = self.AP
        win = tk.Toplevel(self.root)
        self._asst = win
        win.title("AI Assistant · DeepSeek")
        win.configure(bg=A["bg"])
        win.geometry("640x660")
        win.minsize(520, 520)

        tk.Label(win, text="AI Assistant", bg=A["bg"], fg=A["text"],
                 font=(self.ui, 15, "bold")).pack(anchor="w", padx=16, pady=(14, 0))
        tk.Label(win, text="Fix the original script, improve a translation, or back-translate "
                           "it to check the meaning.", bg=A["bg"], fg=A["muted"],
                 font=(self.ui, 9), justify="left", wraplength=600).pack(anchor="w", padx=16, pady=(2, 8))

        top = ttk.Frame(win)
        top.pack(fill="x", padx=16)
        ttk.Label(top, text="Task").pack(side="left")
        self._asst_task = tk.StringVar(value=ASSISTANT_TASKS[0])
        ttk.Combobox(top, textvariable=self._asst_task, values=ASSISTANT_TASKS,
                     state="readonly", width=24).pack(side="left", padx=6)
        ttk.Label(top, text="Language").pack(side="left", padx=(12, 0))
        names = [n for _, n in LANG_ORDER]
        self._asst_lang = tk.StringVar(value=self._code2name.get(self._cur_code(), names[0]))
        ttk.Combobox(top, textvariable=self._asst_lang, values=names,
                     state="readonly", width=14).pack(side="left", padx=6)

        cf = ttk.Frame(win)
        cf.pack(fill="x", padx=16, pady=(8, 0))
        ttk.Label(cf, text="Custom instruction (for the Custom task):").pack(side="left")
        self._asst_custom = tk.StringVar()
        ttk.Entry(cf, textvariable=self._asst_custom).pack(side="left", fill="x", expand=True, padx=6)

        inhdr = ttk.Frame(win)
        inhdr.pack(fill="x", padx=16, pady=(10, 2))
        ttk.Label(inhdr, text="Input").pack(side="left")
        ttk.Button(inhdr, text="Load last .srt", command=self._asst_load_srt).pack(side="right")
        self._asst_in = scrolledtext.ScrolledText(win, height=9, font=(self.mono, 9),
                                                  relief="flat", bd=0, bg=A["panel"], fg=A["text"],
                                                  highlightthickness=1, highlightbackground=A["border"],
                                                  insertbackground=A["text"], padx=8, pady=6)
        self._asst_in.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        runrow = ttk.Frame(win)
        runrow.pack(fill="x", padx=16)
        self._asst_run_btn = self._accent_button(runrow, "Ask DeepSeek", self._asst_run, primary=True)
        self._asst_run_btn.pack(side="left")
        ttk.Button(runrow, text="Copy result", command=self._asst_copy).pack(side="left", padx=6)
        ttk.Button(runrow, text="↧ Input", command=self._asst_result_to_input).pack(side="left")
        ttk.Button(runrow, text="⇄ A/B preview", command=self._asst_ab).pack(side="left", padx=6)
        self._asst_status = tk.Label(win, text="Ready", bg=A["bg"], fg=A["muted"], font=(self.ui, 9))
        self._asst_status.pack(anchor="w", padx=16, pady=(6, 2))

        ttk.Label(win, text="Result").pack(anchor="w", padx=16)
        self._asst_out = scrolledtext.ScrolledText(win, height=9, font=(self.mono, 9),
                                                   relief="flat", bd=0, bg=A["panel"], fg=A["text"],
                                                   highlightthickness=1, highlightbackground=A["border"],
                                                   insertbackground=A["text"], padx=8, pady=6)
        self._asst_out.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        win.protocol("WM_DELETE_WINDOW", lambda: (setattr(self, "_asst", None), win.destroy()))

        # Auto-load the newest dubbed .srt so the box is never empty on open.
        self._asst_load_srt(quiet=True)

    def _asst_copy(self):
        txt = self._asst_out.get("1.0", "end").strip()
        if txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self._asst_status.config(text="Result copied.", fg=self.AP["ok"])

    def _asst_result_to_input(self):
        txt = self._asst_out.get("1.0", "end").strip()
        if txt:
            self._asst_in.delete("1.0", "end")
            self._asst_in.insert("1.0", txt)

    def _asst_ab(self):
        """Side-by-side A/B preview of the script: Input (A) vs Result (B),
        aligned line by line with changed lines highlighted and synced scroll."""
        a = self._asst_in.get("1.0", "end").rstrip("\n")
        b = self._asst_out.get("1.0", "end").rstrip("\n")
        if not a.strip() and not b.strip():
            self._asst_status.config(text="Nothing to compare — load a script and ask DeepSeek first.",
                                     fg=self.AP["warn"])
            return
        A = self.AP
        if getattr(self, "_asst_ab_win", None) is not None and self._asst_ab_win.winfo_exists():
            self._asst_ab_win.destroy()
        win = tk.Toplevel(self.root)
        self._asst_ab_win = win
        win.title("A/B script preview · Original ⇄ Result")
        win.configure(bg=A["bg"])
        win.geometry("920x600")
        win.minsize(640, 360)

        hdr = ttk.Frame(win)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(hdr, text="A · Original / Input", bg=A["bg"], fg=A["muted"],
                 font=(self.ui, 9, "bold")).pack(side="left", expand=True)
        tk.Label(hdr, text="B · Result", bg=A["bg"], fg=A["muted"],
                 font=(self.ui, 9, "bold")).pack(side="left", expand=True)

        body = ttk.Frame(win)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        sb = ttk.Scrollbar(body, orient="vertical")
        sb.pack(side="right", fill="y")

        def _mk():
            t = tk.Text(body, wrap="word", font=(self.mono, 9), relief="flat", bd=0,
                        bg=A["panel"], fg=A["text"], highlightthickness=1,
                        highlightbackground=A["border"], padx=8, pady=6)
            t.tag_configure("chg", background="#FFF3C4", foreground="#5b4a00")
            t.tag_configure("num", foreground=A["muted"])
            return t

        ta, tb = _mk(), _mk()
        ta.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tb.pack(side="left", fill="both", expand=True)

        la, lb = a.split("\n"), b.split("\n")
        nchg = 0
        for i in range(max(len(la), len(lb))):
            xa = la[i] if i < len(la) else ""
            xb = lb[i] if i < len(lb) else ""
            diff = xa.strip() != xb.strip()
            nchg += diff
            tag = ("chg",) if diff else ()
            ta.insert("end", f"{i + 1:>3}  ", ("num",)); ta.insert("end", xa + "\n", tag)
            tb.insert("end", f"{i + 1:>3}  ", ("num",)); tb.insert("end", xb + "\n", tag)
        ta.config(state="disabled"); tb.config(state="disabled")

        # One scrollbar drives both panes; mouse wheel scrolls both in lockstep.
        ta.config(yscrollcommand=sb.set)
        sb.config(command=lambda *ar: (ta.yview(*ar), tb.yview(*ar)))

        def _wheel(e):
            d = -1 if e.delta > 0 else 1
            ta.yview_scroll(d, "units"); tb.yview_scroll(d, "units")
            return "break"

        for t in (ta, tb):
            t.bind("<MouseWheel>", _wheel)

        ft = ttk.Frame(win)
        ft.pack(fill="x", padx=12, pady=(0, 12))
        msg = (f"{nchg} of {max(len(la), len(lb))} lines differ — highlighted."
               if (a.strip() and b.strip()) else
               "Only one side has text — run a task to fill the other.")
        tk.Label(ft, text=msg, bg=A["bg"], fg=A["muted"], font=(self.ui, 9)).pack(side="left")
        ttk.Button(ft, text="Close", command=win.destroy).pack(side="right")

    def _asst_run(self):
        text = self._asst_in.get("1.0", "end").strip()
        if not text:
            self._asst_status.config(text="Enter or load some text first.", fg=self.AP["warn"])
            return
        # Read the key on the MAIN thread (tk vars are not thread-safe).
        key = (self.deepseek_key.get().strip() or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            self._asst_status.config(text="No DeepSeek API key — paste one in the Translation box above.",
                                     fg=self.AP["warn"])
            return
        task = self._asst_task.get()
        lang_name = self._asst_lang.get()
        if task == "Correct original script":
            system = ("You are a meticulous transcription editor. Fix speech-to-text errors, "
                      "punctuation, spacing and obviously wrong characters in the user's text "
                      "WITHOUT changing its meaning or its language. Keep one line per subtitle "
                      "(same number of lines). Return ONLY the corrected text.")
        elif task == "Improve translation":
            system = (f"You are a professional video-dubbing translator. Improve the following "
                      f"{lang_name} dub translation so it sounds natural and spoken, and stays "
                      f"concise enough to fit the original timing. Keep one line per subtitle and "
                      f"the same number of lines. Return ONLY the improved {lang_name} translation.")
        elif task == "Back-translation check":
            system = (f"Translate the user's text into {lang_name}, line by line, as literally as is "
                      f"natural, so they can verify the meaning is correct. Keep one line per input "
                      f"line. Return ONLY the {lang_name} translation.")
        else:
            system = self._asst_custom.get().strip() or \
                "You are a helpful assistant for video dubbing. Follow the user's instruction."
        self._asst_status.config(text="Asking DeepSeek…", fg=self.AP["accent"])
        self._asst_run_btn.config(state="disabled")

        def work():
            out, err = self._deepseek(system, text, key)

            def show():
                if not (getattr(self, "_asst", None) and self._asst.winfo_exists()):
                    return
                self._asst_run_btn.config(state="normal")
                if err:
                    self._asst_status.config(text=err, fg=self.AP["err"])
                    return
                self._asst_out.delete("1.0", "end")
                self._asst_out.insert("1.0", out or "")
                self._asst_status.config(text="Done.", fg=self.AP["ok"])
            self.root.after(0, show)
        threading.Thread(target=work, daemon=True).start()

    # ---------- voices ----------
    def _cur_code(self) -> str:
        """The currently selected target-language code."""
        return self._name2code.get(self.lang_sel.get(), LANG_ORDER[0][0])

    def _refresh_voices(self, code: str):
        """Repopulate the voice dropdown for `code`, preferring the saved/default voice."""
        choices = VOICE_CHOICES.get(code, [])
        labels = [c["label"] for c in choices]
        self.voice_cb["values"] = labels
        want = (self.s.get("voices", {}) or {}).get(code) or DEFAULT_VOICE.get(code)
        sel = next((c["label"] for c in choices if c["id"] == want),
                   labels[0] if labels else "")
        self.voice_sel.set(sel)

    def _voice_id(self, lang: str) -> str:
        """Resolve the chosen voice id for `lang` (only the current language has a live
        selection; others fall back to their saved or default voice)."""
        if lang == self._cur_code():
            label = self.voice_sel.get()
            for c in VOICE_CHOICES.get(lang, []):
                if c["label"] == label:
                    return c["id"]
        return (self.s.get("voices", {}) or {}).get(lang) or DEFAULT_VOICE.get(lang, "")

    def _preview_voice(self, lang):
        voice = self._voice_id(lang)
        if not voice:
            return
        self.status.config(text=f"Previewing {voice}...", fg=self.AP["warn"])

        def work():
            try:
                import asyncio
                import edge_tts
                import winsound
                os.makedirs(WORK, exist_ok=True)
                mp3 = os.path.join(WORK, "_voice_preview.mp3")
                wav = os.path.join(WORK, "_voice_preview.wav")
                text = SAMPLE_TEXT.get(lang, "Hello, this is a sample of this voice.")
                asyncio.run(edge_tts.Communicate(text, voice).save(mp3))
                subprocess.run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                "-i", mp3, wav], creationflags=_NOWIN)
                winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.root.after(0, lambda: self.status.config(text="Ready", fg=self.AP["ok"]))
            except Exception as e:
                msg = str(e)[:70]
                self.q.put(("log", f"(voice preview unavailable: {msg})\n"))
                self.root.after(0, lambda: self.status.config(text="Ready", fg=self.AP["ok"]))

        threading.Thread(target=work, daemon=True).start()

    def _apply_reels_preset(self):
        self.reels_auto.set(True)
        self.rights_owned.set(True)
        self.cover.set(True)
        self.fb.set(True)
        self.keepmusic.set(True)
        self.srt.set(True)
        self.burn.set(True)
        self.audioonly.set(False)
        self.clone.set(False)
        self.gender.set(True)
        self.speakers.set(True)
        self.scenefx.set(True)
        self.reuse_analysis.set(True)
        self.length_fit.set(True)
        self.model.set("medium")
        self.tone.set("original")
        self.status.config(text="Facebook Reels Auto applied", fg=self.AP["ok"])

    # ---------- queue ----------
    def _expand_paths(self, paths):
        out = []
        for p in paths:
            if os.path.isdir(p):
                for r, _d, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(VIDEO_EXT):
                            out.append(os.path.join(r, f))
            elif p.lower().endswith(VIDEO_EXT):
                out.append(p)
        return out

    def _add(self, paths):
        if self.running:
            return
        existing = {it["path"] for it in self.queue}
        for p in self._expand_paths(paths):
            if p not in existing:
                self.queue.append({"path": p, "region": None, "box": None, "status": "pending"})
                existing.add(p)
        self._refresh_queue()
        if self.sel is None and self.queue:
            self.qlist.selection_set(0)
            self._on_select(None)

    def _on_drop(self, event):
        self._add(list(self.root.tk.splitlist(event.data)))

    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Add video(s)",
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.avi *.ts"), ("All files", "*.*")])
        if paths:
            self._add(list(paths))

    def _remove_sel(self):
        if self.sel is None or self.running:
            return
        if 0 <= self.sel < len(self.queue):
            self.queue.pop(self.sel)
            self.sel = None
            self.tkimg = None
            self.canvas.delete("all")
            self.region_lbl.config(text="(no video selected)")
            self._refresh_queue()

    def _clear_queue(self):
        if self.running:
            return
        self.queue = []
        self.sel = None
        self.tkimg = None
        self.canvas.delete("all")
        self.region_lbl.config(text="(no video selected)")
        self._refresh_queue()

    def _move(self, delta):
        if self.sel is None or self.running:
            return
        new = self.sel + delta
        if 0 <= new < len(self.queue):
            self.queue[self.sel], self.queue[new] = self.queue[new], self.queue[self.sel]
            self.sel = new
            self._refresh_queue()
            self.qlist.selection_clear(0, "end")
            self.qlist.selection_set(new)
            self.qlist.see(new)

    def _refresh_queue(self):
        self.qlist.delete(0, "end")
        for it in self.queue:
            mark = " [box]" if it["region"] else ""
            self.qlist.insert("end", f"{_ICON.get(it['status'], '•')} {os.path.basename(it['path'])}{mark}")
        if self.sel is not None and 0 <= self.sel < len(self.queue):
            self.qlist.selection_set(self.sel)

    def _on_select(self, _evt):
        sel = self.qlist.curselection()
        if not sel:
            return
        self.sel = sel[0]
        item = self.queue[self.sel]
        self.dur = self._duration(item["path"])
        self.slider.set(0.25)
        self._show_at_slider()

    # ---------- preview (non-blocking) ----------
    def _duration(self, video):
        ffprobe = self.ffprobe if os.path.exists(self.ffprobe) else "ffprobe"
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=nw=1:nk=1", video],
                               capture_output=True, text=True, creationflags=_NOWIN)
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    def _show_at_slider(self):
        if self.sel is None:
            return
        self._preview_token += 1
        token = self._preview_token
        video = self.queue[self.sel]["path"]
        t = self.slider.get() * self.dur * 0.98 if self.dur > 0 else 1.0
        threading.Thread(target=self._extract_preview,
                         args=(video, t, token), daemon=True).start()

    def _extract_preview(self, video, t, token):
        png = os.path.join(WORK, "_preview.png")
        os.makedirs(WORK, exist_ok=True)
        try:
            subprocess.run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                            "-ss", str(t), "-i", video, "-frames:v", "1",
                            "-vf", f"scale=w={PREVIEW_MAX}:h={PREVIEW_MAX}:force_original_aspect_ratio=decrease",
                            png], creationflags=_NOWIN)
        except Exception:
            return
        self.root.after(0, lambda: self._apply_preview(png, token))

    def _apply_preview(self, png, token):
        if token != self._preview_token or self.sel is None:
            return
        try:
            img = tk.PhotoImage(file=png)
        except Exception:
            return
        self.tkimg = img
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        item = self.queue[self.sel]
        if item["region"] and not item["box"]:
            x, y, w, h = item["region"]
            item["box"] = (int(x * img.width()), int(y * img.height()),
                           int((x + w) * img.width()), int((y + h) * img.height()))
        self._redraw_box()
        self._update_region_lbl()

    def _preview_cover(self):
        if self.sel is None or self.tkimg is None:
            return
        item = self.queue[self.sel]
        if item["region"]:
            x, y, w, h = item["region"]
        else:
            b = max(8, min(40, self.band.get())) / 100.0
            x, y, w, h = 0.0, 1.0 - b, 1.0, b
        src = os.path.join(WORK, "_preview.png")
        dst = os.path.join(WORK, "_preview_cov.png")
        if not os.path.exists(src):
            return
        try:
            subprocess.run([self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                            "-filter_complex",
                            f"[0:v]split=2[a][b];[b]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},gblur=sigma=40:steps=3[bb];"
                            f"[a][bb]overlay=W*{x}:H*{y}", dst], creationflags=_NOWIN)
            img = tk.PhotoImage(file=dst)
        except Exception:
            return
        self.tkimg = img
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self._redraw_box()

    # ---------- box ----------
    def _press(self, e):
        self._bx0, self._by0 = e.x, e.y
        self.canvas.delete("box")

    def _drag(self, e):
        self.canvas.delete("box")
        self.canvas.create_rectangle(self._bx0, self._by0, e.x, e.y,
                                     outline="#ff3b3b", width=2, tags="box")

    def _release(self, e):
        if self.sel is None or self.tkimg is None:
            return
        iw, ih = self.tkimg.width(), self.tkimg.height()
        x0, x1 = sorted((max(0, min(iw, self._bx0)), max(0, min(iw, e.x))))
        y0, y1 = sorted((max(0, min(ih, self._by0)), max(0, min(ih, e.y))))
        item = self.queue[self.sel]
        if x1 - x0 < 6 or y1 - y0 < 6:
            item["region"] = None
            item["box"] = None
        else:
            item["box"] = (x0, y0, x1, y1)
            item["region"] = (x0 / iw, y0 / ih, (x1 - x0) / iw, (y1 - y0) / ih)
        self._redraw_box()
        self._update_region_lbl()
        self._refresh_queue()

    def _redraw_box(self):
        self.canvas.delete("box")
        if self.sel is not None and self.queue[self.sel]["box"]:
            self.canvas.create_rectangle(*self.queue[self.sel]["box"],
                                         outline="#ff3b3b", width=2, tags="box")

    def _clear_box(self):
        if self.sel is None:
            return
        self.queue[self.sel]["region"] = None
        self.queue[self.sel]["box"] = None
        self.canvas.delete("box")
        self._update_region_lbl()
        self._refresh_queue()

    def _update_region_lbl(self):
        if self.sel is None:
            self.region_lbl.config(text="(no video selected)")
            return
        r = self.queue[self.sel]["region"]
        if r:
            self.region_lbl.config(text=f"Box set: {int(r[2] * 100)}% wide x {int(r[3] * 100)}% tall")
        else:
            self.region_lbl.config(text="No box - uses bottom strip default")

    # ---------- run / stop ----------
    def _langs(self) -> str:
        return self._cur_code()

    def _start(self):
        if self.running:
            return
        if not self.queue:
            messagebox.showwarning("No videos", "Add videos to the queue first.")
            return
        langs = self._langs()
        if not langs:
            messagebox.showwarning("No language", "Tick at least one language.")
            return
        self._save_settings()
        self.error_lines = []
        os.makedirs(OUT, exist_ok=True)
        os.makedirs(WORK, exist_ok=True)
        videos = []
        for it in self.queue:
            region = ",".join(f"{v:.4f}" for v in it["region"]) if it["region"] else ""
            videos.append({"path": it["path"], "region": region})
            it["status"] = "pending"
        reels = self.reels_auto.get()
        cfg = {"videos": videos, "out": OUT, "ffmpeg": self.ffmpeg, "work": WORK,
               "langs": langs, "model": self.model.get(),
               "preset": "facebook_reels" if reels else "",
               "rights_mode": "owned_or_licensed" if self.rights_owned.get() else "",
               "keepmusic": self.keepmusic.get() or reels,
               "cover": self.cover.get() or reels,
               "band": self.band.get(), "fb": self.fb.get() or reels,
               "srt": self.srt.get() or reels, "naming": "firstline",
               "burn": self.burn.get() or reels,
               "tone": self.tone.get(), "audioonly": self.audioonly.get(),
               "clone": self.clone.get() and not (self.speakers.get() or reels or self.single_speaker.get()),
               "gender": (self.gender.get() or reels) and not self.single_speaker.get(),
               "speakers": (self.speakers.get() or reels) and not self.single_speaker.get(),
               "scenefx": self.scenefx.get(),
               "reuse_analysis": self.reuse_analysis.get(),
               "length_fit": self.length_fit.get() or reels,
               "multimodal": self.multimodal.get() or reels,
               "multimodal_vision": self.multimodal_vision.get(),
               "multimodal_fps": 1.0,
               "expected_speakers": 1 if self.single_speaker.get() else None,
               "speaker_override": {
                   "enforce_single_speaker": self.single_speaker.get(),
                   "primary_speaker_id": "Speaker_1",
                   "diarization_sensitivity": 0.0 if self.single_speaker.get() else 0.5,
               },
               "deepseek_key": "",
               "voices": {lg: self._voice_id(lg) for lg in langs.split(",") if lg}}
        cfg = optimize_dub_job_config(cfg, os.path.join(BASE, "system_memory.json"))
        self._runtime_deepseek_key = self.deepseek_key.get().strip()
        try:
            with open(JOBS, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write job file:\n{e}")
            return
        self._refresh_queue()
        self.running = True
        self.cancel = False
        self.run_btn.config(state="disabled", text="Working…")
        self.stop_btn.config(state="normal")
        self.status.config(text="Working…", fg=self.AP["warn"])
        threading.Thread(target=self._worker, daemon=True).start()

    def _stop(self):
        self.cancel = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.status.config(text="Stopping…", fg=self.AP["err"])

    def _worker(self):
        cmd = [sys.executable, "-u", os.path.join(BASE, "dub.py"), "--jobs", JOBS]
        try:
            env = os.environ.copy()
            if getattr(self, "_runtime_deepseek_key", ""):
                env["DEEPSEEK_API_KEY"] = self._runtime_deepseek_key
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, encoding="utf-8", errors="replace",
                                 creationflags=_NOWIN, env=env)
            self.proc = p
            for line in p.stdout:
                line = line.rstrip("\n")
                if line.startswith("__PCT__"):
                    self.q.put(("pct", line.split()[-1]))
                elif line.startswith("__STAGE__"):
                    self.q.put(("stage", line[len("__STAGE__"):].strip()))
                elif line.startswith("__FILE__"):
                    self.q.put(("file", line.split()[1]))
                elif line.startswith("__FILEDONE__"):
                    self.q.put(("filedone", line.split()[1]))
                elif line.startswith("__OUT__"):
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        self.q.put(("out", (parts[1], parts[2])))
                else:
                    if re.search(r"\b(ERROR|FAILED|Traceback|cannot|No audio was received)\b", line):
                        self.q.put(("err", line + "\n"))
                    else:
                        self.q.put(("log", line + "\n"))
            p.wait()
            self.proc = None
        except Exception as e:
            self.q.put(("err", f"ERROR: {e}\n"))
        self.q.put(("done", None))

    def _pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", val)
                    self.log.see("end")
                elif kind == "err":
                    self.log.insert("end", val, ("err",))
                    self.log.see("end")
                    self.error_lines.append(val.strip())
                elif kind == "stage":
                    self.stage_lbl.config(text=val)
                    self.bar["value"] = 0
                    self.pct_lbl.config(text="0%")
                    self.eta_lbl.config(text="")
                    self.stage_start = time.time()
                elif kind == "pct":
                    try:
                        n = int(val)
                    except ValueError:
                        n = 0
                    self.bar["value"] = n
                    self.pct_lbl.config(text=f"{n}%")
                    if n > 3 and self.stage_start:
                        el = time.time() - self.stage_start
                        self.eta_lbl.config(text=f"~{int(el / n * (100 - n))}s left")
                elif kind in ("file", "filedone"):
                    try:
                        idx = int(val) - 1
                    except ValueError:
                        idx = -1
                    if 0 <= idx < len(self.queue):
                        self.queue[idx]["status"] = "working" if kind == "file" else "done"
                        self._refresh_queue()
                elif kind == "out":
                    self._ab_add(val[0], val[1])
                elif kind == "done":
                    self._on_run_done()
        except queue.Empty:
            pass
        self.root.after(120, self._pump)

    def _on_run_done(self):
        self.running = False
        self.proc = None
        self.run_btn.config(state="normal", text="Dub All")
        self.stop_btn.config(state="disabled")
        if self.cancel:
            self.status.config(text="Stopped", fg=self.AP["err"])
            self.stage_lbl.config(text="Stopped")
            return
        self.status.config(text="Done", fg=self.AP["ok"])
        self.stage_lbl.config(text="Done")
        self.bar["value"] = 100
        self.pct_lbl.config(text="100%")
        n_done = sum(1 for it in self.queue if it["status"] == "done")
        if self.error_lines:
            sample = "\n".join(self.error_lines[:5])
            messagebox.showwarning("Finished with errors",
                                   f"{n_done} video(s) finished. Some errors were logged:\n\n{sample}")
        else:
            messagebox.showinfo("Done", f"All {n_done} video(s) dubbed.\nOpening the output folder.")
        self._open_out()

    def _open_out(self):
        os.makedirs(OUT, exist_ok=True)
        try:
            os.startfile(OUT)
        except Exception:
            pass


def main():
    lock = grab_single_instance_lock()
    if lock is None:
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("Already running",
                             "Douyin Dubbing is already open in another window.\n"
                             "Close that one first - running two at once will throttle the voice service.")
        return
    root = TkinterDnD.Tk() if _DND else tk.Tk()
    if _THEMED:
        try:
            sv_ttk.set_theme("light")
        except Exception:
            pass
    DubApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
