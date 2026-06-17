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
}

# The voice each language starts on (matches dub.py's built-in defaults).
DEFAULT_VOICE = {
    "vi": "vi-VN-HoaiMyNeural", "id": "id-ID-GadisNeural", "ms": "ms-MY-YasminNeural",
    "es": "es-MX-DaliaNeural", "en": "en-US-AriaNeural",
}

# Short line spoken when the user clicks ▶ to preview a voice.
SAMPLE_TEXT = {
    "vi": "Xin chào, đây là giọng đọc thử.",
    "id": "Halo, ini adalah contoh suara.",
    "ms": "Helo, ini ialah contoh suara.",
    "es": "Hola, esta es una muestra de voz.",
    "en": "Hello, this is a sample of this voice.",
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
        self.s = self._load_settings()
        root.title("Douyin Dubbing")
        geom = self.s.get("geom")
        root.geometry(geom if geom else "780x920")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._restore_queue()
        self.root.after(120, self._pump)

    # ---------- settings ----------
    def _load_settings(self) -> dict:
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_settings(self):
        data = {"vi": self.vi.get(), "es": self.es.get(), "idn": self.idn.get(),
                "en": self.en.get(), "ms": self.ms.get(), "cover": self.cover.get(),
                "band": self.band.get(),
                "keepmusic": self.keepmusic.get(), "model": self.model.get(),
                "fb": self.fb.get(), "srt": self.srt.get(), "burn": self.burn.get(),
                "tone": self.tone.get(), "audioonly": self.audioonly.get(),
                "clone": self.clone.get(),
                "gender": self.gender.get(),
                "scenefx": self.scenefx.get(),
                "deepseek_key": self.deepseek_key.get(),
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

    # ---------- UI ----------
    def _build(self):
        s = self.s
        pad = {"padx": 10, "pady": 3}

        self.drop = tk.Label(
            self.root, height=2, relief="ridge", borderwidth=2,
            bg="#dbe6f5", fg="#1a3357", font=("Segoe UI", 11),
            text=("Drag videos or a folder here  ↓   (or click Browse)" if _DND
                  else "Click Browse to add video(s)"))
        self.drop.pack(fill="x", **pad)
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
        self.qlist = tk.Listbox(qf, height=6, width=42, font=("Segoe UI", 9), activestyle="dotbox")
        self.qlist.pack(side="left", fill="both", padx=(6, 0), pady=6)
        qsb = ttk.Scrollbar(qf, orient="vertical", command=self.qlist.yview)
        qsb.pack(side="left", fill="y", pady=6)
        self.qlist.config(yscrollcommand=qsb.set)
        self.qlist.bind("<<ListboxSelect>>", self._on_select)
        self.qlist.bind("<Delete>", lambda e: self._remove_sel())

        pf = ttk.LabelFrame(left, text="Preview - drag a box over THIS video's subtitle")
        pf.pack(fill="x", pady=(8, 0))
        self.canvas = tk.Canvas(pf, width=PREVIEW_MAX, height=180, bg="#222",
                                highlightthickness=1, highlightbackground="#888", cursor="cross")
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
        self.region_lbl = tk.Label(pf, text="(no video selected)", fg="#555")
        self.region_lbl.pack(fill="x", padx=8)
        bb = ttk.Frame(pf)
        bb.pack(pady=4)
        ttk.Button(bb, text="Clear box", command=self._clear_box).pack(side="left", padx=3)
        ttk.Button(bb, text="Preview cover", command=self._preview_cover).pack(side="left", padx=3)

        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        lang = ttk.LabelFrame(right, text="Languages & voices")
        lang.pack(fill="x")
        self.vi = tk.BooleanVar(value=s.get("vi", True))
        self.es = tk.BooleanVar(value=s.get("es", True))
        self.idn = tk.BooleanVar(value=s.get("idn", False))
        self.en = tk.BooleanVar(value=s.get("en", False))
        self.ms = tk.BooleanVar(value=s.get("ms", False))
        self.voice_var = {}
        ttk.Label(lang, text="Tick a language, pick its voice, ▶ to preview:",
                  foreground="#666").grid(row=0, column=0, columnspan=3,
                                          sticky="w", padx=8, pady=(4, 2))
        self._lang_row(lang, 1, "Vietnamese", self.vi, "vi")
        self._lang_row(lang, 2, "Indonesian", self.idn, "id")
        self._lang_row(lang, 3, "Malay", self.ms, "ms")
        self._lang_row(lang, 4, "Spanish", self.es, "es")
        self._lang_row(lang, 5, "English", self.en, "en")

        opt = ttk.LabelFrame(right, text="Options")
        opt.pack(fill="x", pady=(8, 0))
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
        self.scenefx = tk.BooleanVar(value=s.get("scenefx", False))
        cb_scenefx = ttk.Checkbutton(opt, text="Keep scene sounds (panting/shouts, not music)",
                                     variable=self.scenefx)
        cb_scenefx.pack(anchor="w", padx=8)
        _Tooltip(cb_scenefx, "Bring back the original action sounds (panting, grunts, fight shouts) in the "
                             "gaps between dialogue, while still dropping the music. Uses the separated "
                             "voice track, so it needs Demucs (first run downloads it). Note: pure "
                             "impact/hit sounds live in the music track and won't return, and faint "
                             "original speech can occasionally leak in the gaps.")
        mf = ttk.Frame(opt)
        mf.pack(fill="x", padx=8, pady=2)
        ttk.Label(mf, text="Accuracy:").pack(side="left")
        self.model = tk.StringVar(value=s.get("model", "small"))
        cb_model = ttk.Combobox(mf, textvariable=self.model,
                                values=["tiny", "base", "small", "medium"],
                                width=8, state="readonly")
        cb_model.pack(side="left", padx=4)
        _Tooltip(cb_model, "Whisper transcription size. base/small for speed, medium for better accuracy on long clips.")
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

        rb = ttk.Frame(right)
        rb.pack(fill="x", pady=10)
        try:
            self.run_btn = ttk.Button(rb, text="DUB ALL", command=self._start, style="Accent.TButton")
        except Exception:
            self.run_btn = ttk.Button(rb, text="DUB ALL", command=self._start)
        self.run_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(rb, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        prog = ttk.Frame(self.root)
        prog.pack(fill="x", **pad)
        self.stage_lbl = tk.Label(prog, text="Idle", anchor="w", fg="#333",
                                  font=("Segoe UI", 9, "bold"))
        self.stage_lbl.pack(fill="x")
        row = ttk.Frame(prog)
        row.pack(fill="x")
        self.bar = ttk.Progressbar(row, maximum=100, mode="determinate")
        self.bar.pack(side="left", fill="x", expand=True)
        self.pct_lbl = tk.Label(row, text="0%", width=5)
        self.pct_lbl.pack(side="left")
        self.eta_lbl = tk.Label(row, text="", width=12, fg="#666")
        self.eta_lbl.pack(side="left")

        self.log = scrolledtext.ScrolledText(self.root, height=7, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, **pad)
        self.log.tag_configure("err", foreground="#c0392b")

        bot = ttk.Frame(self.root)
        bot.pack(fill="x", **pad)
        ttk.Button(bot, text="Open output folder", command=self._open_out).pack(side="left")
        self.status = tk.Label(bot, text="Ready", fg="#070")
        self.status.pack(side="right")

    # ---------- voices ----------
    def _lang_row(self, parent, r, text, var, lang):
        ttk.Checkbutton(parent, text=text, variable=var).grid(
            row=r, column=0, sticky="w", padx=8, pady=2)
        labels = [c["label"] for c in VOICE_CHOICES.get(lang, [])]
        v = tk.StringVar(value=self._initial_voice_label(lang))
        self.voice_var[lang] = v
        cb = ttk.Combobox(parent, textvariable=v, values=labels, width=26, state="readonly")
        cb.grid(row=r, column=1, sticky="w", padx=4)
        pv = ttk.Button(parent, text="▶", width=3,
                        command=lambda lg=lang: self._preview_voice(lg))
        pv.grid(row=r, column=2, padx=2)
        _Tooltip(pv, "Hear a short sample of the selected voice.")

    def _initial_voice_label(self, lang):
        choices = VOICE_CHOICES.get(lang, [])
        want = self.s.get("voices", {}).get(lang) or DEFAULT_VOICE.get(lang)
        for c in choices:
            if c["id"] == want:
                return c["label"]
        return choices[0]["label"] if choices else ""

    def _voice_id(self, lang):
        label = self.voice_var[lang].get() if lang in self.voice_var else ""
        for c in VOICE_CHOICES.get(lang, []):
            if c["label"] == label:
                return c["id"]
        return DEFAULT_VOICE.get(lang, "")

    def _preview_voice(self, lang):
        voice = self._voice_id(lang)
        if not voice:
            return
        self.status.config(text=f"Previewing {voice}...", fg="#a60")

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
                self.root.after(0, lambda: self.status.config(text="Ready", fg="#070"))
            except Exception as e:
                msg = str(e)[:70]
                self.q.put(("log", f"(voice preview unavailable: {msg})\n"))
                self.root.after(0, lambda: self.status.config(text="Ready", fg="#070"))

        threading.Thread(target=work, daemon=True).start()

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
        out = []
        if self.vi.get():
            out.append("vi")
        if self.es.get():
            out.append("es")
        if self.idn.get():
            out.append("id")
        if self.ms.get():
            out.append("ms")
        if self.en.get():
            out.append("en")
        return ",".join(out)

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
        cfg = {"videos": videos, "out": OUT, "ffmpeg": self.ffmpeg, "work": WORK,
               "langs": langs, "model": self.model.get(),
               "keepmusic": self.keepmusic.get(), "cover": self.cover.get(),
               "band": self.band.get(), "fb": self.fb.get(),
               "srt": self.srt.get(), "naming": "firstline", "burn": self.burn.get(),
               "tone": self.tone.get(), "audioonly": self.audioonly.get(),
               "clone": self.clone.get(),
               "gender": self.gender.get(),
               "scenefx": self.scenefx.get(),
               "deepseek_key": self.deepseek_key.get().strip(),
               "voices": {lg: self._voice_id(lg) for lg in langs.split(",") if lg}}
        try:
            with open(JOBS, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write job file:\n{e}")
            return
        self._refresh_queue()
        self.running = True
        self.cancel = False
        self.run_btn.config(state="disabled", text="Working...")
        self.stop_btn.config(state="normal")
        self.status.config(text="Working...", fg="#a60")
        threading.Thread(target=self._worker, daemon=True).start()

    def _stop(self):
        self.cancel = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.status.config(text="Stopping...", fg="#a00")

    def _worker(self):
        cmd = [sys.executable, "-u", os.path.join(BASE, "dub.py"), "--jobs", JOBS]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, encoding="utf-8", errors="replace",
                                 creationflags=_NOWIN)
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
                elif kind == "done":
                    self._on_run_done()
        except queue.Empty:
            pass
        self.root.after(120, self._pump)

    def _on_run_done(self):
        self.running = False
        self.proc = None
        self.run_btn.config(state="normal", text="DUB ALL")
        self.stop_btn.config(state="disabled")
        if self.cancel:
            self.status.config(text="Stopped", fg="#a00")
            self.stage_lbl.config(text="Stopped")
            return
        self.status.config(text="Done", fg="#070")
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
