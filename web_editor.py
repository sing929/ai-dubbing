#!/usr/bin/env python3
"""
Browser-based dubbing editor (Phase 1).

A dark, side-by-side script editor in the spirit of the commercial dubbing tools:
  - Speaker | Source | Translation rows, aligned per subtitle line
  - Original / Translated video (or audio) preview
  - A colour-coded segment timeline that seeks the player
  - An AI assistant (DeepSeek) to correct the script / tune the translation /
    back-translate for a meaning check

It is deliberately dependency-light: only Python's stdlib plus `deep_translator`
(already installed for dub.py). It does NOT `import dub`, so it never pulls in
torch / faster-whisper and starts instantly.

Data sources (read from the existing project, no re-dub needed):
  - Source segments: _audio_work/_segs_<name>.json  ({start,end,text,lang} per line)
  - Translations:    computed on demand via Google translate, cached next to the
                     segs file as _audio_work/_tr_<name>_<lang>.json (and editable)
  - Original audio:  _audio_work/<name>.wav   (extracted source track)
  - Translated video: newest matching dubbed/*.mp4

Run:  python web_editor.py        (opens http://127.0.0.1:8765 in your browser)
"""
from __future__ import annotations

import glob
import hashlib
import html
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_audio_work")
OUT = os.path.join(HERE, "dubbed")
HTML_PATH = os.path.join(HERE, "editor.html")
SETTINGS = os.path.join(HERE, "_dubapp_settings.json")  # shared with dub_app.py

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Folders the /media endpoint is allowed to serve from (everything else is denied).
def _safe_roots() -> list[str]:
    roots = [WORK, OUT, HERE]
    # The last job's source video usually lives outside the project (e.g. Desktop).
    try:
        jobs = json.load(open(os.path.join(WORK, "_jobs.json"), encoding="utf-8"))
        for v in jobs.get("videos", []):
            d = os.path.dirname(v.get("path", ""))
            if d and os.path.isdir(d):
                roots.append(d)
        home = os.path.expanduser("~")
        for sub in ("Desktop", "Downloads", "Videos"):
            p = os.path.join(home, sub)
            if os.path.isdir(p):
                roots.append(p)
    except Exception:
        pass
    # Trust the source/output folders recorded in any project file (a source video
    # may live outside the common dirs above).
    try:
        for pf in glob.glob(os.path.join(OUT, "*.dubproj.json")):
            d = json.load(open(pf, encoding="utf-8"))
            for k in ("source", "output"):
                dd = os.path.dirname(d.get(k, ""))
                if dd and os.path.isdir(dd):
                    roots.append(dd)
    except Exception:
        pass
    return [os.path.normcase(os.path.abspath(r)) for r in roots]


SAFE_ROOTS = _safe_roots()


def _allowed(path: str) -> bool:
    ap = os.path.normcase(os.path.abspath(path))
    return any(ap == r or ap.startswith(r + os.sep) for r in SAFE_ROOTS)


# ---------------------------------------------------------------- project data
def _segs_files() -> list[str]:
    return sorted(glob.glob(os.path.join(WORK, "_segs_*.json")), key=os.path.getmtime, reverse=True)


def _base_of(segs_file: str) -> str:
    """'_segs_download (1).json' -> 'download (1)'."""
    b = os.path.basename(segs_file)
    return b[len("_segs_"):-len(".json")]


def _job_langs() -> list[str]:
    try:
        jobs = json.load(open(os.path.join(WORK, "_jobs.json"), encoding="utf-8"))
        langs = jobs.get("langs", "en")
        return [x.strip() for x in re.split(r"[,\s]+", langs) if x.strip()] or ["en"]
    except Exception:
        return ["en"]


def _find_source_media(base: str) -> str | None:
    """Original track for the 'Original' tab: prefer the extracted <base>.wav, else
    a same-named source video in any safe root."""
    wav = os.path.join(WORK, base + ".wav")
    if os.path.exists(wav):
        return wav
    for root in SAFE_ROOTS:
        for ext in (".mp4", ".mkv", ".mov", ".webm", ".wav", ".mp3"):
            cand = os.path.join(root, base + ext)
            if os.path.exists(cand):
                return cand
    return None


def _newest_output() -> str | None:
    vids = glob.glob(os.path.join(OUT, "*.mp4"))
    return max(vids, key=os.path.getmtime) if vids else None


LANG_NAMES = {
    "zh": "Chinese", "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish", "ru": "Russian",
    "nl": "Dutch", "cs": "Czech", "ar": "Arabic", "zh-CN": "Chinese", "hu": "Hungarian",
    "ko": "Korean", "ja": "Japanese", "hi": "Hindi", "id": "Indonesian", "vi": "Vietnamese",
    "ms": "Malay",
}


def _tr_cache_path(base: str, lang: str) -> str:
    safe = re.sub(r"[^\w.-]", "_", f"{base}_{lang}")
    return os.path.join(WORK, f"_tr_{safe}.json")


def _translate_lines(lines: list[str], lang: str) -> list[str]:
    """Google-translate source lines to `lang`. Best-effort, line by line so one bad
    line can't blank the rest. Mirrors dub.py's source='auto' approach."""
    from deep_translator import GoogleTranslator
    out: list[str] = []
    tgt = "zh-CN" if lang == "zh" else lang
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            out.append("")
            continue
        try:
            out.append(GoogleTranslator(source="auto", target=tgt).translate(s) or "")
        except Exception:
            out.append("")
    return out


def load_project(base: str, lang: str) -> dict:
    segs_file = os.path.join(WORK, f"_segs_{base}.json")
    segs = json.load(open(segs_file, encoding="utf-8"))
    src_lang = segs[0].get("lang", "auto") if segs else "auto"

    # translations (cached + editable)
    cache_path = _tr_cache_path(base, lang)
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}
    tr = cache.get("tr") or {}
    missing = [i for i, s in enumerate(segs) if str(i) not in tr]
    if missing:
        fresh = _translate_lines([segs[i]["text"] for i in missing], lang)
        for i, t in zip(missing, fresh):
            tr[str(i)] = t
        json.dump({"tr": tr}, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    rows = []
    for i, s in enumerate(segs):
        rows.append({
            "i": i,
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "src": s.get("text", ""),
            "tr": tr.get(str(i), ""),
            "speaker": s.get("speaker", 0),
        })

    out_media = _newest_output()
    src_media = _find_source_media(base)
    poster = os.path.join(WORK, "_preview.png")
    return {
        "id": "segs:" + base,
        "kind": "segs",
        "base": base,
        "src_lang": src_lang,
        "src_lang_name": LANG_NAMES.get(src_lang, src_lang.title()),
        "lang": lang,
        "lang_name": LANG_NAMES.get(lang, lang.title()),
        "rows": rows,
        "duration": rows[-1]["end"] if rows else 0.0,
        "original_url": "/media?path=" + urllib.parse.quote(src_media) if src_media else "",
        "translated_url": "/media?path=" + urllib.parse.quote(out_media) if out_media else "",
        "original_name": os.path.basename(src_media) if src_media else "",
        "translated_name": os.path.basename(out_media) if out_media else "",
        "poster_url": "/media?path=" + urllib.parse.quote(poster) if os.path.exists(poster) else "",
    }


def save_translations(base: str, lang: str, tr: dict) -> None:
    cache_path = _tr_cache_path(base, lang)
    json.dump({"tr": {str(k): v for k, v in tr.items()}}, open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False)


# ------------------------------------------------------------ project files
# dub.py (Phase 2) writes dubbed/<output>.dubproj.json per run, holding the exact
# source/output pairing + per-line src/translation/speaker/voice. These are richer
# and correctly paired, so we list them first; bases with only a _segs cache (older
# dubs) fall back to the on-the-fly translation path.
def _proj_files() -> list[str]:
    return sorted(glob.glob(os.path.join(OUT, "*.dubproj.json")), key=os.path.getmtime, reverse=True)


def list_projects() -> list[dict]:
    items, proj_bases = [], set()
    for f in _proj_files():
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        base, lang = d.get("base", "?"), d.get("lang", "?")
        proj_bases.add(base)
        items.append({"id": "proj:" + os.path.basename(f),
                      "label": f"{base} · {LANG_NAMES.get(lang, lang).upper()}",
                      "kind": "proj", "lang": lang, "mtime": os.path.getmtime(f)})
    for f in _segs_files():
        base = _base_of(f)
        if base in proj_bases:
            continue
        items.append({"id": "segs:" + base, "label": f"{base} · (translate)",
                      "kind": "segs", "lang": None, "mtime": os.path.getmtime(f)})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


def load_proj_file(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    segs = d.get("segments", [])
    rows = [{"i": s["i"], "start": s["start"], "end": s["end"], "src": s["src"],
             "tr": s.get("tr", ""), "speaker": s.get("speaker", 0),
             "role": s.get("role"), "gender": s.get("gender"), "voice": s.get("voice")}
            for s in segs]
    src, out = d.get("source"), d.get("output")
    src_ok = bool(src and os.path.exists(src) and _allowed(src))
    out_ok = bool(out and os.path.exists(out) and _allowed(out))
    lang, slang = d.get("lang", "en"), d.get("src_lang", "auto")
    poster = os.path.join(WORK, "_preview.png")
    nspk = len({r["speaker"] for r in rows})
    return {
        "id": "proj:" + os.path.basename(path),
        "kind": "proj",
        "base": d.get("base", ""),
        "story_bible": d.get("story_bible") or {},
        "src_lang": slang, "src_lang_name": LANG_NAMES.get(slang, slang.title()),
        "lang": lang, "lang_name": LANG_NAMES.get(lang, lang.title()),
        "rows": rows,
        "duration": d.get("duration") or (rows[-1]["end"] if rows else 0.0),
        "speakers": nspk,
        "original_url": "/media?path=" + urllib.parse.quote(src) if src_ok else "",
        "translated_url": "/media?path=" + urllib.parse.quote(out) if out_ok else "",
        "original_name": os.path.basename(src) if src else "",
        "translated_name": os.path.basename(out) if out else "",
        "poster_url": "/media?path=" + urllib.parse.quote(poster) if os.path.exists(poster) else "",
    }


def save_proj_file(path: str, tr: dict) -> None:
    d = json.load(open(path, encoding="utf-8"))
    for s in d.get("segments", []):
        key = str(s["i"])
        if key in tr:
            s["tr"] = tr[key]
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def deepseek(system: str, user: str, key: str) -> tuple[str | None, str | None]:
    body = json.dumps({"model": DEEPSEEK_MODEL, "temperature": 1.0, "stream": False,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode("utf-8")
    req = urllib.request.Request(DEEPSEEK_URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, f"DeepSeek error: {str(e)[:120]}"


def _saved_key() -> str:
    """The DeepSeek key. Prefer the DEEPSEEK_API_KEY env var (the secure place to
    keep a secret); fall back to a key a previous GUI/jobs run wrote to disk so
    existing setups keep working. New keys are no longer written to disk when the
    env var is set (see save_settings)."""
    env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env:
        return env
    for p in (os.path.join(HERE, "_dubapp_settings.json"), os.path.join(WORK, "_jobs.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
            k = (d.get("deepseek_key") or "").strip()
            if k:
                return k
        except Exception:
            pass
    return ""


# ---------------------------------------------------------- dub job runner
# Import (dub a new video) and Re-dub (re-voice edited translations) both drive
# the SAME pipeline the GUI uses: `python dub.py --jobs <cfg.json>`. We don't
# reimplement dubbing; we reuse it and stream its __STAGE__/__PCT__/__OUT__ log.
JOB = {"proc": None, "lines": [], "done": True, "error": None, "label": ""}
_NEW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _find_ffmpeg() -> str:
    cands = glob.glob(os.path.join(HERE, "ffmpeg*", "bin", "ffmpeg.exe"))
    return cands[0] if cands else "ffmpeg"


def _base_cfg() -> dict:
    """Start from the GUI's last settings (clone/gender/speakers/voices/key/langs)
    so an Import reproduces the workflow the user already configured."""
    try:
        cfg = json.load(open(os.path.join(WORK, "_jobs.json"), encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg.setdefault("out", OUT)
    cfg.setdefault("work", WORK)
    cfg.setdefault("ffmpeg", _find_ffmpeg())
    cfg.setdefault("langs", ",".join(_job_langs()))
    cfg.setdefault("model", "medium")
    cfg.setdefault("naming", "firstline")
    cfg["reuse_analysis"] = True
    cfg.pop("texts_override", None)
    return cfg


def _pick_video() -> str:
    """Open a native OS file dialog in a throwaway process (a browser can't hand us
    a real local path, and re-uploading multi-GB video would be absurd)."""
    code = (
        "import tkinter as tk, tkinter.filedialog as fd\n"
        "r=tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "print(fd.askopenfilename(title='Choose a video to dub',"
        " filetypes=[('Video','*.mp4 *.mkv *.mov *.webm *.avi *.m4v'),('All files','*.*')]))\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300, creationflags=_NEW)
        return r.stdout.strip()
    except Exception:
        return ""


def build_import_cfg(path: str) -> dict:
    cfg = _base_cfg()
    cfg["videos"] = [{"path": path, "region": ""}]
    return cfg


# ---------------------------------------------------- full Dub page (home)
# The home page reproduces every dub_app.py control. We reuse dub_app's constant
# tables (voices/languages/samples) so there's ONE source of truth, and we read/
# write the SAME _dubapp_settings.json the tkinter GUI uses, so the two are linked.
def _dub_app():
    import importlib
    return importlib.import_module("dub_app")  # safe: no Tk root at import, dnd guarded


def load_settings() -> dict:
    try:
        return json.load(open(SETTINGS, encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> dict:
    cur = load_settings()
    cur.update(data or {})
    try:
        # Don't persist the API key in plaintext when the env var already provides
        # it - that's the secret's proper home. (If there's no env var we still
        # store it, so pasting a key into the GUI keeps working as before.)
        to_write = dict(cur)
        if os.environ.get("DEEPSEEK_API_KEY", "").strip():
            to_write.pop("deepseek_key", None)
        json.dump(to_write, open(SETTINGS, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception as e:
        print(f"  [warn] could not save settings: {e}", flush=True)
    return cur


# The small JSON sidecars in _audio_work hold the slow-to-recompute analysis
# (transcription, segments, translations) and must survive a cache clear, so a
# later re-dub into another language stays fast. Everything else under
# _audio_work - the big extracted WAVs, Demucs stems, per-segment TTS - is a pure
# intermediate that the next dub regenerates, so it's safe to reclaim.
_CACHE_KEEP_EXT = (".json",)


def _iter_cache_files():
    """Yield (path, size, is_disposable) for every file under _audio_work."""
    for root, _dirs, files in os.walk(WORK):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            keep = fn.lower().endswith(_CACHE_KEEP_EXT)
            yield p, size, (not keep)


def cache_stats() -> dict:
    """How much the working cache is using, and how much a clear would reclaim."""
    total = reclaimable = files = 0
    for _p, size, disposable in _iter_cache_files():
        total += size
        files += 1
        if disposable:
            reclaimable += size
    return {"total_bytes": total, "reclaimable_bytes": reclaimable, "files": files}


def clean_work_cache() -> dict:
    """Delete the regenerable intermediates (big WAVs, Demucs/TTS output), keeping
    the small analysis JSON so language re-dubs stay fast. Returns bytes freed."""
    freed = removed = 0
    for p, size, disposable in list(_iter_cache_files()):
        if not disposable:
            continue
        try:
            os.remove(p)
            freed += size
            removed += 1
        except OSError:
            pass
    # Drop any directories left empty by the sweep (htdemucs/<base>, tts_<base>...).
    for root, dirs, files in os.walk(WORK, topdown=False):
        if root == WORK:
            continue
        if not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:
                pass
    return {"freed_bytes": freed, "removed": removed}


def dub_meta() -> dict:
    try:
        da = _dub_app()
        meta = {"lang_order": da.LANG_ORDER, "voice_choices": da.VOICE_CHOICES,
                "default_voice": da.DEFAULT_VOICE, "sample_text": da.SAMPLE_TEXT}
    except Exception as e:
        meta = {"lang_order": [["en", "English"]], "voice_choices": {},
                "default_voice": {}, "sample_text": {}, "meta_error": str(e)}
    meta["models"] = ["tiny", "base", "small", "medium"]
    meta["settings"] = load_settings()
    return meta


def _pick_videos() -> list[str]:
    code = (
        "import tkinter as tk, tkinter.filedialog as fd, json, sys\n"
        "r=tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "fs=fd.askopenfilenames(title='Add videos to dub',"
        " filetypes=[('Video','*.mp4 *.mov *.mkv *.webm *.avi *.ts *.m4v'),('All files','*.*')])\n"
        "sys.stdout.write(json.dumps(list(fs)))\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300, creationflags=_NEW)
        return json.loads(r.stdout.strip() or "[]")
    except Exception:
        return []


def _region_str(r) -> str:
    return ",".join(f"{float(v):.4f}" for v in r) if r else ""


def build_dub_cfg(s: dict, queue: list[dict]) -> dict:
    """Mirror dub_app._start: assemble the jobs cfg from settings + queue."""
    try:
        default_voice = _dub_app().DEFAULT_VOICE
    except Exception:
        default_voice = {}
    lang = s.get("lang") or "en"
    voices = s.get("voices") or {}
    speakers = bool(s.get("speakers"))
    preset = s.get("preset") or ""
    rights_mode = s.get("rights_mode") or ""
    facebook_reels = preset == "facebook_reels"
    if facebook_reels:
        speakers = True
        rights_mode = rights_mode or "owned_or_licensed"
    return {
        "videos": [{"path": it["path"], "region": _region_str(it.get("region"))} for it in queue],
        "out": OUT, "work": WORK, "ffmpeg": _find_ffmpeg(),
        "langs": lang, "model": s.get("model", "medium"),
        "preset": preset, "rights_mode": rights_mode,
        "keepmusic": bool(s.get("keepmusic")) or facebook_reels,
        "cover": bool(s.get("cover", True)) or facebook_reels,
        "band": int(s.get("band", 18)), "fb": bool(s.get("fb")) or facebook_reels,
        "srt": bool(s.get("srt")) or facebook_reels, "naming": "firstline",
        "burn": bool(s.get("burn")) or facebook_reels,
        "tone": s.get("tone", "original"), "audioonly": bool(s.get("audioonly")),
        "clone": bool(s.get("clone")) and not speakers and not facebook_reels,
        "gender": bool(s.get("gender")) or facebook_reels,
        "speakers": speakers, "scenefx": bool(s.get("scenefx")),
        "reuse_analysis": bool(s.get("reuse_analysis", True)),
        "length_fit": bool(s.get("length_fit")) or facebook_reels,
        # Leave the key out of the on-disk jobs file when the env var is set; the
        # dub subprocess inherits DEEPSEEK_API_KEY and dub.py falls back to it.
        "deepseek_key": "" if os.environ.get("DEEPSEEK_API_KEY", "").strip()
                        else (s.get("deepseek_key") or "").strip(),
        "voices": {lang: voices.get(lang) or default_voice.get(lang, "")},
    }


def voice_preview(lang: str, voice: str) -> str | None:
    """Synthesize the per-language sample line with edge-tts (same lib dub.py uses)."""
    try:
        import asyncio
        import edge_tts
        da = _dub_app()
        text = da.SAMPLE_TEXT.get(lang) or "Hello, this is a sample of this voice."
        out = os.path.join(WORK, "_voice_preview_web.mp3")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(edge_tts.Communicate(text, voice).save(out))
        finally:
            loop.close()
        return out if os.path.exists(out) else None
    except Exception as e:
        print(f"  [warn] voice preview failed: {e}", flush=True)
        return None


def _stop_job() -> None:
    p = JOB.get("proc")
    if p is not None and p.poll() is None:
        try:
            p.terminate()
        except Exception:
            pass


def build_redub_cfg(proj_file: str, tr: dict | None = None) -> tuple[dict | None, dict | None, str | None]:
    """Build a jobs cfg that re-voices a project with its (edited) translations.
    Returns (cfg, proj_dict, error)."""
    d = json.load(open(proj_file, encoding="utf-8"))
    if tr:
        save_proj_file(proj_file, {str(k): v for k, v in tr.items()})
        d = json.load(open(proj_file, encoding="utf-8"))
    src = d.get("source", "")
    if not (src and os.path.exists(src) and _allowed(src)):
        return None, d, f"Original video not found: {src}"
    lang = d.get("lang", "en")
    cfg = _base_cfg()
    cfg["videos"] = [{"path": src, "region": ""}]
    cfg["langs"] = lang
    cfg["naming"] = "firstline"
    cfg["audioonly"] = bool(d.get("audio_only"))
    cfg["preset"] = d.get("preset", cfg.get("preset", ""))
    cfg["rights_mode"] = d.get("rights_mode", cfg.get("rights_mode", ""))
    cfg["texts_override"] = {lang: [s.get("tr", "") for s in d.get("segments", [])]}
    return cfg, d, None


def _start_job(cfg: dict, label: str) -> tuple[bool, str | None]:
    if JOB["proc"] is not None and JOB["proc"].poll() is None:
        return False, "A dub is already running."
    jobs_path = os.path.join(WORK, "_jobs_editor.json")
    json.dump(cfg, open(jobs_path, "w", encoding="utf-8"), ensure_ascii=False)
    JOB.update(proc=None, lines=[], done=False, error=None, label=label)
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "dub.py"), "--jobs", jobs_path],
                         cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", creationflags=_NEW)
    JOB["proc"] = p

    def reader():
        for line in p.stdout:
            JOB["lines"].append(line.rstrip("\n"))
            if len(JOB["lines"]) > 600:
                JOB["lines"] = JOB["lines"][-400:]
        p.wait()
        JOB["done"] = True

    threading.Thread(target=reader, daemon=True).start()
    return True, None


# ---------------------------------------------------------------- HTTP handler
class Handler(BaseHTTPRequestHandler):
    server_version = "DubEditor/1.0"

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---- GET ----
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/" or u.path == "/index.html":
                return self._send_html()
            if u.path == "/api/projects":
                return self._json({"projects": list_projects(), "langs": _job_langs(),
                                   "lang_names": LANG_NAMES, "has_key": bool(_saved_key())})
            if u.path == "/api/project":
                pid = q.get("id", [""])[0]
                lang = q.get("lang", ["en"])[0]
                if not pid:
                    items = list_projects()
                    if not items:
                        return self._json({"error": "No projects found. Dub something first."}, 404)
                    pid = items[0]["id"]
                if pid.startswith("proj:"):
                    return self._json(load_proj_file(os.path.join(OUT, pid[5:])))
                base = pid[5:] if pid.startswith("segs:") else pid
                return self._json(load_project(base, lang))
            if u.path == "/api/meta":
                return self._json(dub_meta())
            if u.path == "/api/cache":
                return self._json(cache_stats())
            if u.path == "/api/job":
                running = JOB["proc"] is not None and JOB["proc"].poll() is None
                return self._json({"running": running, "done": JOB["done"],
                                   "label": JOB["label"], "lines": JOB["lines"][-60:]})
            if u.path == "/media":
                return self._serve_media(q.get("path", [""])[0])
            self.send_error(404)
        except BrokenPipeError:
            pass
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ---- POST ----
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/upload":
                # Browser file upload: raw bytes in the body, filename in ?name=.
                # Reliable everywhere (no native dialog), and on localhost it's just a
                # disk copy. Saved under uploads/ then queued by its real path.
                #
                # Browsers name almost every download "download.mp4", so uploading a
                # different clip used to overwrite the previous one AND collide on the
                # work-file name "download" - the dub then reused the old video's
                # separated voice. We make the saved name content-addressed instead:
                # the same video always maps to the same name (so the analysis cache
                # still reuses correctly), but two different videos never share one.
                qs = urllib.parse.parse_qs(u.query)
                name = os.path.basename(qs.get("name", ["video"])[0]) or "video"
                stem, ext = os.path.splitext(name)
                updir = os.path.join(HERE, "uploads")
                os.makedirs(updir, exist_ok=True)
                tmp = os.path.join(updir, f".incoming_{os.getpid()}_{int(time.time()*1000)}")
                remaining = int(self.headers.get("Content-Length", 0))
                digest = hashlib.sha1()
                with open(tmp, "wb") as f:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                short = digest.hexdigest()[:8]
                name = f"{stem}_{short}{ext}"
                dest = os.path.join(updir, name)
                os.replace(tmp, dest)          # atomic; re-uploading the same clip just overwrites itself
                return self._json({"path": os.path.abspath(dest), "name": name})
            if u.path == "/api/save":
                b = self._body()
                pid = b.get("id", "")
                tr = {str(k): v for k, v in b.get("tr", {}).items()}
                if pid.startswith("proj:"):
                    save_proj_file(os.path.join(OUT, pid[5:]), tr)
                else:
                    save_translations(b["base"], b["lang"], tr)
                return self._json({"ok": True})
            if u.path == "/api/translate":
                b = self._body()
                return self._json({"tr": _translate_lines(b.get("lines", []), b.get("lang", "en"))})
            if u.path == "/api/deepseek":
                b = self._body()
                key = (b.get("key") or "").strip() or _saved_key()
                if not key:
                    return self._json({"error": "No DeepSeek API key found."}, 400)
                out, err = deepseek(b.get("system", ""), b.get("user", ""), key)
                return self._json({"error": err} if err else {"reply": out})
            if u.path == "/api/settings":
                return self._json({"settings": save_settings(self._body())})
            if u.path == "/api/add_videos":
                return self._json({"paths": _pick_videos()})
            if u.path == "/api/voice_preview":
                b = self._body()
                out = voice_preview(b.get("lang", "en"), b.get("voice", ""))
                if not out:
                    return self._json({"error": "Preview failed (edge-tts/voice?)."}, 400)
                return self._json({"url": "/media?path=" + urllib.parse.quote(out)})
            if u.path == "/api/stop":
                _stop_job()
                return self._json({"ok": True})
            if u.path == "/api/cleanup":
                if JOB["proc"] is not None and JOB["proc"].poll() is None:
                    return self._json({"error": "A dub is running; stop it first."}, 409)
                return self._json(clean_work_cache())
            if u.path == "/api/dub":
                b = self._body()
                s = b.get("settings", {})
                queue = b.get("queue", [])
                if not queue:
                    return self._json({"error": "Add at least one video."}, 400)
                if not (s.get("lang")):
                    return self._json({"error": "Pick a target language."}, 400)
                save_settings({**s, "queue": queue})
                ok, err = _start_job(build_dub_cfg(s, queue),
                                     f"Dubbing {len(queue)} video(s) → {s.get('lang')}")
                return self._json({"started": ok, "error": err})
            if u.path == "/api/import":
                path = _pick_video()
                if not path or not os.path.exists(path):
                    return self._json({"cancelled": True})
                cfg = build_import_cfg(path)
                ok, err = _start_job(cfg, "Dubbing " + os.path.basename(path))
                return self._json({"started": ok, "error": err,
                                   "name": os.path.basename(path),
                                   "langs": cfg.get("langs", "")})
            if u.path == "/api/redub":
                b = self._body()
                pid = b.get("id", "")
                if not pid.startswith("proj:"):
                    return self._json({"error": "Re-dub needs a real dubbed project "
                                                "(dub it once in the app to create one)."}, 400)
                cfg, d, err = build_redub_cfg(os.path.join(OUT, pid[5:]), b.get("tr"))
                if err:
                    return self._json({"error": err}, 400)
                ok, serr = _start_job(cfg, "Re-dubbing " + d.get("base", ""))
                return self._json({"started": ok, "error": serr})
            self.send_error(404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ---- helpers ----
    def _send_html(self):
        try:
            data = open(HTML_PATH, "rb").read()
        except FileNotFoundError:
            self.send_error(500, "editor.html missing next to web_editor.py")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_media(self, path: str):
        path = urllib.parse.unquote(path)
        if not path or not os.path.exists(path) or not _allowed(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                start = max(0, start)
                end = min(end, size - 1)
                partial = True
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _kill_port(port: int) -> None:
    """Terminate a stale process still holding our fixed port (almost always a
    previous editor instance), so relaunching always replaces it on the SAME URL.
    Without this, an old build keeps answering on :8765 with missing endpoints."""
    if os.name != "nt":
        return
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                             text=True, creationflags=_NEW).stdout
    except Exception:
        return
    me = str(os.getpid())
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
            pid = parts[4]
            if pid not in ("0", me):
                try:
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   capture_output=True, creationflags=_NEW)
                except Exception:
                    pass


def _bind(port: int) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    p = s.getsockname()[1]
    s.close()
    return p


def _free_port(preferred=8765) -> int:
    try:
        return _bind(preferred)
    except OSError:
        _kill_port(preferred)          # stale old editor? take the port back
        time.sleep(0.6)
    try:
        return _bind(preferred)
    except OSError:
        return _bind(0)                # last resort: a random free port


def main():
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Dub editor running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        srv.shutdown()


if __name__ == "__main__":
    main()
