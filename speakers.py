#!/usr/bin/env python3
# speakers.py - OPTIONAL multi-speaker diarization for dub.py.
#
# Groups the transcribed segments by WHO is speaking, so each distinct speaker can
# be given their own voice (or be voice-cloned from their own lines) instead of one
# voice for the whole video. Mirrors voiceclone.py / xtts.py: it's an add-on - if the
# audio libraries aren't importable, available() is False and dub.py falls back to the
# single-voice / gender path, so the tool still runs everywhere.
#
# It deliberately uses only libraries the base tool already ships (librosa + scikit-
# learn + numpy) - no heavy neural speaker-embedding model and no native build. Each
# segment gets a compact voice "fingerprint" (MFCC mean/std + median pitch), and the
# segments are clustered into speakers; the speaker COUNT is picked automatically by
# silhouette score. This separates clearly different voices (the usual male-lead /
# female-lead / side-character mix in short dramas) well; it can't split two very
# similar same-gender voices the way a neural model would. The interface (diarize ->
# per-segment integer labels) is the part to keep if a stronger backend is swapped in.

import os


def available() -> bool:
    """True if the libraries needed to fingerprint and cluster voices are present.
    This is the always-on FLOOR (MFCC+pitch backend); the neural pyannote backend is
    an optional upgrade on top - see neural_available()."""
    try:
        import librosa, numpy, sklearn  # noqa: F401
        return True
    except Exception:
        return False


def _hf_token() -> str:
    """The HuggingFace access token pyannote needs to download its gated pipeline.
    Read from the usual env vars; empty string when none is set."""
    import os
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN",
              "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""


def neural_available() -> bool:
    """True if the neural pyannote backend can run: pyannote.audio + torch importable
    AND an HF token is available (the speaker-diarization pipeline is gated). When
    False we silently use the MFCC+pitch backend, so nothing breaks without it."""
    if not _hf_token():
        return False
    try:
        import torch  # noqa: F401
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


# pyannote model id. 3.1 is the current community pipeline; override with env
# PYANNOTE_PIPELINE to pin a different revision or a locally-cached copy.
_PYANNOTE_PIPELINE = "pyannote/speaker-diarization-3.1"


def _diarize_pyannote(audio_wav: str, segs: list[dict]) -> list[int] | None:
    """Neural diarization via pyannote.audio. Runs the pretrained diarization
    pipeline on `audio_wav`, then assigns each segment in `segs` the speaker whose
    turns OVERLAP it most (the pipeline finds its own speaker turns independent of our
    transcription boundaries, so we project them onto our segments by time-overlap).
    Returns per-segment integer labels in first-spoken order, or None on any failure
    so the caller can fall back to the MFCC backend. Never raises."""
    n = len(segs)
    if n == 0:
        return []
    try:
        import os
        import torch
        from pyannote.audio import Pipeline
    except Exception:
        return None
    token = _hf_token()
    if not token:
        return None
    pipe_id = os.environ.get("PYANNOTE_PIPELINE", _PYANNOTE_PIPELINE)
    try:
        pipeline = Pipeline.from_pretrained(pipe_id, use_auth_token=token)
        if pipeline is None:                       # bad token / un-accepted licence
            return None
        try:                                       # GPU when present; harmless on CPU
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:
            pass
        diarization = pipeline(audio_wav)
    except Exception as e:
        print(f"  pyannote diarization failed ({str(e)[:60]}); using MFCC backend.",
              flush=True)
        return None

    # Collect (start, end, speaker) turns, then for each seg pick the speaker with the
    # greatest total temporal overlap.
    try:
        turns = [(float(t.start), float(t.end), str(spk))
                 for t, _, spk in diarization.itertracks(yield_label=True)]
    except Exception:
        return None
    if not turns:
        return [0] * n

    raw: list[str | None] = []
    for s in segs:
        a = float(s.get("start", 0.0))
        b = float(s.get("end", a))
        best_spk, best_ov = None, 0.0
        for (ts, te, spk) in turns:
            ov = min(b, te) - max(a, ts)
            if ov > best_ov:
                best_ov, best_spk = ov, spk
        raw.append(best_spk)
    # A seg with no overlap (silence/blip) inherits the previous speaker, else the next.
    for i in range(n):
        if raw[i] is None:
            raw[i] = (raw[i - 1] if i > 0 and raw[i - 1] is not None
                      else next((r for r in raw[i + 1:] if r is not None), "SPK0"))
    if len(set(raw)) <= 1:
        return [0] * n
    # Relabel so ids appear in first-spoken order (0,1,2,... down the timeline).
    order: dict[str, int] = {}
    for r in raw:
        if r not in order:
            order[r] = len(order)
    return [order[r] for r in raw]


def _fingerprint(clip, sr, np, librosa):
    """A small fixed-length vector that characterises one speaker's voice in `clip`."""
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
    parts = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    try:
        f0 = librosa.yin(clip, fmin=70, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        parts.append([float(np.median(f0)) if f0.size else 0.0])
    except Exception:
        parts.append([0.0])
    return np.concatenate([np.asarray(p, dtype="float32").ravel() for p in parts])


def diarize(audio_wav: str, segs: list[dict], max_speakers: int = 6,
            min_silhouette: float = 0.18, backend: str = "auto") -> list[int] | None:
    """Return a per-segment speaker index (0..k-1), aligned 1:1 with `segs`, or None
    if diarization can't run. The number of speakers k is chosen automatically; when
    no clear split exists every segment gets 0.

    backend:
      "auto"     - use the neural pyannote backend when it's installed AND an HF token
                   is set (best at splitting similar same-gender voices); otherwise the
                   built-in MFCC+pitch backend. Env SPEAKERS_BACKEND overrides this.
      "pyannote" - force neural; falls back to MFCC if pyannote fails at runtime.
      "mfcc"     - force the dependency-free MFCC+pitch backend.

    audio_wav should be the ISOLATED voice (Demucs `vocals.wav`) for best results -
    music/effects in a raw mix blur both backends.
    """
    import os
    backend = (os.environ.get("SPEAKERS_BACKEND") or backend or "auto").lower()
    use_neural = backend == "pyannote" or (backend == "auto" and neural_available())
    if use_neural:
        labels = _diarize_pyannote(audio_wav, segs)
        if labels is not None:
            return labels
        # pyannote unavailable/failed -> fall through to the MFCC backend.
    return _diarize_fingerprint(audio_wav, segs, max_speakers, min_silhouette)


def _diarize_fingerprint(audio_wav: str, segs: list[dict], max_speakers: int = 6,
                         min_silhouette: float = 0.18) -> list[int] | None:
    """MFCC+pitch fingerprint + agglomerative clustering backend (the original,
    dependency-free path). See module header for its strengths/limits."""
    n = len(segs)
    if n == 0:
        return []
    if n <= 2:
        return [0] * n
    try:
        import numpy as np
        import librosa
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except Exception:
        return None
    try:
        y, sr = librosa.load(audio_wav, sr=16000, mono=True)
    except Exception:
        return None
    if y.size == 0:
        return [0] * n

    dur = y.size / sr
    feats = []
    floor = int(0.30 * sr)            # pad ultra-short lines so MFCC/pitch are stable
    for s in segs:
        a = int(max(0.0, s.get("start", 0.0)) * sr)
        b = int(min(dur, s.get("end", 0.0)) * sr)
        clip = y[a:b] if b > a else y[a:a + floor]
        if clip.size < floor:
            clip = np.pad(clip, (0, floor - clip.size))
        try:
            feats.append(_fingerprint(clip, sr, np, librosa))
        except Exception:
            feats.append(np.zeros_like(feats[0]) if feats else np.zeros(41, dtype="float32"))

    try:
        X = StandardScaler().fit_transform(np.vstack(feats))
    except Exception:
        return None

    # Pick the speaker count with the strongest, most separated clustering.
    best = (1, [0] * n, -1.0)
    upper = min(max_speakers, n - 1)
    for k in range(2, upper + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            score = silhouette_score(X, labels)
        except Exception:
            continue
        if score > best[2]:
            best = (k, list(labels), score)

    k, labels, score = best
    if k == 1 or score < min_silhouette:
        return [0] * n          # no convincing split -> treat as a single speaker

    # Drop "phantom" speakers: a single narrator's emphasis / laughter / music-bleed
    # lines can cluster off on their own. A real recurring speaker should hold a fair
    # share of the lines, so merge any cluster below min_share into the nearest big
    # cluster (by centroid). This keeps narration as ONE voice instead of splitting it.
    labels = np.asarray(labels)
    min_share = max(3, int(0.12 * n))
    centroids = {c: X[labels == c].mean(axis=0) for c in set(labels.tolist())}
    big = [c for c in centroids if int((labels == c).sum()) >= min_share]
    if not big:
        return [0] * n
    if len(big) < len(centroids):
        big_arr = {c: centroids[c] for c in big}
        for i in range(n):
            if labels[i] not in big:
                labels[i] = min(big_arr, key=lambda c: float(np.linalg.norm(X[i] - big_arr[c])))
    if len(set(labels.tolist())) <= 1:
        return [0] * n          # everything collapsed to one real speaker

    # Relabel so ids appear in first-spoken order (0,1,2,... down the timeline).
    order: dict[int, int] = {}
    for lab in labels.tolist():
        if lab not in order:
            order[lab] = len(order)
    return [order[lab] for lab in labels.tolist()]
