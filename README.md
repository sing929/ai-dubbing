# Local AI Video Dubbing

Free, local, no-account tool that **translates a video into another language and
re-voices it** — transcribe, translate, speak with a natural AI voice, and mux the new
audio back onto the video. Optionally make the dub **sound like the original speaker**.

Everything runs on your own PC. The core pipeline uses only free / open-source tools.

## What it does

- Transcribe speech (faster-whisper)
- Translate (Google / MyMemory)
- Speak it with a free neural voice — pick from a per-language **voice picker**
  (Microsoft edge-tts), or local Piper voices
- Keep the original background music (Demucs voice/music separation)
- Mux the new audio onto the video (ffmpeg)
- Extras: cover burned-in subtitles, burn translated subtitles, export `.srt`,
  reframe to vertical 9:16, audio-only export, batch queue
- Match the original speaker's **pace, pitch and loudness** ("Match original voice")
- **Optional voice cloning** — re-colour the dub to the original speaker's actual voice
  (OpenVoice v2, GPU recommended) — see [SETUP_VOICECLONE.md](SETUP_VOICECLONE.md)

## Quick start

1. Install **ffmpeg** and put `ffmpeg.exe` + `ffprobe.exe` on your PATH
   (or drop an `ffmpeg*/bin` folder next to `dub_app.py`).
2. `pip install -r requirements.txt`
3. `python dub_app.py` (or double-click `run_dub.bat`)
4. Drag a video in, tick a language, pick a voice, click **DUB ALL**.

## Optional: sound like the original speaker

Needs an NVIDIA GPU and a one-time setup. Follow
[SETUP_VOICECLONE.md](SETUP_VOICECLONE.md), then tick **"Clone original speaker's
voice"** in the app. Without the setup the option simply falls back to the normal AI
voice, so the tool still runs on any machine.

## Built on

faster-whisper, deep-translator, edge-tts, Piper, Demucs, ffmpeg, and (optional)
OpenVoice v2. Respect each project's license, and only dub content you have the rights
to use.
