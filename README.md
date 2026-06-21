# Local AI Video Dubbing

Free, local, no-account tool that **translates a video into another language and
re-voices it** — transcribe, translate, speak with a natural AI voice, and mux the new
audio back onto the video. Optionally make the dub **sound like the original speaker**.

Everything runs on your own PC. The core pipeline uses only free / open-source tools.

## What it does

- Transcribe speech (faster-whisper)
- Translate (DeepSeek API when a key is set, else free Google / MyMemory)
- Speak it with a free neural voice — pick from a per-language **voice picker**
  (Microsoft edge-tts), or local Piper voices
- **English**: optional local **XTTS v2** voice — more natural than Piper, with
  automatic male/female voices per speaker
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

## Optional: better translation (DeepSeek)

Paste a DeepSeek API key into the **DeepSeek API key** box (or set a
`DEEPSEEK_API_KEY` environment variable) for more natural, timing-aware
translations with no Google rate-limiting. The key is saved locally only
(it's gitignored) and never committed. Leave it blank to use the free Google
translator. DeepSeek only translates — the AI voice is unchanged.

## Optional: natural English voice (XTTS v2)

For a much more natural **English** voice than Piper, install Coqui XTTS v2
(local, free):

```
pip install coqui-tts "transformers<5"
```

English then uses XTTS automatically; other languages stay on edge-tts / Piper.
With the **gender** option on, male and female speakers get distinct male/female
voices. A CUDA GPU is recommended — on CPU it works but is slow. Set `XTTS_DISABLE=1`
to turn it off. Note: the XTTS-v2 model is under the Coqui Public Model License
(non-commercial) — review it before commercial use.

## Optional: MOSS local voice

MOSS-TTS-Nano is installed beside this app at `E:\dub\MOSS-TTS-Nano` and can be
enabled from the GUI with **Use MOSS local voice (experimental, CPU OK)**. It uses
the ONNX CPU backend for supported languages and can clone from the original voice
when **Clone original speaker's voice** is also enabled. First use downloads the
ONNX model files into `E:\dub\MOSS-TTS-Nano\models`; after that it runs locally.

## Optional: sound like the original speaker

Needs an NVIDIA GPU and a one-time setup. Follow
[SETUP_VOICECLONE.md](SETUP_VOICECLONE.md), then tick **"Clone original speaker's
voice"** in the app. Without the setup the option simply falls back to the normal AI
voice, so the tool still runs on any machine.

## Built on

faster-whisper, optional DeepSeek API (translation), deep-translator, edge-tts, Piper,
optional Coqui XTTS v2 (natural English voice), Demucs, ffmpeg, and (optional)
OpenVoice v2. Respect each project's license, and only dub content you have the rights
to use.
