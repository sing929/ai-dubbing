# Setup: "Clone original speaker's voice" (optional)

This add-on makes the dub **sound like the original speaker** in the source video,
using [OpenVoice v2](https://github.com/myshell-ai/OpenVoice) zero-shot voice
conversion. It is **optional** — the dubbing tool works fine without it (it just uses
the normal AI voice).

## Requirements

- An **NVIDIA GPU** + recent driver (CPU works but is very slow).
- **Python 3.10 or 3.11** (OpenVoice does not support 3.12+ well).
- The base tool already installed: `pip install -r requirements.txt`.

## Install (run in the project folder)

1. **PyTorch with CUDA** (pick the URL matching your CUDA; `cu121` is common):
   ```
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Verify the GPU is visible — this must print `True`:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **OpenVoice v2 + helpers**:
   ```
   pip install git+https://github.com/myshell-ai/OpenVoice.git
   pip install -r requirements-voiceclone.txt
   ```

3. **Download the model checkpoints.** Get `checkpoints_v2` from the OpenVoice
   releases (a file like `checkpoints_v2_0417.zip`) and unzip it into this folder so
   these files exist:
   ```
   checkpoints_v2/converter/config.json
   checkpoints_v2/converter/checkpoint.pth
   ```
   (Or put them anywhere and set the env var `OPENVOICE_CKPT` to that `converter` folder.)

## Use

1. `python dub_app.py`
2. Tick **"Clone original speaker's voice"**.
3. Dub a short clip first and listen. The first run downloads extra models.

## Troubleshooting

- **"checkpoint not found"** → step 3 didn't land in the right place; or set
  `OPENVOICE_CKPT` to your `checkpoints_v2/converter` folder.
- **"no CUDA GPU detected — SLOW"** → torch can't see the GPU; redo step 1 and confirm
  `torch.cuda.is_available()` is `True`.
- Cross-language cloning is good but not perfect — expect a recognisable match of the
  original voice with some artefacts.
- If a segment fails mid-run, the tool automatically falls back to the normal AI voice
  for that segment, so you always get a finished dub.
