# Parakeet TDT 0.6B V3 Speech Transcription App

A Windows-compatible speech transcription application using NVIDIA's Parakeet TDT 0.6B V3 model, powered by onnx-asr for GPU-accelerated inference via DirectML.

## Features

- **Multilingual Transcription**: Supports 25 European languages (bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk)
- **GPU Acceleration**: Uses DirectML for GPU inference on Windows (no CUDA required)
- **Automatic Punctuation and Capitalization**: Produces clean, readable transcripts
- **Word-level Timestamps**: Click segments in the table to play corresponding audio
- **Long Audio Support**: Handles audio up to 24 minutes
- **Web Interface**: Gradio-based UI for easy file upload and transcription

## Requirements

- Python 3.8+
- Windows 10/11 with GPU support
- Virtual environment (uv recommended)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/parakeet-wrapper.git
cd parakeet-wrapper
```

2. Create and activate virtual environment:
```bash
uv venv
uv run --python .venv python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
.venv\Scripts\python.exe get-pip.py
```

3. Install dependencies:
```bash
.venv\Scripts\pip install -r requirements.txt
```

## Usage

Run the application:
```bash
uv run python app.py
```

Open your browser to `http://localhost:7860` and upload audio files for transcription.

## Supported Audio Formats

- MP3, WAV, FLAC, OGG
- Automatically resampled to 16kHz mono if needed

## Model

Uses the ONNX-converted Parakeet TDT 0.6B V3 model from [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).

## License

CC BY 4.0 (same as original model)

## Credits

- Original model: NVIDIA Parakeet TDT 0.6B V3
- ONNX conversion: [istupakov](https://huggingface.co/istupakov)
- Inference: [onnx-asr](https://github.com/alphacep/onnx-asr)
