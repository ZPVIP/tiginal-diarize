# tiginal-diarize

English | [中文](README.zh.md)

`tiginal-diarize` is a blazing-fast, lightweight, and self-contained command-line speaker diarization tool.

Powered by NVIDIA's **Nemotron-3 Diarization** (Streaming Sortformer architecture, 8-speaker int8 quantized ONNX model, ~120MB). Built purely in Rust, it outputs clean, structured JSON to `stdout`, specifically designed for streaming ASR (such as Tiginal / R2T2) alignment, meeting minutes generation, and SRT subtitle exports.

---

## Features

- 🚀 **Fast Local Inference**: Powered by ONNX Runtime with CPU multi-threading and SIMD vectorization. Zero Python dependency.
- 📦 **Compact Footprint**: The entire quantized model is only ~120MB—no need for heavyweight inference servers or daemon processes.
- 🎯 **Pipe-Friendly Output**: All progress, diagnostics, and informational logs go to `stderr`, keeping `stdout` strictly formatted JSON for `jq`, scripts, or downstream applications.
- 🎵 **Broad Audio Support**: Native WAV parsing (supports 8/16/24/32-bit PCM and automatic resampling). Automatically falls back to `ffmpeg` if installed to support MP3, M4A, FLAC, AAC, Opus, etc.
- 🛠️ **Verbose Diagnostics**: Built-in `--verbose` flag displays audio sample counts, STFT frames, chunked encoder step progress, clustering details, and real-time factor (RTF).

---

## Installation

### 1. Via Homebrew (macOS / Linux)

```bash
# Add tap and install directly from this repository
brew tap ZPVIP/tiginal-diarize https://github.com/ZPVIP/tiginal-diarize
brew install tiginal-diarize

# Or build locally from formula
brew install --build-from-source Formula/tiginal-diarize.rb
```

### 2. Via Cargo

```bash
# Build and install from source
cargo install --path .
```

### 3. Pre-built Binaries

Download pre-compiled binaries from the GitHub Releases page, extract the archive, and place `tiginal-diarize` into `/usr/local/bin` or any directory in your `$PATH`.

---

## CLI Reference & Usage

```bash
tiginal-diarize [OPTIONS] <AUDIO_PATH>
```

### Options

| Option | Short | Description | Default / Example |
| :--- | :--- | :--- | :--- |
| `<AUDIO_PATH>` | - | **Required**. Path to the target audio file | `meeting.wav` |
| `--speakers` | `-s` | Fix number of speakers. Defaults to automatic clustering | `-s 2` |
| `--model-path` | `-m` | Custom path to `model_quantized.onnx` | `-m /path/to/model.onnx` |
| `--download` | - | Automatically download the ~120MB model from Hugging Face if missing | `--download` |
| `--output` | `-o` | Output file path (defaults to `stdout`) | `-o turns.json` |
| `--verbose` | `-v` | Output detailed progress and diagnostic logs to `stderr` | `-v` |
| `--help` | `-h` | Print help information | `-h` |
| `--version` | `-V` | Print version information | `-V` |

---

## Model Storage & Management

By default, `tiginal-diarize` resolves the ONNX model from:
- **Default Path**: `~/.cache/tiginal/models/nemotron-3-diarization/`
  - Required files: `model_quantized.onnx` (~300 KB) and `model_quantized.onnx_data` (~120 MB)
- If the model is not found:
  - Pass the `--download` flag to trigger an automated download from Hugging Face.
  - Or manually click **Download Model** in the Tiginal GUI under **Settings > Model Engines**.

---

## Usage Examples

### 1. Basic Diarization (Automatic Speaker Detection)
```bash
tiginal-diarize input.wav
```

**JSON Output (`stdout`):**
```json
[
  {
    "speaker": 0,
    "start": 0.16,
    "end": 2.45,
    "start_ms": 160,
    "end_ms": 2450
  },
  {
    "speaker": 1,
    "start": 2.6,
    "end": 5.12,
    "start_ms": 2600,
    "end_ms": 5120
  }
]
```

### 2. Constrain to 2 Speakers with Verbose Logging
```bash
tiginal-diarize -s 2 -v input.wav
```

**Verbose Diagnostics (`stderr`):**
```text
[info] Starting tiginal-diarize v0.1.0
[info] Target audio: input.wav
[info] Found cached model at: ~/.cache/tiginal/models/nemotron-3-diarization/model_quantized.onnx
[info] Successfully decoded WAV with native hound parser: 160000 samples (10.00s)
[info] Audio duration: 10.00s (0.17 minutes)
[info] Initializing ONNX session with 8 intra-threads...
[info] Audio features: 1001 frames (10.01s), 126 encoder steps
[progress] Diarization: 100% (step 126/126)
[info] Thresholding speaker probabilities and bridging short pauses...
[info] Identified 4 speaker turns across 2 unique speakers
[done] Finished in 0.42s (RTF: 0.042x realtime)
```

### 3. Filter Specific Speaker with `jq`
```bash
tiginal-diarize input.wav | jq '.[] | select(.speaker == 0)'
```

### 4. Save Directly to File
```bash
tiginal-diarize input.wav -o results.json
```

---

## Development, Testing & Release

### Prerequisites
- Rust 1.85+ (recommended via Homebrew or `rustup`):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Development & Debugging
```bash
# Check compiler diagnostics
cargo check

# Run with verbose logging for debugging
cargo run -- -v /path/to/test.wav

# Run tests
cargo test
```

### Production Build
```bash
# Compile optimized release binary
cargo build --release

# Inspect binary size
ls -lh target/release/tiginal-diarize
```

### Release Workflow
1. Bump `version = "x.y.z"` in `Cargo.toml`.
2. Commit and create a Git tag:
   ```bash
   git add .
   git commit -m "chore: release v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```
3. Calculate SHA256 of the release tarball and update `Formula/tiginal-diarize.rb`:
   ```bash
   curl -sL https://github.com/ZPVIP/tiginal-diarize/archive/refs/tags/v0.1.0.tar.gz | sha256sum
   ```
