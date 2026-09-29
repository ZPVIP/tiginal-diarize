# tiginal-diarize

[English](README.md) | 中文

`tiginal-diarize` 是一个极速、轻量、开箱即用的本地说话人日志与分离（Speaker Diarization）命令行工具。

底层基于 NVIDIA 的 **Nemotron-3 Diarization**（Streaming Sortformer 架构，8 说话人量化 ONNX 模型，约 120MB）。整个工具使用纯 Rust 实现，标准输出输出结构化 JSON，专为流式 ASR（如 Tiginal / R2T2）对齐、会议纪要生成与 SRT 字幕导出设计。

---

## 特性

- 🚀 **极速本地推理**：基于 ONNX Runtime（支持 CPU 多线程与 SIMD 加速），无需 Python 环境。
- 📦 **轻量体积**：整套模型仅约 120MB（int8 量化），无需启动大模型推理服务器。
- 🎯 **干净的管道输出**：所有推理日志输出至 `stderr`，`stdout` 保持为纯净的 JSON，方便直接管道传递给 `jq`、Python 或其他下游工具。
- 🎵 **灵活的音频支持**：内置原生 WAV 解析（支持 8/16/24/32 位 PCM 及采样率重采样）；若系统安装了 `ffmpeg`，自动支持 MP3, M4A, FLAC, AAC, Opus 等全格式音频。
- 🛠️ **完善的调试模式**：内置 `--verbose` 详细调试信息，展示音频帧数、采样点、分块推断进度、说话人聚类与 RTF 耗时。

---

## 安装方式

### 1. 通过 Homebrew 安装（macOS / Linux）

```bash
# 直接从本仓库添加 tap 并安装
brew tap ZPVIP/tiginal-diarize https://github.com/ZPVIP/tiginal-diarize
brew install tiginal-diarize

# 或者通过本地 Formula 构建安装
brew install --build-from-source Formula/tiginal-diarize.rb
```

### 2. 通过 Cargo 安装

```bash
# 从本地仓库构建安装
cargo install --path .
```

### 3. 直接下载预编译二进制

在 GitHub Releases 页面下载对应系统的压缩包，解压后将 `tiginal-diarize` 放置于 `/usr/local/bin` 或你的 `$PATH` 路径中。

---

## 快速上手与用法

```bash
tiginal-diarize [OPTIONS] <AUDIO_PATH>
```

### 命令行参数详解

| 参数 | 短参数 | 说明 | 示例 / 默认值 |
| :--- | :--- | :--- | :--- |
| `<AUDIO_PATH>` | - | **必填**。待分析的音频文件路径 | `record.wav` |
| `--speakers` | `-s` | 固定说话人数量。若不传则由模型算法自动聚类探测 | `-s 2` |
| `--model-path` | `-m` | 自定义 Nemotron-3 ONNX 模型路径 | `-m /path/to/model_quantized.onnx` |
| `--download` | - | 若本地缓存未找到模型文件，自动从 Hugging Face 下载 | `--download` |
| `--output` | `-o` | 将 JSON 结果写入指定文件（默认打印到 stdout） | `-o output.json` |
| `--verbose` | `-v` | 输出详细运行日志和推断进度至 stderr，方便排查调试 | `-v` |
| `--help` | `-h` | 查看帮助文档 | `-h` |
| `--version` | `-V` | 查看版本号 | `-V` |

### 模型文件位置与管理

默认情况下，`tiginal-diarize` 会从系统缓存目录读取模型：
- **默认路径**：`~/.cache/tiginal/models/nemotron-3-diarization/`
  - 核心文件：`model_quantized.onnx`（约 300KB）与 `model_quantized.onnx_data`（约 120MB）
- 若本地不存在该文件：
  - 传入 `--download` 参数可自动下载；
  - 或者在 Tiginal 客户端的 **Model Engines** 页面中点击一键下载。

---

## 使用示例

### 1. 基础使用（自动探测说话人数量）
```bash
tiginal-diarize input.wav
```

**标准输出（stdout）JSON 示例：**
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

### 2. 固定 2 位说话人，并打印详细调试日志
```bash
tiginal-diarize -s 2 -v input.wav
```

**stderr 详细日志：**
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

### 3. 结合 `jq` 过滤提取某个 Speaker 的时间段
```bash
tiginal-diarize input.wav | jq '.[] | select(.speaker == 0)'
```

### 4. 保存为输出文件
```bash
tiginal-diarize input.wav -o turns.json
```

---

## 本地开发、调试与发布指南

### 1. 环境准备
- 安装 Rust 工具链（1.85+，推荐使用 Homebrew 或 rustup 安装）：
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### 2. 本地开发与实时调试
```bash
# 检查代码语法与类型
cargo check

# 运行代码并开启 --verbose 进行单步调试
cargo run -- -v /path/to/test.wav

# 运行测试用例
cargo test
```

### 3. 编译发布包（Release Build）
```bash
# 编译生产优化二进制
cargo build --release

# 查看产物
ls -lh target/release/tiginal-diarize
```

### 4. 版本发布（Release Workflow）
1. 在 `Cargo.toml` 中升级 `version = "x.y.z"`。
2. 提交代码并打上 Git Tag：
   ```bash
   git add .
   git commit -m "chore: release v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```
3. 计算发布压缩包的 SHA256 并更新 `Formula/tiginal-diarize.rb`：
   ```bash
   curl -sL https://github.com/ZPVIP/tiginal-diarize/archive/refs/tags/v0.1.0.tar.gz | sha256sum
   ```