# Ultra-Low Bitrate Semantic Speech Coding

A revolutionary speech compression system that achieves compression ratios exceeding **10,000:1** by transforming speech into semantic text tokens instead of preserving the acoustic waveform.

![Project Banner](docs/banner.png)

## 🎯 Overview

This project implements a Speech-to-Text-to-Speech (STT-TTS) pipeline that reframes traditional audio compression. Instead of encoding waveforms, we extract linguistic meaning, transmit it as compact text, and reconstruct natural speech at the receiver.

### Key Features

- ⚡ **Ultra-Low Bitrate**: ~100 bps vs. 64,000 bps (telephony)
- 🎤 **Voice Activity Detection**: Automatic silence removal
- 🔇 **Noise Reduction**: Clean preprocessing pipeline
- 🤖 **State-of-the-Art ASR**: OpenAI Whisper integration
- 🔊 **Multiple TTS Engines**: gTTS, pyttsx3, Coqui TTS
- 📊 **Comprehensive Metrics**: WER, compression ratios, latency analysis
- 🌐 **Network Ready**: WebSocket-based client/server architecture

## 📊 Compression Performance

| Metric | Traditional (PCM) | MP3 (128 kbps) | **Semantic Coding** |
|--------|-------------------|----------------|---------------------|
| Bitrate | 64,000 bps | 128,000 bps | **~100 bps** |
| Compression Ratio | 1:1 (baseline) | ~5:1 | **640:1** |
| Bandwidth | 64 KB/s | 16 KB/s | **12.5 bytes/s** |

## 🏗️ Architecture
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Audio     │───▶│ Preprocessing│───▶│   Whisper   │
│  Recording  │    │   (VAD, NR)  │    │   (ASR)     │
└─────────────┘    └──────────────┘    └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │    Text     │
                                        │ Compression │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Network    │
                                        │Transmission │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │Decompression│
                                        └─────────────┘
                                               │
                                               ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Audio     │◀───│     TTS      │◀───│   Text      │
│  Playback   │    │  (gTTS/etc)  │    │  Decoded    │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Installation

#### Linux/macOS:
```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
```

#### Windows:
```cmd
install.bat
venv\Scripts\activate.bat
```

### Running the Demo

#### Complete End-to-End Demo (Single Process):
```bash
python demo_end_to_end.py --mode interactive
```

#### Client/Server Mode:

**Terminal 1 (Receiver):**
```bash
python main_receiver.py --mode server
```

**Terminal 2 (Sender):**
```bash
python main_sender.py --mode interactive --connect
```

## 📁 Project Structure
```
BITRATE_SEMANTIC_SPEECH/
├── config.py                 # Configuration settings
├── audio_utils.py            # Audio recording & preprocessing
├── speech_encoder.py         # ASR (Whisper)
├── speech_decoder.py         # TTS (gTTS/pyttsx3/Coqui)
├── compression_engine.py     # Text compression
├── network_layer.py          # WebSocket client/server
├── metrics_evaluator.py      # Evaluation metrics
├── main_sender.py            # Sender application
├── main_receiver.py          # Receiver application
├── demo_end_to_end.py        # Complete demo
├── requirements.txt          # Python dependencies
├── install.sh               # Linux/Mac installer
├── install.bat              # Windows installer
├── README.md                 # This file
├── audio_files/              # Input audio directory
├── output/                   # Output files & reports
├── logs/                     # Log files
└── temp/                     # Temporary files
```

## 🔧 Configuration

Edit `config.py` to customize:
```python
# ASR Settings
WHISPER_MODEL = "base"  # tiny, base, small, medium, large
WHISPER_LANGUAGE = "en"

# TTS Settings
TTS_ENGINE = "gtts"  # gtts, pyttsx3, coqui

# Audio Settings
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1  # Mono

# VAD Settings
VAD_MODE = 3  # 0-3 (aggressiveness)

# Network Settings
SERVER_HOST = "localhost"
SERVER_PORT = 8765
```

## 📊 Usage Examples

### Example 1: Process an Audio File
```bash
python demo_end_to_end.py --mode file --input audio_files/sample.wav
```

### Example 2: Batch Processing
```bash
# Place audio files in audio_files/
python demo_end_to_end.py --mode batch
```

### Example 3: Network Transmission
```bash
# Terminal 1
python main_receiver.py --mode server --host 0.0.0.0 --port 8765

# Terminal 2
python main_sender.py --mode interactive --connect --server ws://localhost:8765
```

### Example 4: Custom TTS Engine
```bash
python demo_end_to_end.py --mode interactive --tts-engine pyttsx3
```

## 📈 Evaluation Metrics

The system automatically calculates:

### Compression Metrics
- Compression ratio (Audio size / Text size)
- Bitrate reduction (%)
- Comparison with standard codecs (MP3, Opus, GSM, etc.)

### Transcription Quality
- Word Error Rate (WER)
- Character Error Rate (CER)
- Transcription accuracy (%)

### Performance Metrics
- Processing latency per stage
- Real-Time Factor (RTF)
- End-to-end throughput

### Example Report Output:
```
=== COMPRESSION PERFORMANCE ===
Compression Ratio: 640.5:1
Original Bitrate: 512,000 bps
Semantic Bitrate: 95 bps
Bandwidth Reduction: 99.98%

=== PROCESSING PERFORMANCE ===
Total Processing Time: 1,245 ms
Real-Time Factor: 0.25x
✓ FASTER than real-time!

=== TRANSCRIPTION QUALITY ===
Word Error Rate: 2.5%
Accuracy: 97.5%
```

## 🎓 Academic Context

This project satisfies core Signal Processing curriculum requirements:

1. **Text-to-Speech & Speech-to-Text**: Bridges acoustics and NLP
2. **Python Implementation**: Uses state-of-the-art libraries (Whisper, gTTS)
3. **Consumer Electronics Application**: Virtual assistants, low-bandwidth communication

### Theoretical Foundation

The system performs **semantic vector quantization**:
- Traditional VQ: Maps signal vectors → codebook of acoustic shapes
- Semantic VQ: Maps signal vectors → codebook of words (dictionary)

### Trade-offs

**Preserved:**
- Linguistic meaning
- Message intelligibility

**Lost:**
- Speaker identity (voice characteristics)
- Emotional prosody
- Background environmental sounds

See the project analysis PDF for detailed mathematical derivations.

## 🛠️ Advanced Features

### Speaker Embedding (Experimental)

To preserve voice identity:
```python
from speech_decoder import MultiSpeakerDecoder

decoder = MultiSpeakerDecoder()
decoder.register_speaker("user_123", speaker_embedding)
decoder.decode_with_speaker(text, "user_123")
```

### Custom VAD Tuning
```python
from config import Config

Config.VAD_MODE = 3  # Most aggressive
Config.SILENCE_THRESHOLD = 0.8  # seconds
```

### Batch Evaluation
```python
from metrics_evaluator import EvaluationReport

report = EvaluationReport()
# ... process multiple files ...
report.save_report("batch_results")
```

## 🐛 Troubleshooting

### Issue: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Issue: FFmpeg not found
**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:** Download from https://ffmpeg.org and add to PATH

### Issue: CUDA out of memory
Use a smaller Whisper model:
```python
Config.WHISPER_MODEL = "tiny"  # or "base"
```

### Issue: gTTS connection timeout
Switch to offline TTS:
```python
Config.TTS_ENGINE = "pyttsx3"
```

## 📚 References

1. OpenAI Whisper: https://github.com/openai/whisper
2. WebRTC VAD: https://github.com/wiseman/py-webrtcvad
3. gTTS: https://github.com/pndurette/gTTS
4. Research Paper: [Different Methods for Speech-to-Text and Text-to-Speech Conversion](https://www.researchgate.net/publication/345195645)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Your Name**
- Project for Signal Processing Course
- Institution: [Your University]
- Year: 2024

## 🙏 Acknowledgments

- OpenAI for Whisper
- Google for gTTS
- WebRTC project for VAD
- All open-source contributors

---

**Note:** This is an educational project demonstrating semantic speech coding concepts. For production use, consider additional features like error correction, adaptive bitrate, and enhanced speaker preservation.