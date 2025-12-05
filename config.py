"""
Configuration file for Ultra-Low Bitrate Semantic Speech Coding
Centralizes all system parameters for easy tuning
"""

import os
from pathlib import Path

class Config:
    # ============ AUDIO SETTINGS ============
    SAMPLE_RATE = 16000  # Hz - Standard for Whisper
    CHANNELS = 1  # Mono
    DTYPE = 'float32'
    CHUNK_DURATION = 5  # seconds - Recording chunk size
    
    # ============ ASR (Speech-to-Text) SETTINGS ============
    WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
    # Model size vs speed tradeoff:
    # - tiny: ~1GB RAM, fastest, 74M params
    # - base: ~1GB RAM, fast, 74M params  
    # - small: ~2GB RAM, balanced, 244M params
    # - medium: ~5GB RAM, accurate, 769M params
    WHISPER_LANGUAGE = "en"  # Set to None for auto-detection
    
    # ============ VAD (Voice Activity Detection) SETTINGS ============
    VAD_MODE = 3  # Aggressiveness: 0 (least) to 3 (most aggressive)
    VAD_FRAME_DURATION = 30  # ms - must be 10, 20, or 30
    SILENCE_THRESHOLD = 2.5 # Seconds of silence before stopping
    
    # ============ NOISE REDUCTION SETTINGS ============
    NOISE_REDUCE_ENABLED = True
    STATIONARY_NOISE = True  # For constant background noise
    
    # ============ TTS (Text-to-Speech) SETTINGS ============
    TTS_ENGINE = "gtts"  # Options: gtts, pyttsx3, coqui
    TTS_LANGUAGE = "en"
    TTS_SPEED = 1.0  # Speed multiplier (pyttsx3 only)
    
    # ============ NETWORK SETTINGS ============
    SERVER_HOST = "localhost"
    SERVER_PORT = 8765
    USE_ENCRYPTION = True
    
    # ============ COMPRESSION SETTINGS ============
    TEXT_ENCODING = "utf-8"
    COMPRESSION_ENABLED = True  # Enable additional text compression
    
    # ============ FILE PATHS ============
    BASE_DIR = Path(__file__).parent
    AUDIO_DIR = BASE_DIR / "audio_files"
    OUTPUT_DIR = BASE_DIR / "output"
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Create directories if they don't exist
    for directory in [AUDIO_DIR, OUTPUT_DIR, LOGS_DIR, TEMP_DIR]:
        directory.mkdir(exist_ok=True)
    
    # ============ METRICS & EVALUATION ============
    CALCULATE_WER = True  # Word Error Rate
    CALCULATE_COMPRESSION_RATIO = True
    SAVE_INTERMEDIATE_FILES = True  # For debugging
    
    # ============ PERFORMANCE SETTINGS ============
    USE_GPU = True  # Use CUDA if available
    MAX_RETRIES = 3  # Network retry attempts
    TIMEOUT = 30  # seconds - Network timeout
    
    @classmethod
    def get_device(cls):
        """Determine if GPU is available"""
        import torch
        if cls.USE_GPU and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("ULTRA-LOW BITRATE SEMANTIC SPEECH CODING - CONFIG")
        print("=" * 60)
        print(f"Whisper Model: {cls.WHISPER_MODEL}")
        print(f"Sample Rate: {cls.SAMPLE_RATE} Hz")
        print(f"TTS Engine: {cls.TTS_ENGINE}")
        print(f"Device: {cls.get_device()}")
        print(f"VAD Mode: {cls.VAD_MODE} (Aggressiveness)")
        print(f"Noise Reduction: {'Enabled' if cls.NOISE_REDUCE_ENABLED else 'Disabled'}")
        print("=" * 60)

if __name__ == "__main__":
    Config.print_config()