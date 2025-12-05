"""
Speech Encoder - Converts audio to text using Whisper
This is the "compression" step in semantic coding
"""

import whisper
import torch
import logging
import time
from typing import Dict, Optional
import numpy as np
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechEncoder:
    """Encodes speech audio into text tokens"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.device = config.get_device()
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load Whisper model"""
        logger.info(f"Loading Whisper model '{self.config.WHISPER_MODEL}' on {self.device}...")
        start_time = time.time()
        
        self.model = whisper.load_model(
            self.config.WHISPER_MODEL,
            device=self.device
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    def encode(self, audio: np.ndarray) -> Dict:
        """
        Encode audio to text
        
        Args:
            audio: Audio samples (float32, normalized)
            
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language: Detected language
                - segments: Word-level timestamps
                - duration: Processing time
        """
        if len(audio) == 0:
            logger.warning("Empty audio provided")
            return {
                'text': '',
                'language': '',
                'segments': [],
                'duration': 0
            }
        
        logger.info("Starting transcription...")
        start_time = time.time()
        
        # Whisper expects float32 audio between -1 and 1
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            language=self.config.WHISPER_LANGUAGE,
            task='transcribe',
            fp16=(self.device == 'cuda')  # Use FP16 on GPU for speed
        )
        
        processing_time = time.time() - start_time
        
        # Extract results
        transcription = {
            'text': result['text'].strip(),
            'language': result['language'],
            'segments': result['segments'],
            'duration': processing_time
        }
        
        logger.info(f"Transcription complete in {processing_time:.2f}s")
        logger.info(f"Detected language: {transcription['language']}")
        logger.info(f"Transcribed text: '{transcription['text']}'")
        
        return transcription
    
    def encode_with_timestamps(self, audio: np.ndarray) -> Dict:
        """
        Encode with word-level timestamps
        Useful for lip-sync or alignment applications
        """
        result = self.encode(audio)
        
        # Extract word-level timing
        words_with_timing = []
        for segment in result['segments']:
            for word_info in segment.get('words', []):
                words_with_timing.append({
                    'word': word_info['word'],
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        result['words'] = words_with_timing
        return result
    
    def get_compression_metrics(self, audio: np.ndarray, text: str) -> Dict:
        """
        Calculate compression statistics
        
        Returns:
            Dictionary with compression ratios and bitrates
        """
        # Audio metrics
        audio_size_bytes = audio.nbytes
        audio_duration = len(audio) / self.config.SAMPLE_RATE
        audio_bitrate = (audio_size_bytes * 8) / audio_duration
        
        # Text metrics
        text_size_bytes = len(text.encode(self.config.TEXT_ENCODING))
        text_bitrate = (text_size_bytes * 8) / audio_duration if audio_duration > 0 else 0
        
        # Compression ratio
        compression_ratio = audio_size_bytes / text_size_bytes if text_size_bytes > 0 else 0
        
        metrics = {
            'audio_size_bytes': audio_size_bytes,
            'text_size_bytes': text_size_bytes,
            'audio_duration_seconds': audio_duration,
            'audio_bitrate_bps': audio_bitrate,
            'text_bitrate_bps': text_bitrate,
            'compression_ratio': compression_ratio,
            'bandwidth_reduction_percent': ((audio_bitrate - text_bitrate) / audio_bitrate * 100) if audio_bitrate > 0 else 0
        }
        
        return metrics
    
    def benchmark_model(self, test_audio: np.ndarray, iterations: int = 5):
        """
        Benchmark transcription speed
        
        Args:
            test_audio: Audio sample for testing
            iterations: Number of test iterations
        """
        logger.info(f"Running benchmark ({iterations} iterations)...")
        
        times = []
        for i in range(iterations):
            start = time.time()
            _ = self.encode(test_audio)
            elapsed = time.time() - start
            times.append(elapsed)
            logger.info(f"Iteration {i+1}: {elapsed:.3f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"\nBenchmark Results:")
        logger.info(f"Average: {avg_time:.3f}s (±{std_time:.3f}s)")
        logger.info(f"Min: {min(times):.3f}s")
        logger.info(f"Max: {max(times):.3f}s")
        
        # Real-time factor
        audio_duration = len(test_audio) / self.config.SAMPLE_RATE
        rtf = avg_time / audio_duration
        logger.info(f"Real-time Factor: {rtf:.2f}x")
        
        if rtf < 1.0:
            logger.info("✓ Faster than real-time!")
        else:
            logger.warning("✗ Slower than real-time")


# Example usage
if __name__ == "__main__":
    from audio_utils import AudioRecorder, AudioPreprocessor
    
    # Record audio
    recorder = AudioRecorder()
    print("\n=== RECORDING ===")
    audio = recorder.record_audio(duration=5)
    
    # Preprocess
    preprocessor = AudioPreprocessor()
    clean_audio = preprocessor.preprocess(audio)
    
    # Encode to text
    print("\n=== ENCODING (Speech-to-Text) ===")
    encoder = SpeechEncoder()
    result = encoder.encode(clean_audio)
    
    print(f"\nTranscribed Text: '{result['text']}'")
    print(f"Language: {result['language']}")
    print(f"Processing Time: {result['duration']:.2f}s")
    
    # Compression metrics
    print("\n=== COMPRESSION ANALYSIS ===")
    metrics = encoder.get_compression_metrics(clean_audio, result['text'])
    
    print(f"Audio Size: {metrics['audio_size_bytes']:,} bytes")
    print(f"Text Size: {metrics['text_size_bytes']:,} bytes")
    print(f"Compression Ratio: {metrics['compression_ratio']:.1f}:1")
    print(f"Audio Bitrate: {metrics['audio_bitrate_bps']:,.0f} bps")
    print(f"Text Bitrate: {metrics['text_bitrate_bps']:.0f} bps")
    print(f"Bandwidth Reduction: {metrics['bandwidth_reduction_percent']:.1f}%")