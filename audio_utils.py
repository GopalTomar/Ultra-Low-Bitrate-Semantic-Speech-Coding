"""
Audio utility functions for recording, preprocessing, and VAD
Handles all low-level audio operations
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import noisereduce as nr
from scipy import signal
import librosa
import logging
from typing import Tuple, Optional
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioRecorder:
    """Handles audio recording with VAD"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.vad = webrtcvad.Vad(config.VAD_MODE)
        self.sample_rate = config.SAMPLE_RATE
        self.channels = config.CHANNELS
        
    def record_audio(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds (None for VAD-based)
        
        Returns:
            numpy array of audio samples
        """
        if duration:
            logger.info(f"Recording for {duration} seconds...")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.config.DTYPE
            )
            sd.wait()
            return audio_data.flatten()
        else:
            return self._record_with_vad()
    
    def _record_with_vad(self) -> np.ndarray:
        """
        Record audio with Voice Activity Detection
        Stops automatically when silence is detected
        """
        logger.info("Recording with VAD (speak now, pause to stop)...")
        
        frames = []
        silence_frames = 0
        max_silence_frames = int(
            self.config.SILENCE_THRESHOLD * 1000 / self.config.VAD_FRAME_DURATION
        )
        
        frame_size = int(self.sample_rate * self.config.VAD_FRAME_DURATION / 1000)
        
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16'
        )
        
        with stream:
            while True:
                frame_data, _ = stream.read(frame_size)
                frame_bytes = frame_data.tobytes()
                
                # Check if frame contains speech
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                
                if is_speech:
                    frames.append(frame_data)
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames > max_silence_frames and len(frames) > 0:
                        logger.info("Silence detected. Stopping recording.")
                        break
        
        # Concatenate all frames
        if frames:
            audio_data = np.concatenate(frames, axis=0).flatten()
            # Convert from int16 to float32
            audio_data = audio_data.astype(np.float32) / 32768.0
            return audio_data
        else:
            logger.warning("No speech detected!")
            return np.array([])


class AudioPreprocessor:
    """Handles audio preprocessing and enhancement"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            audio: Raw audio samples
            
        Returns:
            Preprocessed audio
        """
        if len(audio) == 0:
            return audio
        
        # Step 1: Normalize amplitude
        audio = self._normalize(audio)
        
        # Step 2: Noise reduction
        if self.config.NOISE_REDUCE_ENABLED:
            audio = self._reduce_noise(audio)
        
        # Step 3: High-pass filter to remove low-frequency rumble
        audio = self._highpass_filter(audio)
        
        # Step 4: Trim silence from edges
        audio = self._trim_silence(audio)
        
        return audio
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.config.SAMPLE_RATE,
                stationary=self.config.STATIONARY_NOISE
            )
            logger.info("Noise reduction applied")
            return reduced
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _highpass_filter(self, audio: np.ndarray, cutoff: int = 80) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise
        
        Args:
            audio: Input audio
            cutoff: Cutoff frequency in Hz
        """
        nyquist = self.config.SAMPLE_RATE / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def _trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from beginning and end"""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed


class AudioAnalyzer:
    """Analyzes audio properties for metrics"""
    
    @staticmethod
    def calculate_duration(audio: np.ndarray, sample_rate: int) -> float:
        """Calculate audio duration in seconds"""
        return len(audio) / sample_rate
    
    @staticmethod
    def calculate_size_bytes(audio: np.ndarray) -> int:
        """Calculate raw audio size in bytes"""
        return audio.nbytes
    
    @staticmethod
    def calculate_bitrate(audio: np.ndarray, sample_rate: int) -> float:
        """Calculate bitrate in bits per second"""
        duration = AudioAnalyzer.calculate_duration(audio, sample_rate)
        size_bits = audio.nbytes * 8
        return size_bits / duration if duration > 0 else 0
    
    @staticmethod
    def extract_features(audio: np.ndarray, sample_rate: int) -> dict:
        """
        Extract audio features for analysis
        
        Returns:
            Dictionary with MFCC, pitch, energy, etc.
        """
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        
        # Energy
        features['energy'] = np.sum(audio ** 2) / len(audio)
        
        return features


def save_audio(audio: np.ndarray, filepath: str, sample_rate: int = Config.SAMPLE_RATE):
    """Save audio to file"""
    sf.write(filepath, audio, sample_rate)
    logger.info(f"Audio saved to: {filepath}")


def load_audio(filepath: str, target_sr: int = Config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio from file and resample if necessary
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    logger.info(f"Audio loaded from: {filepath}")
    return audio, sr


# Example usage
if __name__ == "__main__":
    # Test recording
    recorder = AudioRecorder()
    audio = recorder.record_audio(duration=3)
    
    # Test preprocessing
    preprocessor = AudioPreprocessor()
    clean_audio = preprocessor.preprocess(audio)
    
    # Analyze
    analyzer = AudioAnalyzer()
    print(f"Duration: {analyzer.calculate_duration(clean_audio, Config.SAMPLE_RATE):.2f}s")
    print(f"Size: {analyzer.calculate_size_bytes(clean_audio)} bytes")
    print(f"Bitrate: {analyzer.calculate_bitrate(clean_audio, Config.SAMPLE_RATE):.0f} bps")
    
    # Save
    save_audio(clean_audio, Config.TEMP_DIR / "test_recording.wav")