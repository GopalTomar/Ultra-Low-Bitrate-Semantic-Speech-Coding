"""
Speech Decoder - Converts text back to speech
This is the "decompression" step in semantic coding
Supports multiple TTS engines: gTTS, pyttsx3, Coqui TTS
"""

import logging
import time
import numpy as np
from typing import Optional, Dict
from pathlib import Path
import soundfile as sf
import sounddevice as sd

# TTS Imports
from gtts import gTTS
import pyttsx3

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechDecoder:
    """Decodes text tokens back into speech audio"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.engine_type = config.TTS_ENGINE
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine"""
        logger.info(f"Initializing TTS engine: {self.engine_type}")
        
        if self.engine_type == "pyttsx3":
            self.engine = pyttsx3.init()
            self._configure_pyttsx3()
        elif self.engine_type == "gtts":
            # gTTS doesn't need initialization
            pass
        elif self.engine_type == "coqui":
            self._initialize_coqui()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_type}")
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine parameters"""
        # Set speech rate
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', rate * self.config.TTS_SPEED)
        
        # Set volume
        self.engine.setProperty('volume', 1.0)
        
        # List available voices (optional)
        voices = self.engine.getProperty('voices')
        logger.info(f"Available voices: {len(voices)}")
        
        # You can set a specific voice here
        # self.engine.setProperty('voice', voices[0].id)
    
    def _initialize_coqui(self):
        """Initialize Coqui TTS (if available)"""
        try:
            from TTS.api import TTS
            self.engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("Coqui TTS initialized successfully")
        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            raise
    
    def decode(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Decode text to speech audio
        
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Audio samples as numpy array
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for synthesis")
            return np.array([])
        
        logger.info(f"Synthesizing text: '{text[:50]}...'")
        start_time = time.time()
        
        if self.engine_type == "gtts":
            audio = self._decode_gtts(text, output_path)
        elif self.engine_type == "pyttsx3":
            audio = self._decode_pyttsx3(text, output_path)
        elif self.engine_type == "coqui":
            audio = self._decode_coqui(text, output_path)
        else:
            raise ValueError(f"Unsupported engine: {self.engine_type}")
        
        synthesis_time = time.time() - start_time
        logger.info(f"Synthesis completed in {synthesis_time:.2f}s")
        
        return audio
    
    def _decode_gtts(self, text: str, output_path: Optional[str]) -> np.ndarray:
        """Synthesize using Google TTS"""
        import tempfile
        from pydub import AudioSegment
        
        # Create temporary file if no output path specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            output_path = temp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang=self.config.TTS_LANGUAGE, slow=False)
        tts.save(output_path)
        
        # Load audio using pydub (gTTS outputs MP3)
        audio_segment = AudioSegment.from_mp3(output_path)
        
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Convert to float32 and normalize
        if audio_segment.sample_width == 2:  # 16-bit
            samples = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = samples.astype(np.float32) / 2147483648.0
        
        # Handle stereo to mono conversion
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Resample if necessary
        original_sr = audio_segment.frame_rate
        if original_sr != self.config.SAMPLE_RATE:
            import librosa
            samples = librosa.resample(
                samples,
                orig_sr=original_sr,
                target_sr=self.config.SAMPLE_RATE
            )
        
        logger.info(f"gTTS audio generated: {len(samples)} samples")
        return samples
    
    def _decode_pyttsx3(self, text: str, output_path: Optional[str]) -> np.ndarray:
        """Synthesize using pyttsx3 (offline)"""
        import tempfile
        
        # Create temporary file if needed
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
        
        # Synthesize and save
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        
        # Load the generated audio
        import librosa
        audio, sr = librosa.load(output_path, sr=self.config.SAMPLE_RATE, mono=True)
        
        logger.info(f"pyttsx3 audio generated: {len(audio)} samples")
        return audio
    
    def _decode_coqui(self, text: str, output_path: Optional[str]) -> np.ndarray:
        """Synthesize using Coqui TTS"""
        import tempfile
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
        
        # Generate speech
        self.engine.tts_to_file(text=text, file_path=output_path)
        
        # Load audio
        import librosa
        audio, sr = librosa.load(output_path, sr=self.config.SAMPLE_RATE, mono=True)
        
        logger.info(f"Coqui TTS audio generated: {len(audio)} samples")
        return audio
    
    def decode_and_play(self, text: str):
        """
        Synthesize text and play it immediately
        
        Args:
            text: Text to synthesize and play
        """
        audio = self.decode(text)
        
        if len(audio) > 0:
            logger.info("Playing synthesized audio...")
            sd.play(audio, self.config.SAMPLE_RATE)
            sd.wait()
            logger.info("Playback complete")
        else:
            logger.warning("No audio to play")
    
    def batch_decode(self, text_list: list, output_dir: Path) -> list:
        """
        Decode multiple text inputs
        
        Args:
            text_list: List of text strings
            output_dir: Directory to save audio files
            
        Returns:
            List of audio arrays
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        audio_list = []
        
        for i, text in enumerate(text_list):
            logger.info(f"Processing {i+1}/{len(text_list)}")
            output_path = output_dir / f"output_{i:03d}.wav"
            audio = self.decode(text, str(output_path))
            audio_list.append(audio)
        
        return audio_list
    
    def get_synthesis_metrics(self, text: str, audio: np.ndarray) -> Dict:
        """
        Calculate synthesis performance metrics
        
        Returns:
            Dictionary with timing and quality metrics
        """
        audio_duration = len(audio) / self.config.SAMPLE_RATE
        text_length = len(text)
        
        # Estimate speaking rate (words per minute)
        word_count = len(text.split())
        speaking_rate = (word_count / audio_duration) * 60 if audio_duration > 0 else 0
        
        metrics = {
            'text_length_chars': text_length,
            'text_length_words': word_count,
            'audio_duration_seconds': audio_duration,
            'speaking_rate_wpm': speaking_rate,
            'characters_per_second': text_length / audio_duration if audio_duration > 0 else 0
        }
        
        return metrics


class MultiSpeakerDecoder(SpeechDecoder):
    """
    Advanced decoder with speaker embedding support
    Uses speaker vectors to maintain voice identity
    """
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.speaker_embeddings = {}
    
    def register_speaker(self, speaker_id: str, embedding: np.ndarray):
        """
        Register a speaker embedding
        
        Args:
            speaker_id: Unique identifier for speaker
            embedding: Speaker embedding vector (e.g., from pyannote)
        """
        self.speaker_embeddings[speaker_id] = embedding
        logger.info(f"Registered speaker: {speaker_id}")
    
    def decode_with_speaker(
        self,
        text: str,
        speaker_id: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Decode text with specific speaker voice
        Note: Requires Coqui TTS with multi-speaker support
        """
        if speaker_id not in self.speaker_embeddings:
            logger.warning(f"Unknown speaker: {speaker_id}, using default voice")
            return self.decode(text, output_path)
        
        if self.engine_type != "coqui":
            logger.warning("Multi-speaker only supported with Coqui TTS")
            return self.decode(text, output_path)
        
        # TODO: Implement speaker-conditioned synthesis with Coqui
        # This requires loading a multi-speaker TTS model
        logger.info(f"Synthesizing with speaker: {speaker_id}")
        return self.decode(text, output_path)


# Example usage
if __name__ == "__main__":
    # Test different engines
    test_text = "Hello! This is a test of the ultra-low bitrate semantic speech coding system."
    
    print("\n=== Testing gTTS (Online) ===")
    Config.TTS_ENGINE = "gtts"
    decoder_gtts = SpeechDecoder()
    audio_gtts = decoder_gtts.decode(test_text)
    decoder_gtts.decode_and_play(test_text)
    
    print("\n=== Testing pyttsx3 (Offline) ===")
    Config.TTS_ENGINE = "pyttsx3"
    decoder_pyttsx3 = SpeechDecoder()
    audio_pyttsx3 = decoder_pyttsx3.decode(test_text)
    decoder_pyttsx3.decode_and_play(test_text)
    
    # Metrics
    print("\n=== Synthesis Metrics ===")
    metrics = decoder_gtts.get_synthesis_metrics(test_text, audio_gtts)
    print(f"Text Length: {metrics['text_length_words']} words")
    print(f"Audio Duration: {metrics['audio_duration_seconds']:.2f}s")
    print(f"Speaking Rate: {metrics['speaking_rate_wpm']:.0f} WPM")