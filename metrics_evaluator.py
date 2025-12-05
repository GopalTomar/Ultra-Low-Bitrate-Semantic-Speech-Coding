"""
Metrics Evaluator - Calculate WER, compression ratios, and quality metrics
Provides comprehensive evaluation of the semantic coding system
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

# For WER calculation
from jiwer import wer, cer

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressionMetrics:
    """Calculate compression-related metrics"""
    
    @staticmethod
    def calculate_compression_ratio(
        original_size: int,
        compressed_size: int
    ) -> float:
        """Calculate compression ratio"""
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size
    
    @staticmethod
    def calculate_bitrate(
        data_size_bytes: int,
        duration_seconds: float
    ) -> float:
        """Calculate bitrate in bits per second"""
        if duration_seconds == 0:
            return 0
        return (data_size_bytes * 8) / duration_seconds
    
    @staticmethod
    def calculate_bandwidth_reduction(
        original_bitrate: float,
        compressed_bitrate: float
    ) -> float:
        """Calculate bandwidth reduction percentage"""
        if original_bitrate == 0:
            return 0
        return ((original_bitrate - compressed_bitrate) / original_bitrate) * 100
    
    @staticmethod
    def compare_with_standards(audio_bitrate: float, text_bitrate: float) -> Dict:
        """
        Compare semantic coding with standard audio codecs
        
        Returns:
            Dictionary with comparisons
        """
        # Standard codec bitrates (bps)
        standards = {
            'CD Quality': 1411200,  # 44.1 kHz, 16-bit stereo
            'MP3 High': 320000,
            'MP3 Medium': 128000,
            'MP3 Low': 64000,
            'Telephony (PCM)': 64000,  # 8 kHz, 8-bit
            'Opus Low': 32000,
            'Speex': 24600,
            'GSM': 13000,
            'Codec2': 2400  # Ultra-low bitrate vocoder
        }
        
        comparisons = {}
        for name, bitrate in standards.items():
            ratio = bitrate / text_bitrate if text_bitrate > 0 else 0
            comparisons[name] = {
                'bitrate_bps': bitrate,
                'ratio_vs_semantic': ratio,
                'savings_percent': ((bitrate - text_bitrate) / bitrate * 100) if bitrate > 0 else 0
            }
        
        return comparisons


class TranscriptionMetrics:
    """Calculate transcription quality metrics"""
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Transcribed text
            
        Returns:
            WER as a float (0.0 = perfect, 1.0 = completely wrong)
        """
        if not reference or not hypothesis:
            return 1.0
        
        error_rate = wer(reference, hypothesis)
        return error_rate
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Transcribed text
            
        Returns:
            CER as a float
        """
        if not reference or not hypothesis:
            return 1.0
        
        error_rate = cer(reference, hypothesis)
        return error_rate
    
    @staticmethod
    def calculate_accuracy(reference: str, hypothesis: str) -> float:
        """
        Calculate transcription accuracy (1 - WER)
        
        Returns:
            Accuracy percentage (0-100)
        """
        error_rate = TranscriptionMetrics.calculate_wer(reference, hypothesis)
        accuracy = (1 - error_rate) * 100
        return max(0, accuracy)  # Clamp to [0, 100]
    
    @staticmethod
    def detailed_error_analysis(reference: str, hypothesis: str) -> Dict:
        """
        Detailed error analysis
        
        Returns:
            Dictionary with insertions, deletions, substitutions
        """
        from jiwer import compute_measures
        
        measures = compute_measures(reference, hypothesis)
        
        return {
            'wer': measures['wer'],
            'mer': measures['mer'],  # Match Error Rate
            'wil': measures['wil'],  # Word Information Lost
            'wip': measures['wip'],  # Word Information Preserved
            'hits': measures['hits'],
            'substitutions': measures['substitutions'],
            'deletions': measures['deletions'],
            'insertions': measures['insertions']
        }


class QualityMetrics:
    """Audio quality metrics (subjective and objective)"""
    
    @staticmethod
    def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            signal: Clean signal
            noise: Noise signal
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    @staticmethod
    def calculate_pesq(reference_audio: np.ndarray, degraded_audio: np.ndarray) -> float:
        """
        Calculate PESQ (Perceptual Evaluation of Speech Quality)
        Note: Requires pesq library
        
        Returns:
            PESQ score (-0.5 to 4.5, higher is better)
        """
        try:
            from pesq import pesq as pesq_metric
            
            # PESQ requires specific sample rates (8000 or 16000)
            sr = Config.SAMPLE_RATE
            if sr not in [8000, 16000]:
                logger.warning(f"PESQ requires 8kHz or 16kHz, got {sr}Hz")
                return -1.0
            
            score = pesq_metric(sr, reference_audio, degraded_audio, 'wb')
            return score
        except ImportError:
            logger.warning("PESQ library not installed. Install with: pip install pesq")
            return -1.0
        except Exception as e:
            logger.error(f"PESQ calculation failed: {e}")
            return -1.0
    
    @staticmethod
    def calculate_stoi(reference_audio: np.ndarray, degraded_audio: np.ndarray) -> float:
        """
        Calculate STOI (Short-Time Objective Intelligibility)
        Note: Requires pystoi library
        
        Returns:
            STOI score (0 to 1, higher is better)
        """
        try:
            from pystoi import stoi
            
            sr = Config.SAMPLE_RATE
            score = stoi(reference_audio, degraded_audio, sr, extended=False)
            return score
        except ImportError:
            logger.warning("pystoi library not installed. Install with: pip install pystoi")
            return -1.0
        except Exception as e:
            logger.error(f"STOI calculation failed: {e}")
            return -1.0


class PerformanceMetrics:
    """System performance metrics (latency, throughput)"""
    
    def __init__(self):
        self.latencies = {
            'recording': [],
            'preprocessing': [],
            'transcription': [],
            'compression': [],
            'transmission': [],
            'decompression': [],
            'synthesis': [],
            'end_to_end': []
        }
    
    def record_latency(self, stage: str, duration: float):
        """Record latency for a specific stage"""
        if stage in self.latencies:
            self.latencies[stage].append(duration)
        else:
            logger.warning(f"Unknown stage: {stage}")
    
    def get_stage_stats(self, stage: str) -> Dict:
        """Get statistics for a specific stage"""
        if stage not in self.latencies or not self.latencies[stage]:
            return {}
        
        values = self.latencies[stage]
        return {
            'mean_ms': np.mean(values) * 1000,
            'std_ms': np.std(values) * 1000,
            'min_ms': np.min(values) * 1000,
            'max_ms': np.max(values) * 1000,
            'median_ms': np.median(values) * 1000
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all stages"""
        all_stats = {}
        for stage in self.latencies.keys():
            all_stats[stage] = self.get_stage_stats(stage)
        return all_stats
    
    def calculate_real_time_factor(
        self,
        processing_time: float,
        audio_duration: float
    ) -> float:
        """
        Calculate Real-Time Factor
        RTF < 1.0 means faster than real-time
        
        Args:
            processing_time: Time taken to process
            audio_duration: Duration of audio
            
        Returns:
            RTF value
        """
        if audio_duration == 0:
            return float('inf')
        return processing_time / audio_duration
    
    def print_summary(self):
        """Print performance summary"""
        print("\n=== Performance Summary ===")
        stats = self.get_all_stats()
        
        for stage, metrics in stats.items():
            if metrics:
                print(f"\n{stage.upper()}:")
                print(f"  Mean: {metrics['mean_ms']:.2f} ms")
                print(f"  Std: {metrics['std_ms']:.2f} ms")
                print(f"  Min: {metrics['min_ms']:.2f} ms")
                print(f"  Max: {metrics['max_ms']:.2f} ms")


class EvaluationReport:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self, output_dir: Path = Config.OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {},
            'compression': {},
            'transcription': {},
            'quality': {},
            'performance': {}
        }
    
    def add_compression_metrics(self, metrics: Dict):
        """Add compression metrics to report"""
        self.report_data['compression'] = metrics
    
    def add_transcription_metrics(self, metrics: Dict):
        """Add transcription metrics to report"""
        self.report_data['transcription'] = metrics
    
    def add_quality_metrics(self, metrics: Dict):
        """Add quality metrics to report"""
        self.report_data['quality'] = metrics
    
    def add_performance_metrics(self, metrics: Dict):
        """Add performance metrics to report"""
        self.report_data['performance'] = metrics
    
    def add_config(self, config_dict: Dict):
        """Add configuration to report"""
        self.report_data['config'] = config_dict
    
    def generate_markdown_report(self) -> str:
        """Generate markdown formatted report"""
        md = "# Ultra-Low Bitrate Semantic Speech Coding - Evaluation Report\n\n"
        md += f"**Generated:** {self.report_data['timestamp']}\n\n"
        
        # Configuration
        md += "## Configuration\n\n"
        if self.report_data['config']:
            for key, value in self.report_data['config'].items():
                md += f"- **{key}:** {value}\n"
        md += "\n"
        
        # Compression Results
        md += "## Compression Results\n\n"
        if self.report_data['compression']:
            comp = self.report_data['compression']
            md += f"- **Compression Ratio:** {comp.get('compression_ratio', 0):.1f}:1\n"
            md += f"- **Audio Bitrate:** {comp.get('audio_bitrate_bps', 0):,.0f} bps\n"
            md += f"- **Text Bitrate:** {comp.get('text_bitrate_bps', 0):.0f} bps\n"
            md += f"- **Bandwidth Reduction:** {comp.get('bandwidth_reduction_percent', 0):.1f}%\n"
        md += "\n"
        
        # Transcription Quality
        md += "## Transcription Quality\n\n"
        if self.report_data['transcription']:
            trans = self.report_data['transcription']
            md += f"- **Word Error Rate (WER):** {trans.get('wer', 0):.2%}\n"
            md += f"- **Character Error Rate (CER):** {trans.get('cer', 0):.2%}\n"
            md += f"- **Accuracy:** {trans.get('accuracy', 0):.1f}%\n"
        md += "\n"
        
        # Performance
        md += "## Performance\n\n"
        if self.report_data['performance']:
            perf = self.report_data['performance']
            md += f"- **End-to-End Latency:** {perf.get('end_to_end_mean_ms', 0):.0f} ms\n"
            md += f"- **Real-Time Factor:** {perf.get('rtf', 0):.2f}x\n"
        md += "\n"
        
        return md
    
    def save_report(self, filename: str = "evaluation_report"):
        """Save report to files (JSON and Markdown)"""
        # Save JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        logger.info(f"JSON report saved: {json_path}")
        
        # Save Markdown
        md_content = self.generate_markdown_report()
        md_path = self.output_dir / f"{filename}.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        logger.info(f"Markdown report saved: {md_path}")
        
        return json_path, md_path
    
    def print_summary(self):
        """Print report summary to console"""
        print(self.generate_markdown_report())


# Example usage
if __name__ == "__main__":
    # Test compression metrics
    print("=== Compression Metrics ===")
    original_size = 320000  # 5 seconds at 64kbps
    compressed_size = 50  # Text representation
    
    ratio = CompressionMetrics.calculate_compression_ratio(original_size, compressed_size)
    print(f"Compression Ratio: {ratio:.1f}:1")
    
    # Compare with standards
    comparisons = CompressionMetrics.compare_with_standards(64000, 100)
    print("\nComparison with Standard Codecs:")
    for name, data in comparisons.items():
        print(f"{name}: {data['ratio_vs_semantic']:.1f}x better, {data['savings_percent']:.1f}% savings")
    
    # Test transcription metrics
    print("\n=== Transcription Metrics ===")
    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "The quick brown fox jumped over the lazy dog"
    
    wer_score = TranscriptionMetrics.calculate_wer(reference, hypothesis)
    accuracy = TranscriptionMetrics.calculate_accuracy(reference, hypothesis)
    
    print(f"WER: {wer_score:.2%}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Test report generation
    print("\n=== Generating Report ===")
    report = EvaluationReport()
    report.add_compression_metrics({
        'compression_ratio': ratio,
        'audio_bitrate_bps': 64000,
        'text_bitrate_bps': 100,
        'bandwidth_reduction_percent': 99.84
    })
    report.add_transcription_metrics({
        'wer': wer_score,
        'accuracy': accuracy
    })
    
    report.print_summary()