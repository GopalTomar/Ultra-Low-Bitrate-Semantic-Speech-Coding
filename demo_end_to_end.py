"""
Complete End-to-End Demo
Demonstrates the full semantic speech coding pipeline locally
without needing separate sender/receiver processes.

Updates:
- Added Signal Visualization (Waveform/Spectrogram)
- Enhanced error handling
- Improved interactive flow
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
import time
import argparse

# Project Imports
from config import Config
from audio_utils import AudioRecorder, AudioPreprocessor, save_audio, load_audio
from speech_encoder import SpeechEncoder
from speech_decoder import SpeechDecoder
from compression_engine import TextCompressor, SemanticPacket
from metrics_evaluator import (
    CompressionMetrics,
    TranscriptionMetrics,
    PerformanceMetrics,
    EvaluationReport
)

# Import the new Visualizer
try:
    from visualizer import AudioVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    print("Warning: visualizer.py not found. Plotting disabled.")
    VISUALIZER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndDemo:
    """
    End-to-end demonstration of semantic speech coding
    """

    def __init__(self, config: Config = Config):
        self.config = config

        # Initialize all components
        logger.info("Initializing pipeline components...")
        self.recorder = AudioRecorder(config)
        self.preprocessor = AudioPreprocessor(config)
        self.encoder = SpeechEncoder(config)
        self.decoder = SpeechDecoder(config)
        self.compressor = TextCompressor()
        self.performance = PerformanceMetrics()

        logger.info("✓ Pipeline initialized")

    def run_complete_pipeline(
        self,
        audio: np.ndarray,
        reference_text: str = None,
        show_plots: bool = True
    ) -> dict:
        """
        Run the complete pipeline on audio

        Args:
            audio: Input audio samples
            reference_text: Ground truth text (for WER calculation)
            show_plots: Whether to display comparison plots

        Returns:
            Dictionary with all results and metrics
        """
        results = {
            'success': False,
            'stages': {},
            'metrics': {},
            'comparison': {}
        }

        try:
            logger.info("\n" + "=" * 70)
            logger.info("RUNNING COMPLETE PIPELINE")
            logger.info("=" * 70)

            # ============ STAGE 1: PREPROCESSING ============
            logger.info("\n[1/5] Preprocessing (VAD & Noise Reduction)...")
            t_start = time.time()
            clean_audio = self.preprocessor.preprocess(audio)
            t_preprocess = time.time() - t_start
            self.performance.record_latency('preprocessing', t_preprocess)

            # Calculate original audio metrics
            audio_duration = len(clean_audio) / self.config.SAMPLE_RATE
            audio_size = clean_audio.nbytes
            audio_bitrate = (audio_size * 8) / audio_duration if audio_duration > 0 else 0

            logger.info(f"✓ Audio preprocessed: {audio_duration:.2f}s, {audio_size:,} bytes")

            results['stages']['preprocessing'] = {
                'duration': audio_duration,
                'size_bytes': audio_size,
                'bitrate_bps': audio_bitrate,
                'time_ms': t_preprocess * 1000
            }

            if len(clean_audio) == 0:
                logger.error("Audio is empty after preprocessing!")
                return results

            # ============ STAGE 2: SPEECH-TO-TEXT (ENCODING) ============
            logger.info("\n[2/5] Encoding (Acoustic -> Semantic)...")
            t_start = time.time()
            transcription = self.encoder.encode(clean_audio)
            t_encode = time.time() - t_start
            self.performance.record_latency('transcription', t_encode)

            text = transcription['text']
            
            if not text:
                logger.warning("No speech detected by Whisper.")
                return results

            logger.info(f"✓ Transcribed: '{text}'")
            logger.info(f"  Language: {transcription['language']}")
            logger.info(f"  Time: {t_encode:.2f}s")

            results['stages']['encoding'] = {
                'text': text,
                'language': transcription['language'],
                'time_ms': t_encode * 1000
            }

            # ============ STAGE 3: COMPRESSION ============
            logger.info("\n[3/5] Compressing Text Tokens...")
            t_start = time.time()

            # Create semantic packet
            packet = SemanticPacket(
                text=text,
                speaker_id="demo_user",
                language=transcription['language']
            )

            # Pack and compress
            packed_packet = packet.pack(self.compressor)
            t_compress = time.time() - t_start
            self.performance.record_latency('compression', t_compress)

            text_size = len(text.encode('utf-8'))
            packet_size = len(packed_packet)
            text_bitrate = (text_size * 8) / audio_duration if audio_duration > 0 else 0

            compression_ratio = CompressionMetrics.calculate_compression_ratio(
                audio_size,
                packet_size  # Comparing raw audio to compressed packet
            )

            logger.info(f"✓ Compressed: {text_size} bytes (raw text) → {packet_size} bytes (packet)")
            logger.info(f"  Compression Ratio: {compression_ratio:.1f}:1")
            logger.info(f"  Bitrate: {audio_bitrate:,.0f} bps → {text_bitrate:.0f} bps")

            results['stages']['compression'] = {
                'text_size_bytes': text_size,
                'packet_size_bytes': packet_size,
                'text_bitrate_bps': text_bitrate,
                'compression_ratio': compression_ratio,
                'time_ms': t_compress * 1000
            }

            # ============ STAGE 4: DECOMPRESSION ============
            logger.info("\n[4/5] Decompressing...")
            t_start = time.time()
            unpacked_packet = SemanticPacket.unpack(packed_packet, self.compressor)
            t_decompress = time.time() - t_start
            self.performance.record_latency('decompression', t_decompress)

            logger.info(f"✓ Decompressed: '{unpacked_packet.text}'")

            results['stages']['decompression'] = {
                'text': unpacked_packet.text,
                'time_ms': t_decompress * 1000
            }

            # ============ STAGE 5: TEXT-TO-SPEECH (DECODING) ============
            logger.info("\n[5/5] Decoding (Semantic -> Acoustic)...")
            t_start = time.time()
            synthesized_audio = self.decoder.decode(unpacked_packet.text)
            t_decode = time.time() - t_start
            self.performance.record_latency('synthesis', t_decode)

            synth_duration = len(synthesized_audio) / self.config.SAMPLE_RATE
            synth_size = synthesized_audio.nbytes

            logger.info(f"✓ Synthesized: {synth_duration:.2f}s audio")
            logger.info(f"  Time: {t_decode:.2f}s")

            results['stages']['decoding'] = {
                'audio_duration': synth_duration,
                'audio_size_bytes': synth_size,
                'time_ms': t_decode * 1000
            }

            # ============ CALCULATE OVERALL METRICS ============
            logger.info("\n" + "=" * 70)
            logger.info("CALCULATING METRICS")
            logger.info("=" * 70)

            # Total processing time
            t_total = t_preprocess + t_encode + t_compress + t_decompress + t_decode
            self.performance.record_latency('end_to_end', t_total)

            # Real-time factor
            rtf = self.performance.calculate_real_time_factor(t_total, audio_duration)

            # Compression metrics
            bandwidth_reduction = CompressionMetrics.calculate_bandwidth_reduction(
                audio_bitrate,
                text_bitrate
            )

            # Transcription quality (if reference provided)
            wer_score = None
            accuracy = None
            if reference_text:
                wer_score = TranscriptionMetrics.calculate_wer(reference_text, text)
                accuracy = TranscriptionMetrics.calculate_accuracy(reference_text, text)
                logger.info(f"\nTranscription Quality:")
                logger.info(f"  WER: {wer_score:.2%}")
                logger.info(f"  Accuracy: {accuracy:.1f}%")

            # Package metrics
            results['metrics'] = {
                'total_processing_time_ms': t_total * 1000,
                'real_time_factor': rtf,
                'compression_ratio': compression_ratio,
                'bandwidth_reduction_percent': bandwidth_reduction,
                'original_bitrate_bps': audio_bitrate,
                'semantic_bitrate_bps': text_bitrate,
                'wer': wer_score,
                'accuracy': accuracy
            }

            # Compare with standard codecs
            comparisons = CompressionMetrics.compare_with_standards(
                audio_bitrate,
                text_bitrate
            )
            results['comparison'] = comparisons

            # Store audio for playback/vis
            results['original_audio'] = clean_audio
            results['synthesized_audio'] = synthesized_audio
            results['success'] = True

            # ============ VISUALIZATION ============
            if show_plots and VISUALIZER_AVAILABLE and len(clean_audio) > 0 and len(synthesized_audio) > 0:
                logger.info("Generating signal comparison plots...")
                AudioVisualizer.plot_comparison(
                    clean_audio, 
                    synthesized_audio, 
                    self.config.SAMPLE_RATE
                )

            # ============ PRINT SUMMARY ============
            self.print_summary(results)

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return results

    def print_summary(self, results: dict):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        metrics = results['metrics']

        print("\n📊 COMPRESSION PERFORMANCE:")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.1f}:1")
        print(f"  Original Bitrate: {metrics['original_bitrate_bps']:,.0f} bps")
        print(f"  Semantic Bitrate: {metrics['semantic_bitrate_bps']:.0f} bps")
        print(f"  Bandwidth Reduction: {metrics['bandwidth_reduction_percent']:.2f}%")

        print("\n⚡ PROCESSING PERFORMANCE:")
        print(f"  Total Processing Time: {metrics['total_processing_time_ms']:.0f} ms")
        print(f"  Real-Time Factor: {metrics['real_time_factor']:.2f}x")
        if metrics['real_time_factor'] < 1.0:
            print("  ✓ FASTER than real-time!")
        else:
            print("  ✗ SLOWER than real-time")

        if metrics['wer'] is not None:
            print("\n📝 TRANSCRIPTION QUALITY:")
            print(f"  Word Error Rate: {metrics['wer']:.2%}")
            print(f"  Accuracy: {metrics['accuracy']:.1f}%")

        print("\n📡 COMPARISON WITH STANDARD CODECS:")
        comparisons = results['comparison']
        # Filter for relevant codecs
        targets = ['MP3 High', 'MP3 Low', 'Telephony (PCM)', 'GSM', 'Opus Low']
        for codec_name in targets:
            if codec_name in comparisons:
                comp = comparisons[codec_name]
                print(f"  vs {codec_name:15s}: {comp['ratio_vs_semantic']:6.1f}x better efficiency")

        print("\n" + "=" * 70)

    async def interactive_demo(self):
        """Interactive demonstration"""
        logger.info("\n" + "=" * 70)
        logger.info("INTERACTIVE DEMO MODE")
        logger.info("=" * 70)
        logger.info("\nCommands:")
        logger.info("  'r' - Record and process audio (End-to-End)")
        logger.info("  'f <path>' - Process audio file")
        logger.info("  'p' - Play last synthesized audio")
        logger.info("  'o' - Play original recorded audio")
        logger.info("  's' - Show statistics")
        logger.info("  'q' - Quit")

        last_results = None

        while True:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "\nCommand (r/f/p/o/s/q): "
                )
                command = command.strip().lower()

                if command in ['q', 'quit', 'exit']:
                    break

                elif command == 'r':
                    # Record audio
                    logger.info("\nRecording audio (speak now)...")
                    # Using a fixed duration for demo stability, but VAD is available
                                        # NEW CODE
                    # duration=None enables VAD. It records until you stop speaking.
                    print("   (Speak now... pause for 2 seconds to finish)")
                    audio = self.recorder.record_audio(duration=None)

                    if len(audio) > 0:
                        # Process through pipeline
                        # This will trigger the Visualizer pop-up
                        last_results = self.run_complete_pipeline(audio, show_plots=True)

                        if last_results['success']:
                            print("\nPlaying Synthesized Audio...")
                            import sounddevice as sd
                            sd.play(last_results['synthesized_audio'], self.config.SAMPLE_RATE)
                            sd.wait()
                    else:
                        logger.warning("No audio recorded")

                elif command.startswith('f '):
                    # Load file
                    filepath = command[2:].strip()
                    try:
                        audio, sr = load_audio(filepath, self.config.SAMPLE_RATE)
                        logger.info(f"Loaded audio file: {filepath}")

                        # Process
                        last_results = self.run_complete_pipeline(audio, show_plots=True)

                        if last_results['success']:
                            import sounddevice as sd
                            sd.play(last_results['synthesized_audio'], self.config.SAMPLE_RATE)
                            sd.wait()
                    except Exception as e:
                        logger.error(f"Failed to load file: {e}")

                elif command == 'p':
                    # Play last synthesized
                    if last_results and last_results['success']:
                        import sounddevice as sd
                        logger.info("Playing synthesized output...")
                        sd.play(last_results['synthesized_audio'], self.config.SAMPLE_RATE)
                        sd.wait()
                    else:
                        logger.warning("No processed audio available")

                elif command == 'o':
                    # Play original
                    if last_results and last_results['success']:
                        import sounddevice as sd
                        logger.info("Playing original input...")
                        sd.play(last_results['original_audio'], self.config.SAMPLE_RATE)
                        sd.wait()
                    else:
                        logger.warning("No recorded audio available")

                elif command == 's':
                    # Show statistics
                    self.performance.print_summary()

                else:
                    pass # Ignore empty enters

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        logger.info("\nExiting demo")

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = EvaluationReport()

        # Add configuration
        report.add_config({
            'whisper_model': self.config.WHISPER_MODEL,
            'tts_engine': self.config.TTS_ENGINE,
            'sample_rate': self.config.SAMPLE_RATE,
            'vad_mode': self.config.VAD_MODE
        })

        # Add performance metrics
        perf_stats = self.performance.get_all_stats()
        report.add_performance_metrics(perf_stats)

        # Add compression stats
        comp_stats = self.compressor.get_stats()
        report.add_compression_metrics(comp_stats)

        # Save and print
        report.save_report("end_to_end_demo_report")
        report.print_summary()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="End-to-End Semantic Speech Coding Demo")
    parser.add_argument('--mode', choices=['interactive', 'file', 'batch'], default='interactive',
                       help='Demo mode')
    parser.add_argument('--input', type=str,
                       help='Input audio file (for file mode)')
    parser.add_argument('--reference', type=str,
                       help='Reference text for WER calculation')
    parser.add_argument('--whisper-model', type=str, default='base',
                       help='Whisper model size')
    parser.add_argument('--tts-engine',
                        type=str, default='gtts',
                        choices=['gtts', 'pyttsx3', 'coqui'],
                        help='TTS engine')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable visualization plots')

    args = parser.parse_args()

    # Update configuration
    Config.WHISPER_MODEL = args.whisper_model
    Config.TTS_ENGINE = args.tts_engine
    Config.print_config()

    # Initialize demo
    demo = EndToEndDemo()

    # Run based on mode
    if args.mode == 'interactive':
        await demo.interactive_demo()

    elif args.mode == 'file':
        if not args.input:
            logger.error("Input file required for file mode")
            return

        # Load audio
        audio, sr = load_audio(args.input, Config.SAMPLE_RATE)

        # Load reference text if provided
        reference_text = None
        if args.reference:
            with open(args.reference, 'r') as f:
                reference_text = f.read().strip()

        # Run pipeline
        results = demo.run_complete_pipeline(
            audio, 
            reference_text, 
            show_plots=not args.no_plot
        )

        # Save synthesized audio
        if results['success']:
            output_path = Config.OUTPUT_DIR / "synthesized_output.wav"
            save_audio(results['synthesized_audio'], str(output_path))
            logger.info(f"Synthesized audio saved to: {output_path}")

        # Generate report
        demo.generate_report()

    elif args.mode == 'batch':
        logger.info("Batch processing mode (Visualizations disabled)")

        # Find all audio files in audio directory
        audio_files = list(Config.AUDIO_DIR.glob("*.wav")) + \
                      list(Config.AUDIO_DIR.glob("*.mp3"))

        if not audio_files:
            logger.warning(f"No audio files found in {Config.AUDIO_DIR}")
            return

        logger.info(f"Found {len(audio_files)} audio files")

        for i, audio_file in enumerate(audio_files):
            logger.info(f"\nProcessing file {i+1}/{len(audio_files)}: {audio_file.name}")
            try:
                audio, sr = load_audio(str(audio_file), Config.SAMPLE_RATE)
                results = demo.run_complete_pipeline(audio, show_plots=False)

                if results['success']:
                    output_path = Config.OUTPUT_DIR / f"output_{audio_file.stem}.wav"
                    save_audio(results['synthesized_audio'], str(output_path))
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")

        demo.generate_report()

    logger.info("\nDemo completed successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nDemo stopped by user")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)