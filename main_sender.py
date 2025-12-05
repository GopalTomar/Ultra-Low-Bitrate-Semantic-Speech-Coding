"""
Main Sender Application
Records audio, transcribes it, and sends semantic packets
This is the "encoder" side of the system
"""

import asyncio
import logging
import time
import numpy as np
from pathlib import Path
import argparse

from config import Config
from audio_utils import AudioRecorder, AudioPreprocessor, AudioAnalyzer, save_audio
from speech_encoder import SpeechEncoder
from compression_engine import TextCompressor, SemanticPacket
from network_layer import SemanticClient
from metrics_evaluator import (
    CompressionMetrics,
    PerformanceMetrics,
    EvaluationReport
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SenderApplication:
    """
    Complete sender application
    Orchestrates the entire encoding pipeline
    """

    def __init__(self, config: Config = Config):
        self.config = config

        # Initialize components
        logger.info("Initializing components...")
        self.recorder = AudioRecorder(config)
        self.preprocessor = AudioPreprocessor(config)
        self.encoder = SpeechEncoder(config)
        self.compressor = TextCompressor()
        self.analyzer = AudioAnalyzer()
        self.performance = PerformanceMetrics()

        # Network client (initialized on demand)
        self.client = None

        logger.info("✓ Sender application initialized")

    def process_audio(
        self,
        audio: np.ndarray,
        save_files: bool = False
    ) -> dict:
        """
        Process audio through the complete pipeline

        Args:
            audio: Raw audio samples
            save_files: Whether to save intermediate files

        Returns:
            Dictionary with results and metrics
        """
        results = {
            'success': False,
            'text': '',
            'audio_duration': 0,
            'metrics': {}
        }

        try:
            # Step 1: Preprocessing
            logger.info("Step 1: Preprocessing audio...")
            t_start = time.time()
            clean_audio = self.preprocessor.preprocess(audio)
            t_preprocess = time.time() - t_start
            self.performance.record_latency('preprocessing', t_preprocess)

            if save_files:
                save_audio(
                    clean_audio,
                    self.config.TEMP_DIR / "preprocessed.wav"
                )

            # Calculate audio metrics
            audio_duration = self.analyzer.calculate_duration(
                clean_audio,
                self.config.SAMPLE_RATE
            )
            audio_size = self.analyzer.calculate_size_bytes(clean_audio)
            audio_bitrate = self.analyzer.calculate_bitrate(
                clean_audio,
                self.config.SAMPLE_RATE
            )

            logger.info(
                f"Audio: {audio_duration:.2f}s, {audio_size:,} bytes, {audio_bitrate:,.0f} bps"
            )

            # Step 2: Speech-to-Text (Encoding)
            logger.info("Step 2: Transcribing audio...")
            t_start = time.time()
            transcription = self.encoder.encode(clean_audio)
            t_transcribe = time.time() - t_start
            self.performance.record_latency('transcription', t_transcribe)

            text = transcription['text']
            logger.info(f"Transcribed: '{text}'")

            if not text:
                logger.warning("Empty transcription")
                return results

            # Step 3: Compression
            logger.info("Step 3: Compressing text...")
            t_start = time.time()
            compressed = self.compressor.compress(text, method="zlib")
            t_compress = time.time() - t_start
            self.performance.record_latency('compression', t_compress)

            # Calculate compression metrics
            text_size = len(text.encode('utf-8'))
            compressed_size = len(compressed)
            text_bitrate = (text_size * 8) / audio_duration if audio_duration > 0 else 0
            compression_ratio = CompressionMetrics.calculate_compression_ratio(
                audio_size,
                text_size
            )

            logger.info(f"Text: {text_size} bytes → {compressed_size} bytes compressed")
            logger.info(f"Compression Ratio: {compression_ratio:.1f}:1")
            logger.info(f"Bitrate Reduction: {audio_bitrate:.0f} bps → {text_bitrate:.0f} bps")

            # Package results
            results['success'] = True
            results['text'] = text
            results['audio_duration'] = audio_duration
            results['compressed_data'] = compressed
            results['metrics'] = {
                'audio_size_bytes': audio_size,
                'audio_bitrate_bps': audio_bitrate,
                'text_size_bytes': text_size,
                'compressed_size_bytes': compressed_size,
                'text_bitrate_bps': text_bitrate,
                'compression_ratio': compression_ratio,
                'preprocessing_time_ms': t_preprocess * 1000,
                'transcription_time_ms': t_transcribe * 1000,
                'compression_time_ms': t_compress * 1000,
                'total_processing_time_ms': (
                    t_preprocess + t_transcribe + t_compress
                ) * 1000
            }

            return results

        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return results

    async def send_packet(self, text: str, speaker_id: str = "user"):
        """
        Send semantic packet over network

        Args:
            text: Text to send
            speaker_id: Speaker identifier
        """
        if self.client is None or not self.client.connected:
            logger.error("Client not connected")
            return

        try:
            t_start = time.time()
            await self.client.send(text, speaker_id)
            t_send = time.time() - t_start
            self.performance.record_latency('transmission', t_send)
            logger.info(f"Packet sent in {t_send*1000:.2f} ms")
        except Exception as e:
            logger.error(f"Failed to send packet: {e}")

    async def record_and_send(self, duration: float = None):
        """
        Record audio and send it through the network

        Args:
            duration: Recording duration (None for VAD)
        """
        logger.info("\n" + "=" * 60)
        logger.info("RECORDING AND SENDING")
        logger.info("=" * 60)

        # Record
        t_start = time.time()
        audio = self.recorder.record_audio(duration)
        t_record = time.time() - t_start
        self.performance.record_latency('recording', t_record)

        if len(audio) == 0:
            logger.warning("No audio recorded")
            return

        # Process
        results = self.process_audio(audio, save_files=self.config.SAVE_INTERMEDIATE_FILES)

        if not results['success']:
            logger.error("Audio processing failed")
            return

        # Send
        if self.client and self.client.connected:
            await self.send_packet(results['text'])

        # Calculate end-to-end time
        t_total = (
            t_record +
            results['metrics']['preprocessing_time_ms'] / 1000 +
            results['metrics']['transcription_time_ms'] / 1000 +
            results['metrics']['compression_time_ms'] / 1000
        )
        self.performance.record_latency('end_to_end', t_total)

        # Print summary
        print("\n--- Processing Summary ---")
        print(f"Transcribed Text: '{results['text']}'")
        print(f"Audio Duration: {results['audio_duration']:.2f}s")
        print(f"Compression Ratio: {results['metrics']['compression_ratio']:.1f}:1")
        print(f"Total Processing Time: {t_total:.2f}s")

        rtf = self.performance.calculate_real_time_factor(
            t_total,
            results['audio_duration']
        )
        print(f"Real-Time Factor: {rtf:.2f}x")
        if rtf < 1.0:
            print("✓ Faster than real-time!")
        else:
            print("✗ Slower than real-time")
        print()

    async def interactive_mode(self):
        """Interactive mode for continuous recording and sending"""
        logger.info("\n=== INTERACTIVE MODE ===")
        logger.info("Commands:")
        logger.info("  'r' or 'record' - Record and send audio")
        logger.info("  'c' or 'connect' - Connect to server")
        logger.info("  'd' or 'disconnect' - Disconnect from server")
        logger.info("  's' or 'stats' - Show statistics")
        logger.info("  'q' or 'quit' - Exit\n")

        while True:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nCommand: "
                )
                command = command.lower().strip()

                if command in ['q', 'quit', 'exit']:
                    break

                elif command in ['r', 'record']:
                    await self.record_and_send()

                elif command in ['c', 'connect']:
                    if self.client is None:
                        self.client = SemanticClient()

                    if not self.client.connected:
                        await self.client.connect()
                    else:
                        logger.info("Already connected")

                elif command in ['d', 'disconnect']:
                    if self.client and self.client.connected:
                        await self.client.disconnect()
                    else:
                        logger.info("Not connected")

                elif command in ['s', 'stats']:
                    self.performance.print_summary()

                else:
                    logger.warning(f"Unknown command: {command}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        # Cleanup
        if self.client and self.client.connected:
            await self.client.disconnect()

        logger.info("Exiting interactive mode")

    def generate_report(self, output_filename: str = "sender_report"):
        """Generate evaluation report"""
        logger.info("Generating evaluation report...")

        report = EvaluationReport()

        # Add configuration
        report.add_config({
            'whisper_model': self.config.WHISPER_MODEL,
            'sample_rate': self.config.SAMPLE_RATE,
            'vad_mode': self.config.VAD_MODE,
            'noise_reduction': self.config.NOISE_REDUCE_ENABLED
        })

        # Add performance metrics
        perf_stats = self.performance.get_all_stats()
        report.add_performance_metrics(perf_stats)

        # Add compression stats
        comp_stats = self.compressor.get_stats()
        report.add_compression_metrics(comp_stats)

        # Save report
        json_path, md_path = report.save_report(output_filename)

        # Print summary
        report.print_summary()

        return report


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Semantic Speech Coding - Sender")
    parser.add_argument('--mode', choices=['single', 'interactive', 'batch'], default='interactive',
                       help='Operating mode')
    parser.add_argument('--duration', type=float, default=None,
                       help='Recording duration in seconds (None for VAD)')
    parser.add_argument('--connect', action='store_true',
                       help='Auto-connect to server')
    parser.add_argument('--server', type=str,
                        default=f"ws://{Config.SERVER_HOST}:{Config.SERVER_PORT}",
                        help='Server URI')
    parser.add_argument('--output-dir', type=str, default=str(Config.OUTPUT_DIR),
                       help='Output directory for reports')
    parser.add_argument('--save-files', action='store_true',
                       help='Save intermediate audio files')

    args = parser.parse_args()

    # Update config
    Config.SAVE_INTERMEDIATE_FILES = args.save_files
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.print_config()

    # Initialize application
    app = SenderApplication()

    # Connect to server if requested
    if args.connect:
        try:
            app.client = SemanticClient(server_uri=args.server)
            await app.client.connect()
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            logger.info("Continuing in offline mode...")

    # Run based on mode
    if args.mode == 'single':
        logger.info("Single recording mode")
        await app.record_and_send(duration=args.duration)
        app.generate_report("sender_single_report")

    elif args.mode == 'interactive':
        await app.interactive_mode()
        app.generate_report("sender_interactive_report")

    elif args.mode == 'batch':
        logger.info("Batch mode - recording multiple samples")
        num_samples = 5
        for i in range(num_samples):
            logger.info(f"\n--- Sample {i+1}/{num_samples} ---")
            await app.record_and_send(duration=args.duration)
            await asyncio.sleep(1)

        app.generate_report("sender_batch_report")

    logger.info("\nSender application completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
