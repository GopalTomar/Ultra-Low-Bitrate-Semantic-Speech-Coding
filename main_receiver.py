"""
Main Receiver Application
Receives semantic packets, decompresses, and synthesizes speech
This is the "decoder" side of the system
"""

import asyncio
import logging
import time
from pathlib import Path
import argparse

from config import Config
from speech_decoder import SpeechDecoder
from compression_engine import TextCompressor, SemanticPacket
from network_layer import SemanticServer
from audio_utils import save_audio
from metrics_evaluator import (
    PerformanceMetrics,
    EvaluationReport
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReceiverApplication:
    """
    Complete receiver application
    Orchestrates the entire decoding pipeline
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        
        # Initialize components
        logger.info("Initializing components...")
        self.decoder = SpeechDecoder(config)
        self.compressor = TextCompressor()
        self.performance = PerformanceMetrics()
        
        # Statistics
        self.packets_received = 0
        self.total_text_length = 0
        self.total_audio_duration = 0
        
        logger.info("✓ Receiver application initialized")
    
    def process_packet(
        self,
        packet: SemanticPacket,
        auto_play: bool = True,
        save_audio_file: bool = False
    ) -> dict:
        """
        Process received semantic packet
        
        Args:
            packet: Semantic packet to process
            auto_play: Whether to play audio automatically
            save_audio_file: Whether to save audio to file
            
        Returns:
            Dictionary with results and metrics
        """
        results = {
            'success': False,
            'audio': None,
            'metrics': {}
        }
        
        try:
            text = packet.text
            logger.info(f"Processing packet: '{text}'")
            
            # Step 1: Decompression (already done when unpacking)
            t_start = time.time()
            # Text is already decompressed in packet
            t_decompress = time.time() - t_start
            self.performance.record_latency('decompression', t_decompress)
            
            # Step 2: Text-to-Speech (Decoding)
            logger.info("Synthesizing speech...")
            t_start = time.time()
            
            # Save path if requested
            output_path = None
            if save_audio_file:
                output_path = self.config.OUTPUT_DIR / f"output_{self.packets_received:03d}.wav"
            
            audio = self.decoder.decode(text, str(output_path) if output_path else None)
            t_synthesis = time.time() - t_start
            self.performance.record_latency('synthesis', t_synthesis)
            
            if len(audio) == 0:
                logger.warning("Empty audio generated")
                return results
            
            # Calculate metrics
            audio_duration = len(audio) / self.config.SAMPLE_RATE
            synthesis_metrics = self.decoder.get_synthesis_metrics(text, audio)
            
            logger.info(f"Synthesized {audio_duration:.2f}s of audio")
            logger.info(f"Speaking rate: {synthesis_metrics['speaking_rate_wpm']:.0f} WPM")
            
            # Step 3: Playback
            if auto_play:
                logger.info("Playing audio...")
                t_start = time.time()
                
                import sounddevice as sd
                sd.play(audio, self.config.SAMPLE_RATE)
                sd.wait()
                
                t_playback = time.time() - t_start
                logger.info(f"Playback completed in {t_playback:.2f}s")
            
            # Update statistics
            self.packets_received += 1
            self.total_text_length += len(text)
            self.total_audio_duration += audio_duration
            
            # Calculate end-to-end metrics
            t_total = t_decompress + t_synthesis
            self.performance.record_latency('end_to_end', t_total)
            
            # Package results
            results['success'] = True
            results['audio'] = audio
            results['metrics'] = {
                'text_length': len(text),
                'audio_duration_seconds': audio_duration,
                'decompression_time_ms': t_decompress * 1000,
                'synthesis_time_ms': t_synthesis * 1000,
                'total_processing_time_ms': t_total * 1000,
                'speaking_rate_wpm': synthesis_metrics['speaking_rate_wpm']
            }
            
            # Print summary
            print("\n--- Packet Processed ---")
            print(f"Text: '{text}'")
            print(f"Audio Duration: {audio_duration:.2f}s")
            print(f"Processing Time: {t_total:.3f}s")
            
            rtf = self.performance.calculate_real_time_factor(t_total, audio_duration)
            print(f"Real-Time Factor: {rtf:.2f}x")
            if rtf < 1.0:
                print("✓ Faster than real-time!")
            else:
                print("✗ Slower than real-time")
            print()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}", exc_info=True)
            return results
    
    def handle_received_packet(self, packet: SemanticPacket) -> str:
        """
        Callback for handling received packets
        
        Args:
            packet: Received semantic packet
            
        Returns:
            Response text (if any)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RECEIVED PACKET FROM: {packet.speaker_id or 'unknown'}")
        logger.info(f"{'='*60}")
        
        # Process packet
        results = self.process_packet(
            packet,
            auto_play=True,
            save_audio_file=self.config.SAVE_INTERMEDIATE_FILES
        )
        
        if results['success']:
            # Could send acknowledgment or response
            return None  # No response for now
        else:
            logger.error("Failed to process packet")
            return None
    
    async def start_server(
        self,
        host: str = Config.SERVER_HOST,
        port: int = Config.SERVER_PORT
    ):
        """
        Start the receiver server
        
        Args:
            host: Server host
            port: Server port
        """
        logger.info(f"\n{'='*60}")
        logger.info("STARTING RECEIVER SERVER")
        logger.info(f"{'='*60}\n")
        
        server = SemanticServer(
            host=host,
            port=port,
            on_receive_callback=self.handle_received_packet
        )
        
        try:
            await server.start()
        except KeyboardInterrupt:
            logger.info("\nServer stopped by user")
        finally:
            logger.info("Generating final report...")
            self.generate_report("receiver_server_report")
    
    async def client_mode(self, server_uri: str):
        """
        Run in client mode (for testing without sender)
        
        Args:
            server_uri: Server URI to connect to
        """
        from network_layer import SemanticClient
        
        logger.info(f"\n{'='*60}")
        logger.info("RECEIVER CLIENT MODE (Testing)")
        logger.info(f"{'='*60}\n")
        
        client = SemanticClient(server_uri=server_uri)
        
        try:
            await client.connect()
            
            # Interactive session
            while True:
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "\nEnter text to synthesize (or 'quit'): "
                )
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                
                if command.strip():
                    # Create packet
                    packet = SemanticPacket(
                        text=command,
                        speaker_id="test_user",
                        language="en"
                    )
                    
                    # Process locally
                    self.process_packet(packet, auto_play=True, save_audio_file=False)
        
        except Exception as e:
            logger.error(f"Client mode error: {e}")
        finally:
            await client.disconnect()
            self.generate_report("receiver_client_report")
    
    async def test_mode(self, no_autoplay: bool = False, save_files: bool = False):
        """
        Test mode for local synthesis without network
        
        Args:
            no_autoplay: Disable automatic playback
            save_files: Save audio files
        """
        logger.info(f"\n{'='*60}")
        logger.info("RECEIVER TEST MODE (Local Synthesis)")
        logger.info(f"{'='*60}\n")
        
        # Test with sample texts
        test_texts = [
            "Hello, this is a test of the semantic speech coding system.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing ultra-low bitrate compression with text-to-speech synthesis.",
            "This project demonstrates speech compression using semantic tokens.",
            "Voice activity detection and noise reduction improve audio quality."
        ]
        
        logger.info(f"Testing with {len(test_texts)} sample texts...\n")
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST {i}/{len(test_texts)}")
            logger.info(f"{'='*60}")
            
            # Create semantic packet
            packet = SemanticPacket(
                text=text,
                speaker_id="test_system",
                language="en"
            )
            
            # Process packet
            results = self.process_packet(
                packet,
                auto_play=not no_autoplay,
                save_audio_file=save_files
            )
            
            if results['success']:
                logger.info(f"✓ Test {i} completed successfully")
            else:
                logger.error(f"✗ Test {i} failed")
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Generate final report
        logger.info("\nAll tests completed. Generating report...")
        self.generate_report("receiver_test_report")
    
    def generate_report(self, output_filename: str = "receiver_report"):
        """Generate evaluation report"""
        logger.info("\nGenerating evaluation report...")
        
        report = EvaluationReport()
        
        # Add configuration
        report.add_config({
            'tts_engine': self.config.TTS_ENGINE,
            'sample_rate': self.config.SAMPLE_RATE,
            'tts_language': self.config.TTS_LANGUAGE
        })
        
        # Add performance metrics
        perf_stats = self.performance.get_all_stats()
        report.add_performance_metrics(perf_stats)
        
        # Add receiver statistics
        report.report_data['receiver_stats'] = {
            'packets_received': self.packets_received,
            'total_text_length': self.total_text_length,
            'total_audio_duration': self.total_audio_duration,
            'average_packet_size': self.total_text_length / self.packets_received if self.packets_received > 0 else 0
        }
        
        # Save report
        json_path, md_path = report.save_report(output_filename)
        
        # Print summary
        report.print_summary()
        
        # Print receiver stats
        print("\n=== Receiver Statistics ===")
        print(f"Packets Received: {self.packets_received}")
        print(f"Total Text Length: {self.total_text_length} characters")
        print(f"Total Audio Generated: {self.total_audio_duration:.2f} seconds")
        if self.packets_received > 0:
            print(f"Average Packet Size: {self.total_text_length / self.packets_received:.1f} characters")
        
        return report
    
    async def interactive_mode(self):
        """
        Interactive mode for manual testing
        """
        logger.info(f"\n{'='*60}")
        logger.info("INTERACTIVE MODE")
        logger.info(f"{'='*60}\n")
        logger.info("Commands:")
        logger.info("  's <text>' - Synthesize text")
        logger.info("  'stats'    - Show statistics")
        logger.info("  'report'   - Generate report")
        logger.info("  'q'        - Quit")
        print()
        
        while True:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "\nCommand: "
                )
                
                command = command.strip()
                
                if not command:
                    continue
                
                if command.lower() in ['q', 'quit', 'exit']:
                    break
                
                elif command.lower() == 'stats':
                    self.performance.print_summary()
                    print(f"\nPackets Processed: {self.packets_received}")
                    print(f"Total Text: {self.total_text_length} characters")
                    print(f"Total Audio: {self.total_audio_duration:.2f}s")
                
                elif command.lower() == 'report':
                    self.generate_report("receiver_interactive_report")
                
                elif command.lower().startswith('s '):
                    text = command[2:].strip()
                    if text:
                        packet = SemanticPacket(
                            text=text,
                            speaker_id="interactive_user",
                            language="en"
                        )
                        self.process_packet(packet, auto_play=True, save_audio_file=False)
                    else:
                        logger.warning("No text provided")
                
                else:
                    logger.warning(f"Unknown command: {command}")
                    logger.info("Type 's <text>' to synthesize, 'stats', 'report', or 'q' to quit")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info("\nExiting interactive mode")
        self.generate_report("receiver_interactive_report")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Semantic Speech Coding - Receiver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server to receive packets
  python main_receiver.py --mode server
  
  # Test mode with sample texts
  python main_receiver.py --mode test
  
  # Interactive mode for manual testing
  python main_receiver.py --mode interactive
  
  # Use different TTS engine
  python main_receiver.py --mode test --tts-engine pyttsx3
  
  # Save synthesized audio files
  python main_receiver.py --mode test --save-files
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['server', 'client', 'test', 'interactive'],
        default='server',
        help='Operating mode (default: server)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=Config.SERVER_HOST,
        help=f'Server host (default: {Config.SERVER_HOST})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=Config.SERVER_PORT,
        help=f'Server port (default: {Config.SERVER_PORT})'
    )
    parser.add_argument(
        '--server-uri',
        type=str,
        default=f"ws://{Config.SERVER_HOST}:{Config.SERVER_PORT}",
        help='Server URI for client mode'
    )
    parser.add_argument(
        '--tts-engine',
        type=str,
        choices=['gtts', 'pyttsx3', 'coqui'],
        default=Config.TTS_ENGINE,
        help=f'TTS engine to use (default: {Config.TTS_ENGINE})'
    )
    parser.add_argument(
        '--no-autoplay',
        action='store_true',
        help='Disable automatic audio playback'
    )
    parser.add_argument(
        '--save-files',
        action='store_true',
        help='Save synthesized audio files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Config.OUTPUT_DIR),
        help='Output directory for files'
    )
    
    args = parser.parse_args()
    
    # Update config
    Config.TTS_ENGINE = args.tts_engine
    Config.SAVE_INTERMEDIATE_FILES = args.save_files
    Config.OUTPUT_DIR = Path(args.output_dir)
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Print configuration
    Config.print_config()
    
    # Initialize application
    app = ReceiverApplication()
    
    # Run based on mode
    try:
        if args.mode == 'server':
            logger.info("Starting in SERVER mode...")
            logger.info("Waiting for connections from sender...")
            await app.start_server(host=args.host, port=args.port)
        
        elif args.mode == 'client':
            logger.info("Starting in CLIENT mode...")
            await app.client_mode(server_uri=args.server_uri)
        
        elif args.mode == 'test':
            logger.info("Starting in TEST mode (local synthesis)...")
            await app.test_mode(
                no_autoplay=args.no_autoplay,
                save_files=args.save_files
            )
        
        elif args.mode == 'interactive':
            logger.info("Starting in INTERACTIVE mode...")
            await app.interactive_mode()
        
        logger.info("\nReceiver application completed successfully")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\nApplication stopped by user")
        exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)