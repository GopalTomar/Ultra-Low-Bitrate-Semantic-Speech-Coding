"""
Visualizer Module
Handles plotting of waveforms and spectrograms for signal comparison.
Crucial for demonstrating Signal Processing concepts.

UPDATED: Includes memory protection for long audio files.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging

logger = logging.getLogger(__name__)

class AudioVisualizer:
    @staticmethod
    def plot_comparison(original_audio, synth_audio, sr, title="Semantic Coding: Signal Comparison"):
        """
        Plots waveforms and spectrograms of Original vs Synthesized audio.
        
        NOTE: Automatically truncates visualization to the first 30 seconds
        if the file is longer, preventing Memory Errors (RAM crashes).
        """
        try:
            # ==========================================
            # 1. MEMORY PROTECTION & DATA PREP
            # ==========================================
            MAX_DURATION = 30  # seconds limit for plotting
            max_samples = int(MAX_DURATION * sr)

            # Check if audio is too long
            if len(original_audio) > max_samples:
                print(f"\n[Visualizer] File is large ({len(original_audio)/sr:.2f}s). Plotting first {MAX_DURATION}s to prevent crash.")
                
                # Truncate original
                plot_orig = original_audio[:max_samples]
                
                # Truncate synthesized (handle cases where synth might be shorter/longer)
                limit_synth = min(len(synth_audio), max_samples)
                plot_synth = synth_audio[:limit_synth]
            else:
                # Use full audio if it's short enough
                plot_orig = original_audio
                plot_synth = synth_audio

            # ==========================================
            # 2. PLOTTING LOGIC
            # ==========================================
            plt.figure(figsize=(14, 9))
            plt.suptitle(title, fontsize=16, fontweight='bold')

            # --- Plot 1: Original Waveform ---
            plt.subplot(2, 2, 1)
            librosa.display.waveshow(plot_orig, sr=sr, alpha=0.8)
            plt.title(f"Original Signal (Time Domain)")
            plt.ylabel("Amplitude")
            plt.xlabel("Time (s)")

            # --- Plot 2: Synthesized Waveform ---
            plt.subplot(2, 2, 2)
            # Color orange to distinguish from original
            librosa.display.waveshow(plot_synth, sr=sr, color='orange', alpha=0.8)
            plt.title(f"Synthesized Signal (Time Domain)")
            plt.ylabel("Amplitude")
            plt.xlabel("Time (s)")

            # --- Plot 3: Original Spectrogram ---
            plt.subplot(2, 2, 3)
            # Compute Short-Time Fourier Transform (STFT)
            D_orig = librosa.stft(plot_orig)
            S_db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
            # Display
            librosa.display.specshow(S_db_orig, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Original Spectrogram (Frequency Domain)")
            plt.ylabel("Frequency (Hz)")

            # --- Plot 4: Synthesized Spectrogram ---
            plt.subplot(2, 2, 4)
            # Compute STFT
            D_synth = librosa.stft(plot_synth)
            S_db_synth = librosa.amplitude_to_db(np.abs(D_synth), ref=np.max)
            # Display
            librosa.display.specshow(S_db_synth, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Synthesized Spectrogram (Frequency Domain)")
            plt.ylabel("Frequency (Hz)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            print("\nDisplaying signal comparison plots...")
            # This makes the window pop up. You must close it for the code to continue.
            plt.show()

        except Exception as e:
            logger.error(f"Failed to visualize audio: {e}")
            print(f"Visualization error (Skipping plot): {e}")

# End of file