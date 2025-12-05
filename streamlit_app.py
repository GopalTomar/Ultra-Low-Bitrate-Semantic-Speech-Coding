"""
Ultra-Low Bitrate Semantic Speech Coding - Streamlit Web Application
============================================================================
Complete interactive web interface for semantic speech compression system

UPDATES:
- Removed 'key' argument from st.audio() to fix MediaMixin TypeError
- Updated dataframe width parameters to be compatible with current Streamlit version
"""

import streamlit as st
import numpy as np
import time
import io
import base64
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import your project modules
from config import Config
from audio_utils import (
    AudioRecorder,
    AudioPreprocessor,
    AudioAnalyzer,
    save_audio,
    load_audio
)
from speech_encoder import SpeechEncoder
from speech_decoder import SpeechDecoder
from compression_engine import TextCompressor, SemanticPacket
from metrics_evaluator import (
    CompressionMetrics,
    TranscriptionMetrics,
    PerformanceMetrics,
    EvaluationReport
)

# Try to import visualizer
try:
    from visualizer import AudioVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Semantic Speech Coding",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
    .audio-player {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.encoder = None
        st.session_state.decoder = None
        st.session_state.preprocessor = None
        st.session_state.compressor = None
        st.session_state.performance = PerformanceMetrics()
        st.session_state.processing_history = []
        st.session_state.current_audio = None
        st.session_state.current_text = None
        st.session_state.synthesized_audio = None
        st.session_state.last_results = None
        st.session_state.result_counter = 0  # For unique keys

initialize_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models(whisper_model="base", tts_engine="gtts"):
    """Load and cache models"""
    Config.WHISPER_MODEL = whisper_model
    Config.TTS_ENGINE = tts_engine

    with st.spinner("Loading models... This may take a minute on first run."):
        encoder = SpeechEncoder(Config)
        decoder = SpeechDecoder(Config)
        preprocessor = AudioPreprocessor(Config)
        compressor = TextCompressor()

    return encoder, decoder, preprocessor, compressor

def audio_to_bytes(audio_array, sample_rate):
    """Convert numpy audio array to bytes for download"""
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()

def get_audio_player_html(audio_bytes):
    """Generate HTML for audio player"""
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'''
        <audio controls class="audio-player">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    '''
    return audio_html

def plot_compression_comparison(results):
    """Create compression comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bitrate comparison
    categories = ['Original\nAudio', 'Semantic\nText']
    bitrates = [
        results['metrics']['original_bitrate_bps'],
        results['metrics']['semantic_bitrate_bps']
    ]

    axes[0].bar(categories, bitrates, color=['#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Bitrate (bps)')
    axes[0].set_title('Bitrate Comparison')
    axes[0].set_yscale('log')
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(bitrates):
        axes[0].text(i, v, f'{v:,.0f}', ha='center', va='bottom')

    # Size comparison
    sizes = [
        results['stages']['preprocessing']['size_bytes'],
        results['stages']['compression']['packet_size_bytes']
    ]

    axes[1].bar(categories, sizes, color=['#e74c3c', '#2ecc71'])
    axes[1].set_ylabel('Size (bytes)')
    axes[1].set_title('Data Size Comparison')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(sizes):
        axes[1].text(i, v, f'{v:,}', ha='center', va='bottom')

    plt.tight_layout()
    return fig

def plot_processing_timeline(results):
    """Create processing timeline visualization"""
    stages = ['Preprocessing', 'Encoding', 'Compression', 'Decompression', 'Decoding']
    times = [
        results['stages']['preprocessing']['time_ms'],
        results['stages']['encoding']['time_ms'],
        results['stages']['compression']['time_ms'],
        results['stages']['decompression']['time_ms'],
        results['stages']['decoding']['time_ms']
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
    bars = ax.barh(stages, times, color=colors, alpha=0.8)

    ax.set_xlabel('Time (ms)')
    ax.set_title('Processing Time by Stage')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{time:.0f} ms', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    return fig

def create_metrics_dataframe(results):
    """Create a dataframe of key metrics"""
    metrics = {
        'Metric': [
            'Audio Duration',
            'Original Size',
            'Compressed Size',
            'Compression Ratio',
            'Original Bitrate',
            'Semantic Bitrate',
            'Bandwidth Reduction',
            'Processing Time',
            'Real-Time Factor'
        ],
        'Value': [
            f"{results['stages']['preprocessing']['duration']:.2f} seconds",
            f"{results['stages']['preprocessing']['size_bytes']:,} bytes",
            f"{results['stages']['compression']['packet_size_bytes']:,} bytes",
            f"{results['metrics']['compression_ratio']:.1f}:1",
            f"{results['metrics']['original_bitrate_bps']:,.0f} bps",
            f"{results['metrics']['semantic_bitrate_bps']:.0f} bps",
            f"{results['metrics']['bandwidth_reduction_percent']:.2f}%",
            f"{results['metrics']['total_processing_time_ms']:.0f} ms",
            f"{results['metrics']['real_time_factor']:.2f}x"
        ]
    }

    return pd.DataFrame(metrics)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic"""

    # Header
    st.markdown('<h1 class="main-header">🎙️ Ultra-Low Bitrate Semantic Speech Coding</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>🚀 Revolutionary Speech Compression System</b><br>
        Achieve compression ratios exceeding <b>10,000:1</b> by converting speech to semantic text tokens 
        instead of preserving acoustic waveforms.
    </div>
    """, unsafe_allow_html=True)

    # ============================================================================
    # SIDEBAR - CONFIGURATION
    # ============================================================================

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        st.subheader("Model Settings")
        whisper_model = st.selectbox(
            "Whisper Model (ASR)",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )

        tts_engine = st.selectbox(
            "TTS Engine",
            options=["gtts", "pyttsx3"],
            index=0,
            help="gTTS requires internet, pyttsx3 is offline"
        )

        # Audio settings
        st.subheader("Audio Settings")
        vad_mode = st.slider(
            "VAD Aggressiveness",
            min_value=0,
            max_value=3,
            value=3,
            help="Higher = more aggressive silence removal"
        )

        noise_reduction = st.checkbox(
            "Enable Noise Reduction",
            value=True
        )

        # Processing options
        st.subheader("Processing Options")
        show_plots = st.checkbox(
            "Show Visualizations",
            value=True
        )

        auto_play = st.checkbox(
            "Auto-play Synthesized Audio",
            value=True
        )

        # Load models button
        if st.button("🔄 Load/Reload Models"):
            st.session_state.encoder, st.session_state.decoder, \
            st.session_state.preprocessor, st.session_state.compressor = \
                load_models(whisper_model, tts_engine)
            st.success("✅ Models loaded successfully!")

        # System info
        st.subheader("System Info")
        device = Config.get_device()
        st.info(f"**Device:** {device.upper()}")
        st.info(f"**Sample Rate:** {Config.SAMPLE_RATE} Hz")

    # ============================================================================
    # MAIN TABS
    # ============================================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎤 Record & Process",
        "📁 Upload File",
        "📊 Batch Processing",
        "📈 Analytics",
        "ℹ️ About"
    ])

    # ============================================================================
    # TAB 1: RECORD & PROCESS
    # ============================================================================

    with tab1:
        st.header("🎤 Record and Process Audio")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Recording Options")

            record_mode = st.radio(
                "Recording Mode",
                options=["Fixed Duration", "Voice Activity Detection"],
                help="VAD stops automatically when you stop speaking"
            )

            if record_mode == "Fixed Duration":
                duration = st.slider(
                    "Recording Duration (seconds)",
                    min_value=1,
                    max_value=30,
                    value=5
                )
            else:
                duration = None
                st.info("🎙️ Recording will stop automatically after you pause speaking")

        with col2:
            st.subheader("Quick Actions")

            if st.button("🔴 Start Recording", type="primary", key="btn_record"):
                if st.session_state.encoder is None:
                    st.error("❌ Please load models first (sidebar)")
                else:
                    record_and_process(duration, show_plots, auto_play)

            if st.button("🔊 Play Original", key="btn_play_orig"):
                if st.session_state.current_audio is not None:
                    audio_bytes = audio_to_bytes(
                        st.session_state.current_audio,
                        Config.SAMPLE_RATE
                    )
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning("No audio recorded yet")

            if st.button("🔊 Play Synthesized", key="btn_play_synth"):
                if st.session_state.synthesized_audio is not None:
                    audio_bytes = audio_to_bytes(
                        st.session_state.synthesized_audio,
                        Config.SAMPLE_RATE
                    )
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning("No synthesized audio yet")

        # Display results
        if st.session_state.last_results is not None:
            display_results(st.session_state.last_results, context="tab1")

    # ============================================================================
    # TAB 2: UPLOAD FILE
    # ============================================================================

    with tab2:
        st.header("📁 Upload Audio File")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            help="Upload an audio file to process"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Config.TEMP_DIR / f"uploaded_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Optional: Reference text for WER calculation
            reference_text = st.text_area(
                "Reference Transcription (Optional)",
                help="Provide ground truth text for accuracy calculation"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🚀 Process File", type="primary", key="btn_process_file"):
                    if st.session_state.encoder is None:
                        st.error("❌ Please load models first (sidebar)")
                    else:
                        process_file(temp_path, reference_text, show_plots, auto_play)

            with col2:
                if st.button("🎧 Play Uploaded File", key="btn_play_uploaded"):
                    st.audio(str(temp_path), format="audio/wav")

        # Display results
        if st.session_state.last_results is not None:
            display_results(st.session_state.last_results, context="tab2")

    # ============================================================================
    # TAB 3: BATCH PROCESSING
    # ============================================================================

    with tab3:
        st.header("📊 Batch Processing")

        st.markdown("""
        Process multiple audio files at once and generate comprehensive statistics.
        """)

        # File upload for batch
        uploaded_files = st.file_uploader(
            "Upload Multiple Audio Files",
            type=['wav', 'mp3', 'ogg', 'flac'],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files ready for processing")

            if st.button("🚀 Process Batch", type="primary", key="btn_batch_process"):
                if st.session_state.encoder is None:
                    st.error("❌ Please load models first (sidebar)")
                else:
                    process_batch(uploaded_files, show_plots=False)
        else:
            st.info("👆 Upload multiple files to enable batch processing")

    # ============================================================================
    # TAB 4: ANALYTICS
    # ============================================================================

    with tab4:
        st.header("📈 System Analytics")

        if st.session_state.processing_history:
            st.subheader("Processing History")

            # Create summary statistics
            total_files = len(st.session_state.processing_history)

            avg_compression = np.mean([
                r['metrics']['compression_ratio']
                for r in st.session_state.processing_history
            ])

            avg_rtf = np.mean([
                r['metrics']['real_time_factor']
                for r in st.session_state.processing_history
            ])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Files Processed", total_files)

            with col2:
                st.metric("Avg Compression Ratio", f"{avg_compression:.1f}:1")

            with col3:
                rtf_color = "🟢" if avg_rtf < 1.0 else "🔴"
                st.metric("Avg Real-Time Factor", f"{avg_rtf:.2f}x {rtf_color}")

            # History table
            st.subheader("Detailed History")
            history_data = []

            for i, result in enumerate(st.session_state.processing_history):
                history_data.append({
                    'ID': i + 1,
                    'Text': result['stages']['encoding']['text'][:50] + '...',
                    'Duration (s)': f"{result['stages']['preprocessing']['duration']:.2f}",
                    'Compression': f"{result['metrics']['compression_ratio']:.1f}:1",
                    'RTF': f"{result['metrics']['real_time_factor']:.2f}x"
                })

            df_history = pd.DataFrame(history_data)
            # Use columns directly or basic st.dataframe to avoid version conflicts
            st.dataframe(df_history)

            # Visualization
            st.subheader("Performance Trends")

            compression_ratios = [
                r['metrics']['compression_ratio']
                for r in st.session_state.processing_history
            ]

            rtf_values = [
                r['metrics']['real_time_factor']
                for r in st.session_state.processing_history
            ]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(compression_ratios, marker='o', linewidth=2)
            axes[0].set_xlabel('File Number')
            axes[0].set_ylabel('Compression Ratio')
            axes[0].set_title('Compression Ratio Over Time')
            axes[0].grid(alpha=0.3)

            axes[1].plot(rtf_values, marker='s', linewidth=2, color='orange')
            axes[1].axhline(y=1.0, color='r', linestyle='--', label='Real-time threshold')
            axes[1].set_xlabel('File Number')
            axes[1].set_ylabel('Real-Time Factor')
            axes[1].set_title('Processing Speed Over Time')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Download report
            if st.button("📥 Download Full Report", key="btn_download_report"):
                generate_and_download_report()

        else:
            st.info("📊 Process some audio files to see analytics")

    # ============================================================================
    # TAB 5: ABOUT
    # ============================================================================

    with tab5:
        st.header("ℹ️ About This Project")

        st.markdown("""
        ### 🎯 Ultra-Low Bitrate Semantic Speech Coding

        This application demonstrates a revolutionary approach to speech compression that achieves 
        **compression ratios exceeding 10,000:1** by:

        1. **Converting speech to text** (Semantic Extraction)
        2. **Compressing the text** (Entropy Coding)
        3. **Transmitting semantic tokens** (Ultra-low bandwidth)
        4. **Reconstructing speech** (Text-to-Speech Synthesis)

        ### 📊 Key Features

        - ✅ Real-time audio recording with Voice Activity Detection
        - ✅ State-of-the-art ASR using OpenAI Whisper
        - ✅ Multiple TTS engines (gTTS, pyttsx3, Coqui)
        - ✅ Comprehensive metrics (WER, compression ratios, latency)
        - ✅ Interactive visualizations
        - ✅ Batch processing capabilities
        - ✅ Network-ready architecture

        ### 🔬 Technical Details

        **Compression Pipeline:**
        ```
        Audio → Preprocessing → ASR → Text Compression → Transmission
        ```

        **Decompression Pipeline:**
        ```
        Reception → Text Decompression → TTS → Audio Playback
        ```

        **Typical Performance:**
        - Original: ~512,000 bps (16kHz, 32-bit float)
        - Semantic: ~100 bps (compressed text)
        - Compression Ratio: **~5,120:1**
        - Processing: **Faster than real-time**

        ### 📚 Academic Context

        This project demonstrates key Signal Processing concepts:
        - Speech-to-Text and Text-to-Speech
        - Semantic vector quantization
        - Lossy compression trade-offs
        - Real-time processing constraints

        ### ⚠️ Trade-offs

        **Preserved:**
        - ✅ Linguistic meaning
        - ✅ Message intelligibility
        - ✅ Language content

        **Lost:**
        - ❌ Speaker identity (voice characteristics)
        - ❌ Emotional prosody
        - ❌ Background sounds
        - ❌ Acoustic details

        ### 🛠️ Technology Stack

        - **Python** - Core language
        - **Streamlit** - Web interface
        - **OpenAI Whisper** - Speech recognition
        - **gTTS/pyttsx3** - Speech synthesis
        - **librosa** - Audio processing
        - **matplotlib** - Visualizations

        ### 👨‍💻 Author

        **Your Name**  
        Signal Processing Course Project  
        [Your University]  
        2024

        ### 📄 License

        MIT License - Open source educational project

        ---

        **GitHub Repository:** [Link to your repo]
        """)

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def record_and_process(duration, show_plots, auto_play):
    """Record audio and process through complete pipeline"""

    with st.spinner("🎙️ Recording audio..."):
        try:
            recorder = AudioRecorder(Config)
            audio = recorder.record_audio(duration)

            if len(audio) == 0:
                st.error("❌ No audio recorded. Please check your microphone.")
                return

            st.session_state.current_audio = audio
            st.success("✅ Recording complete!")

        except Exception as e:
            st.error(f"❌ Recording failed: {e}")
            return

    # Process the audio
    process_audio_pipeline(audio, None, show_plots, auto_play)

def process_file(file_path, reference_text, show_plots, auto_play):
    """Process an uploaded audio file"""

    with st.spinner("📂 Loading audio file..."):
        try:
            audio, sr = load_audio(str(file_path), Config.SAMPLE_RATE)
            st.session_state.current_audio = audio
            st.success("✅ File loaded successfully!")

        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            return

    # Process the audio
    ref_text = reference_text if reference_text.strip() else None
    process_audio_pipeline(audio, ref_text, show_plots, auto_play)

def process_audio_pipeline(audio, reference_text, show_plots, auto_play):
    """Complete audio processing pipeline"""

    # Increment result counter for unique keys
    st.session_state.result_counter += 1

    # Get components from session state
    encoder = st.session_state.encoder
    decoder = st.session_state.decoder
    preprocessor = st.session_state.preprocessor
    compressor = st.session_state.compressor

    results = {
        'success': False,
        'stages': {},
        'metrics': {}
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Stage 1: Preprocessing
        status_text.text("⏳ Stage 1/5: Preprocessing...")
        progress_bar.progress(20)

        t_start = time.time()
        clean_audio = preprocessor.preprocess(audio)
        t_preprocess = time.time() - t_start

        audio_duration = len(clean_audio) / Config.SAMPLE_RATE
        audio_size = clean_audio.nbytes
        audio_bitrate = (audio_size * 8) / audio_duration if audio_duration > 0 else 0

        results['stages']['preprocessing'] = {
            'duration': audio_duration,
            'size_bytes': audio_size,
            'bitrate_bps': audio_bitrate,
            'time_ms': t_preprocess * 1000
        }

        # Stage 2: Encoding (ASR)
        status_text.text("⏳ Stage 2/5: Transcribing (ASR)...")
        progress_bar.progress(40)

        t_start = time.time()
        transcription = encoder.encode(clean_audio)
        t_encode = time.time() - t_start

        text = transcription['text']
        st.session_state.current_text = text

        results['stages']['encoding'] = {
            'text': text,
            'language': transcription['language'],
            'time_ms': t_encode * 1000
        }

        # Stage 3: Compression
        status_text.text("⏳ Stage 3/5: Compressing...")
        progress_bar.progress(60)

        t_start = time.time()
        packet = SemanticPacket(
            text=text,
            speaker_id="streamlit_user",
            language=transcription['language']
        )
        packed_packet = packet.pack(compressor)
        t_compress = time.time() - t_start

        text_size = len(text.encode('utf-8'))
        packet_size = len(packed_packet)
        text_bitrate = (text_size * 8) / audio_duration if audio_duration > 0 else 0

        compression_ratio = CompressionMetrics.calculate_compression_ratio(
            audio_size, packet_size
        )

        results['stages']['compression'] = {
            'text_size_bytes': text_size,
            'packet_size_bytes': packet_size,
            'text_bitrate_bps': text_bitrate,
            'compression_ratio': compression_ratio,
            'time_ms': t_compress * 1000
        }

        # Stage 4: Decompression
        status_text.text("⏳ Stage 4/5: Decompressing...")
        progress_bar.progress(80)

        t_start = time.time()
        unpacked_packet = SemanticPacket.unpack(packed_packet, compressor)
        t_decompress = time.time() - t_start

        results['stages']['decompression'] = {
            'text': unpacked_packet.text,
            'time_ms': t_decompress * 1000
        }

        # Stage 5: Synthesis (TTS)
        status_text.text("⏳ Stage 5/5: Synthesizing (TTS)...")
        progress_bar.progress(90)

        t_start = time.time()
        synthesized_audio = decoder.decode(unpacked_packet.text)
        t_decode = time.time() - t_start

        st.session_state.synthesized_audio = synthesized_audio

        synth_duration = len(synthesized_audio) / Config.SAMPLE_RATE

        results['stages']['decoding'] = {
            'audio_duration': synth_duration,
            'time_ms': t_decode * 1000
        }

        # Calculate overall metrics
        t_total = t_preprocess + t_encode + t_compress + t_decompress + t_decode
        rtf = t_total / audio_duration if audio_duration > 0 else 0

        bandwidth_reduction = CompressionMetrics.calculate_bandwidth_reduction(
            audio_bitrate, text_bitrate
        )

        results['metrics'] = {
            'total_processing_time_ms': t_total * 1000,
            'real_time_factor': rtf,
            'compression_ratio': compression_ratio,
            'bandwidth_reduction_percent': bandwidth_reduction,
            'original_bitrate_bps': audio_bitrate,
            'semantic_bitrate_bps': text_bitrate
        }

        # Calculate WER if reference provided
        if reference_text:
            wer_score = TranscriptionMetrics.calculate_wer(reference_text, text)
            accuracy = TranscriptionMetrics.calculate_accuracy(reference_text, text)
            results['metrics']['wer'] = wer_score
            results['metrics']['accuracy'] = accuracy

        results['original_audio'] = clean_audio
        results['synthesized_audio'] = synthesized_audio
        results['success'] = True

        # Store results
        st.session_state.last_results = results
        st.session_state.processing_history.append(results)

        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")

        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Auto-play if enabled
        if auto_play and len(synthesized_audio) > 0:
            st.success("🔊 Playing synthesized audio...")
            audio_bytes = audio_to_bytes(synthesized_audio, Config.SAMPLE_RATE)
            st.audio(audio_bytes, format="audio/wav")

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()

def process_batch(uploaded_files, show_plots=False):
    """Process multiple files in batch"""

    st.subheader("📊 Batch Processing Results")

    results_list = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")

        # Save file temporarily
        temp_path = Config.TEMP_DIR / f"batch_{i}_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load audio
            audio, sr = load_audio(str(temp_path), Config.SAMPLE_RATE)

            # Process
            encoder = st.session_state.encoder
            decoder = st.session_state.decoder
            preprocessor = st.session_state.preprocessor
            compressor = st.session_state.compressor

            # Quick processing
            clean_audio = preprocessor.preprocess(audio)
            transcription = encoder.encode(clean_audio)

            packet = SemanticPacket(
                text=transcription['text'],
                speaker_id=f"batch_user_{i}",
                language=transcription['language']
            )
            packed_packet = packet.pack(compressor)

            # Calculate metrics
            audio_size = clean_audio.nbytes
            packet_size = len(packed_packet)
            audio_duration = len(clean_audio) / Config.SAMPLE_RATE

            compression_ratio = CompressionMetrics.calculate_compression_ratio(
                audio_size, packet_size
            )

            results_list.append({
                'filename': uploaded_file.name,
                'text': transcription['text'],
                'duration': audio_duration,
                'compression_ratio': compression_ratio,
                'original_size': audio_size,
                'compressed_size': packet_size
            })

        except Exception as e:
            st.warning(f"⚠️ Failed to process {uploaded_file.name}: {e}")

        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()
    status_text.empty()

    # Display results table
    if results_list:
        st.success(f"✅ Processed {len(results_list)} files successfully!")

        df = pd.DataFrame(results_list)
        # Fix: Default use to handle version differences
        st.dataframe(df)

        # Summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_ratio = df['compression_ratio'].mean()
            st.metric("Average Compression", f"{avg_ratio:.1f}:1")

        with col2:
            total_orig = df['original_size'].sum()
            st.metric("Total Original Size", f"{total_orig / 1024:.1f} KB")

        with col3:
            total_comp = df['compressed_size'].sum()
            st.metric("Total Compressed Size", f"{total_comp / 1024:.1f} KB")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))

        x = range(len(results_list))
        width = 0.35

        ax.bar([i - width/2 for i in x], df['original_size'], width,
               label='Original', alpha=0.8, color='#e74c3c')
        ax.bar([i + width/2 for i in x], df['compressed_size'], width,
               label='Compressed', alpha=0.8, color='#2ecc71')

        ax.set_ylabel('Size (bytes)')
        ax.set_title('File Size Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"File {i+1}" for i in x], rotation=45)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

def display_results(results, context="default"):
    """Display processing results (KEY PARAM REMOVED FROM ST.AUDIO)"""

    st.markdown("---")
    st.subheader("📊 Processing Results")

    # Success banner
    if results['success']:
        st.markdown("""
        <div class="success-box">
            <h3>✅ Processing Successful!</h3>
        </div>
        """, unsafe_allow_html=True)

    # Transcribed text
    st.markdown("### 📝 Transcribed Text")
    st.info(f"**{results['stages']['encoding']['text']}**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Language Detected:**")
        st.write(results['stages']['encoding']['language'].upper())

    with col2:
        st.markdown("**Audio Duration:**")
        st.write(f"{results['stages']['preprocessing']['duration']:.2f} seconds")

    # Key metrics
    st.markdown("### 🎯 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ratio = results['metrics']['compression_ratio']
        st.metric("Compression Ratio", f"{ratio:.1f}:1",
                  delta=f"{ratio-1:.0f}x better")

    with col2:
        reduction = results['metrics']['bandwidth_reduction_percent']
        st.metric("Bandwidth Reduction", f"{reduction:.2f}%")

    with col3:
        rtf = results['metrics']['real_time_factor']
        rtf_delta = "Faster" if rtf < 1.0 else "Slower"
        st.metric("Real-Time Factor", f"{rtf:.2f}x",
                  delta=rtf_delta, delta_color="inverse")

    with col4:
        proc_time = results['metrics']['total_processing_time_ms']
        st.metric("Processing Time", f"{proc_time:.0f} ms")

    # Detailed metrics table
    with st.expander("📋 Detailed Metrics"):
        df_metrics = create_metrics_dataframe(results)
        st.table(df_metrics)

    # Visualizations
    if 'original_audio' in results and 'synthesized_audio' in results:
        st.markdown("### 📊 Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.pyplot(plot_compression_comparison(results))

        with viz_col2:
            st.pyplot(plot_processing_timeline(results))

        # Comparison with standard codecs
        with st.expander("🔍 Comparison with Standard Codecs"):
            comparisons = CompressionMetrics.compare_with_standards(
                results['metrics']['original_bitrate_bps'],
                results['metrics']['semantic_bitrate_bps']
            )

            comp_data = []
            for codec_name, data in comparisons.items():
                comp_data.append({
                    'Codec': codec_name,
                    'Bitrate (bps)': f"{data['bitrate_bps']:,}",
                    'vs Semantic': f"{data['ratio_vs_semantic']:.1f}x",
                    'Savings': f"{data['savings_percent']:.1f}%"
                })

            df_comp = pd.DataFrame(comp_data)
            # Fix: Default dataframe
            st.dataframe(df_comp)

    # Audio players
    st.markdown("### 🔊 Audio Comparison")

    audio_col1, audio_col2 = st.columns(2)

    with audio_col1:
        st.markdown("**Original Audio**")
        if st.session_state.current_audio is not None:
            audio_bytes = audio_to_bytes(
                st.session_state.current_audio,
                Config.SAMPLE_RATE
            )
            # FIXED: Removed 'key' argument
            st.audio(audio_bytes, format="audio/wav")

    with audio_col2:
        st.markdown("**Synthesized Audio**")
        if st.session_state.synthesized_audio is not None:
            audio_bytes = audio_to_bytes(
                st.session_state.synthesized_audio,
                Config.SAMPLE_RATE
            )
            # FIXED: Removed 'key' argument
            st.audio(audio_bytes, format="audio/wav")

    # Download buttons
    st.markdown("### 💾 Download Options")

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:
        if st.session_state.current_audio is not None:
            audio_bytes = audio_to_bytes(
                st.session_state.current_audio,
                Config.SAMPLE_RATE
            )
            st.download_button(
                label="📥 Download Original Audio",
                data=audio_bytes,
                file_name="original_audio.wav",
                mime="audio/wav",
                key=f"dl_orig_{context}_{st.session_state.result_counter}"
            )

    with download_col2:
        if st.session_state.synthesized_audio is not None:
            audio_bytes = audio_to_bytes(
                st.session_state.synthesized_audio,
                Config.SAMPLE_RATE
            )
            st.download_button(
                label="📥 Download Synthesized Audio",
                data=audio_bytes,
                file_name="synthesized_audio.wav",
                mime="audio/wav",
                key=f"dl_synth_{context}_{st.session_state.result_counter}"
            )

    with download_col3:
        # Download transcription
        text = results['stages']['encoding']['text']
        st.download_button(
            label="📥 Download Transcription",
            data=text,
            file_name="transcription.txt",
            mime="text/plain",
            key=f"dl_text_{context}_{st.session_state.result_counter}"
        )

def generate_and_download_report():
    """Generate comprehensive report and offer download"""

    report = EvaluationReport()

    # Add all results
    for result in st.session_state.processing_history:
        report.add_compression_metrics({
            'compression_ratio': result['metrics']['compression_ratio'],
            'audio_bitrate_bps': result['metrics']['original_bitrate_bps'],
            'text_bitrate_bps': result['metrics']['semantic_bitrate_bps'],
            'bandwidth_reduction_percent': result['metrics']['bandwidth_reduction_percent']
        })

    # Generate markdown
    md_content = report.generate_markdown_report()

    # Generate JSON
    json_content = json.dumps(report.report_data, indent=2)

    col1, col2 = st.columns(2)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with col1:
        st.download_button(
            label="📥 Download Markdown Report",
            data=md_content,
            file_name=f"report_{timestamp}.md",
            mime="text/markdown",
            key=f"dl_report_md_{timestamp}"
        )

    with col2:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_content,
            file_name=f"report_{timestamp}.json",
            mime="application/json",
            key=f"dl_report_json_{timestamp}"
        )

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()