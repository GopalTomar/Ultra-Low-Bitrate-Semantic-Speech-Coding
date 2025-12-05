import sounddevice as sd
import soundfile as sf
import whisper

print("1. Testing Microphone... Say something for 3 seconds.")
fs = 16000
myrecording = sd.rec(int(3 * fs), samplerate=fs, channels=1)
sd.wait()
sf.write('test_audio.wav', myrecording, fs)
print("   Audio saved to test_audio.wav")

print("2. Testing Whisper Model Load (this downloads the model once)...")
model = whisper.load_model("base")
print("   Model loaded successfully.")

print("3. Setup Complete!")