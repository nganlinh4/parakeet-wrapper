from onnx_asr import load_model
from pydub import AudioSegment
import os

MODEL_NAME = "istupakov/parakeet-tdt-0.6b-v3-onnx"
model = load_model(MODEL_NAME)

print("Model type:", type(model))

# Convert mp3 to wav
audio = AudioSegment.from_file("data/example-yt_saTD1u8PorI.mp3")
audio = audio.set_channels(1)  # mono
audio.export("temp.wav", format="wav")

# Try recognize
try:
    result = model.recognize("temp.wav")
    print("Type of result:", type(result))
    print("Result:", result)
except Exception as e:
    print("Error:", e)
finally:
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")